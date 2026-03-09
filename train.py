"""
Two-phase BioCLIP fine-tuning for French bird species classification.

Phase 1 — Head only (encoder frozen):
  Quickly trains the linear classifier on top of frozen BioCLIP features.

Phase 2 — Partial fine-tuning (last N ViT blocks + head):
  Adapts visual representations to fine-grained bird imagery.

Improvements over baseline:
  - MixUp / CutMix (batch-level, 50/50)
  - WeightedRandomSampler (oversamples rare species)
  - Layer-wise LR decay for encoder blocks
  - Linear warmup + cosine annealing
  - Early stopping on val Top-1
  - Gradient clipping (max_norm=1.0)
  - Taxonomy-aware auxiliary loss (genus + family heads)
  - Confusion-aware hard-negative mining (phase 2, boosts confused species)

Usage:
    python train.py
    python train.py --batch-size 128 --phase1-epochs 5 --phase2-epochs 30
    python train.py --resume checkpoints/best_model.pt
    python train.py --no-mixup --patience 10
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy, F1Score
from torchvision.transforms import v2 as T
from tqdm import tqdm

from src.dataset import BirdDataset, create_splits, get_transforms
from src.model import BirdClassifier

SPLITS_DIR = Path("data/splits")
IMAGE_DIR = Path("data/images")
INDEX_PATH = Path("data/image_index.parquet")
CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_loaders(
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build data loaders with weighted sampling for class balance."""
    label_map_path = SPLITS_DIR / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)

    train_ds = BirdDataset(SPLITS_DIR / "train.parquet", IMAGE_DIR, get_transforms("train"), label_map)
    val_ds   = BirdDataset(SPLITS_DIR / "val.parquet",   IMAGE_DIR, get_transforms("val"),   label_map)
    test_ds  = BirdDataset(SPLITS_DIR / "test.parquet",  IMAGE_DIR, get_transforms("val"),   label_map)

    # Weighted sampler: oversample rare species so each species is equally likely
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    print(f"  WeightedRandomSampler active ({len(train_ds):,} samples)")

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, sampler=sampler, **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


def build_mixup_cutmix(num_classes: int) -> T.RandomChoice:
    """Create a batch-level MixUp/CutMix transform (50/50)."""
    return T.RandomChoice([
        T.MixUp(alpha=0.2, num_classes=num_classes),
        T.CutMix(alpha=1.0, num_classes=num_classes),
    ])


def run_epoch(
    model: BirdClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    top1: Accuracy,
    top5: Accuracy,
    mixup_fn=None,
    num_genera: int = 0,
    num_families: int = 0,
) -> float:
    """Run one epoch. If optimizer is None, runs in eval mode (no gradients)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    has_taxonomy = num_genera > 0 and num_families > 0

    genus_criterion = nn.CrossEntropyLoss(label_smoothing=0.1) if has_taxonomy else None
    family_criterion = nn.CrossEntropyLoss(label_smoothing=0.1) if has_taxonomy else None

    top1.reset()
    top5.reset()

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            # Unpack: either (images, species) or (images, species, genus, family)
            if has_taxonomy:
                images, species_labels, genus_labels, family_labels = batch
                genus_labels = genus_labels.to(device)
                family_labels = family_labels.to(device)
            else:
                images, species_labels = batch

            images = images.to(device)
            species_labels = species_labels.to(device)

            # MixUp / CutMix (training only, on species labels)
            using_mixup = False
            if training and mixup_fn is not None:
                images, species_labels = mixup_fn(images, species_labels)
                using_mixup = True

            with torch.cuda.amp.autocast():
                output = model(images)

                if training and has_taxonomy:
                    species_logits, genus_logits, family_logits = output
                else:
                    species_logits = output

                # Species loss: soft targets when MixUp active
                if using_mixup:
                    species_loss = F.cross_entropy(species_logits, species_labels, label_smoothing=0.1)
                else:
                    species_loss = criterion(species_logits, species_labels)

                loss = species_loss

                # Taxonomy auxiliary losses (no MixUp on genus/family — hard labels)
                if training and has_taxonomy:
                    loss = loss + 0.3 * genus_criterion(genus_logits, genus_labels)
                    loss = loss + 0.1 * family_criterion(family_logits, family_labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)

            # Metrics on species only, with hard labels
            if using_mixup:
                # MixUp produces one-hot-like soft labels; take argmax for metrics
                hard_labels = species_labels.argmax(dim=1)
            else:
                hard_labels = species_labels
            top1.update(species_logits, hard_labels)
            top5.update(species_logits, hard_labels)

    n = len(loader.dataset)
    return total_loss / n


def compute_confusion_boost(
    model: BirdClassifier,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    boost_factor: float = 3.0,
) -> dict[int, float]:
    """
    Build confusion matrix on val set, return per-class sampling boost.

    Confused species (high error rate) get higher boost → more training samples.
    Boost formula: 1 + boost_factor * error_rate
    """
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc="confusion"):
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                logits = model(images)
            preds = logits.argmax(dim=1)
            for t, p in zip(labels.cpu(), preds.cpu()):
                confusion[t.item(), p.item()] += 1

    boost: dict[int, float] = {}
    for i in range(num_classes):
        total = confusion[i].sum().item()
        if total == 0:
            boost[i] = 1.0
            continue
        error_rate = 1.0 - confusion[i, i].item() / total
        boost[i] = 1.0 + boost_factor * error_rate
    return boost


def update_sampler_weights(train_loader: DataLoader, boost: dict[int, float]) -> None:
    """Multiply base class-balance weights by confusion boost."""
    ds = train_loader.dataset
    sampler = train_loader.sampler
    base_counts = ds.get_class_counts()
    labels = [ds.label_map[name] for name in ds.df["scientificName"]]
    new_weights = torch.tensor(
        [boost.get(label, 1.0) / base_counts[label] for label in labels],
        dtype=torch.float64,
    )
    sampler.weights = new_weights


def train_phase(
    phase: int,
    model: BirdClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    head_lr: float,
    encoder_lr: float,
    num_classes: int,
    device: torch.device,
    layer_decay: float = 1.0,
    patience: int = 7,
    mixup_fn=None,
    num_genera: int = 0,
    num_families: int = 0,
    hard_mining: bool = False,
) -> float:
    """Train one phase. Returns the best val Top-1 accuracy achieved."""
    print(f"\n{'=' * 60}")
    print(f"Phase {phase}: {'head only' if phase == 1 else 'partial fine-tuning'}")
    print(f"  epochs={num_epochs}, patience={patience}, layer_decay={layer_decay}")
    print(f"  head_lr={head_lr}, encoder_lr={encoder_lr}")
    print(f"  mixup={'ON' if mixup_fn else 'OFF'}")
    if num_genera > 0:
        print(f"  taxonomy heads: {num_genera} genera, {num_families} families")
    print(f"{'=' * 60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.get_param_groups(head_lr=head_lr, encoder_lr=encoder_lr, layer_decay=layer_decay),
        weight_decay=1e-4,
    )

    # Linear warmup (2 epochs) + cosine annealing for the rest
    warmup_epochs = min(2, num_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs - warmup_epochs, 1),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
    )
    scaler = torch.cuda.amp.GradScaler()

    top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    best_val_top1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, top1, top5,
            mixup_fn=mixup_fn, num_genera=num_genera, num_families=num_families,
        )
        train_top1 = top1.compute().item()
        train_top5 = top5.compute().item()

        val_loss = run_epoch(
            model, val_loader, criterion, None, scaler, device, top1, top5,
            num_genera=num_genera, num_families=num_families,
        )
        val_top1 = top1.compute().item()
        val_top5 = top5.compute().item()

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:>2}/{num_epochs} | "
            f"lr {current_lr:.2e} | "
            f"train loss {train_loss:.4f} top1 {train_top1:.3f} top5 {train_top5:.3f} | "
            f"val loss {val_loss:.4f} top1 {val_top1:.3f} top5 {val_top5:.3f}"
        )

        # Hard-negative mining: boost confused species in sampler
        if hard_mining and epoch >= warmup_epochs:
            boost = compute_confusion_boost(model, val_loader, num_classes, device)
            update_sampler_weights(train_loader, boost)
            n_boosted = sum(1 for b in boost.values() if b > 1.5)
            if n_boosted:
                print(f"    Hard-mining: {n_boosted} confused species boosted (max {max(boost.values()):.1f}x)")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            epochs_without_improvement = 0
            model.save(BEST_MODEL_PATH)
            print(f"    -> New best val Top-1: {best_val_top1:.3f}  (saved)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"    Early stopping: no improvement for {patience} epochs")
                break

    return best_val_top1


def evaluate(model: BirdClassifier, test_loader: DataLoader, num_classes: int, device: torch.device) -> None:
    """Evaluate the best model on the test set and print a summary."""
    print(f"\n{'=' * 60}")
    print("Test set evaluation")
    print(f"{'=' * 60}")

    top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)
    f1   = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    model.eval()
    top1.reset(); top5.reset(); f1.reset()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            # Handle both (img, label) and (img, species, genus, family)
            images, species_labels = batch[0], batch[1]
            images, species_labels = images.to(device), species_labels.to(device)
            with torch.cuda.amp.autocast():
                logits = model(images)
            top1.update(logits, species_labels)
            top5.update(logits, species_labels)
            f1.update(logits, species_labels)

    print(f"  Top-1 accuracy : {top1.compute().item():.4f}")
    print(f"  Top-5 accuracy : {top5.compute().item():.4f}")
    print(f"  Macro F1 score : {f1.compute().item():.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train BioCLIP bird classifier")
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--phase1-epochs",  type=int,   default=5)
    parser.add_argument("--phase2-epochs",  type=int,   default=30)
    parser.add_argument("--phase2-blocks",  type=int,   default=6,     help="ViT blocks to unfreeze in phase 2")
    parser.add_argument("--head-lr",        type=float, default=1e-3)
    parser.add_argument("--encoder-lr",     type=float, default=1e-5)
    parser.add_argument("--layer-decay",    type=float, default=0.75,  help="Layer-wise LR decay factor")
    parser.add_argument("--patience",       type=int,   default=7,     help="Early stopping patience (epochs)")
    parser.add_argument("--min-images",     type=int,   default=20,    help="Min images per species (drop if fewer)")
    parser.add_argument("--no-mixup",       action="store_true",       help="Disable MixUp/CutMix")
    parser.add_argument("--no-hard-mining", action="store_true",       help="Disable confusion-aware hard-negative mining")
    parser.add_argument("--resume",         type=str,   default=None,  help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure splits exist
    if not (SPLITS_DIR / "train.parquet").exists():
        print("Splits not found — creating them now...")
        create_splits(index_path=INDEX_PATH, splits_dir=SPLITS_DIR, min_images=args.min_images)

    # Load label maps
    with open(SPLITS_DIR / "label_map.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"Number of classes: {num_classes}")

    # Load taxonomy maps
    num_genera, num_families = 0, 0
    genus_map_path = SPLITS_DIR / "genus_map.json"
    family_map_path = SPLITS_DIR / "family_map.json"
    if genus_map_path.exists() and family_map_path.exists():
        with open(genus_map_path) as f:
            num_genera = len(json.load(f))
        with open(family_map_path) as f:
            num_families = len(json.load(f))
        print(f"Taxonomy: {num_genera} genera, {num_families} families")

    train_loader, val_loader, test_loader = make_loaders(args.batch_size, args.num_workers)

    # MixUp / CutMix
    mixup_fn = None if args.no_mixup else build_mixup_cutmix(num_classes)

    # Build or restore model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = BirdClassifier.load(
            Path(args.resume), num_classes=num_classes,
            num_genera=num_genera, num_families=num_families,
        )
    else:
        model = BirdClassifier(
            num_classes=num_classes, freeze_encoder=True,
            num_genera=num_genera, num_families=num_families,
        )
    model = model.to(device)

    # Phase 1 — head only
    if args.phase1_epochs > 0:
        train_phase(
            phase=1,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.phase1_epochs,
            head_lr=args.head_lr,
            encoder_lr=0.0,
            num_classes=num_classes,
            device=device,
            patience=args.patience,
            mixup_fn=mixup_fn,
            num_genera=num_genera,
            num_families=num_families,
        )

    # Phase 2 — partial fine-tuning
    if args.phase2_epochs > 0:
        model.unfreeze_last_n_blocks(n=args.phase2_blocks)
        train_phase(
            phase=2,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.phase2_epochs,
            head_lr=args.head_lr,
            encoder_lr=args.encoder_lr,
            num_classes=num_classes,
            device=device,
            layer_decay=args.layer_decay,
            patience=args.patience,
            mixup_fn=mixup_fn,
            num_genera=num_genera,
            num_families=num_families,
            hard_mining=not args.no_hard_mining,
        )

    # Final evaluation on test set using best checkpoint
    print(f"\nLoading best model from {BEST_MODEL_PATH} for test evaluation...")
    best_model = BirdClassifier.load(BEST_MODEL_PATH, num_classes=num_classes).to(device)
    evaluate(best_model, test_loader, num_classes, device)


if __name__ == "__main__":
    main()
