"""
Two-phase BioCLIP fine-tuning for French bird species classification.

Phase 1 — Head only (encoder frozen):
  Quickly trains the linear classifier on top of frozen BioCLIP features.

Phase 2 — Partial fine-tuning (last 4 ViT blocks + head):
  Adapts visual representations to fine-grained bird imagery.

Usage:
    python train.py
    python train.py --batch-size 128 --phase1-epochs 5 --phase2-epochs 20
    python train.py --resume checkpoints/best_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
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

def make_loaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    label_map_path = SPLITS_DIR / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)

    train_ds = BirdDataset(SPLITS_DIR / "train.parquet", IMAGE_DIR, get_transforms("train"), label_map)
    val_ds   = BirdDataset(SPLITS_DIR / "val.parquet",   IMAGE_DIR, get_transforms("val"),   label_map)
    test_ds  = BirdDataset(SPLITS_DIR / "test.parquet",  IMAGE_DIR, get_transforms("val"),   label_map)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


def run_epoch(
    model: BirdClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    top1: Accuracy,
    top5: Accuracy,
) -> float:
    """Run one epoch. If optimizer is None, runs in eval mode (no gradients)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    top1.reset()
    top5.reset()

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False, desc="train" if training else "val"):
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(labels)
            top1.update(logits, labels)
            top5.update(logits, labels)

    n = len(loader.dataset)
    return total_loss / n


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
) -> float:
    """Train one phase. Returns the best val Top-1 accuracy achieved."""
    print(f"\n{'=' * 60}")
    print(f"Phase {phase}: {'head only' if phase == 1 else 'partial fine-tuning'}")
    print(f"{'=' * 60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.get_param_groups(head_lr=head_lr, encoder_lr=encoder_lr),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    best_val_top1 = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device, top1, top5)
        train_top1 = top1.compute().item()
        train_top5 = top5.compute().item()

        val_loss = run_epoch(model, val_loader, criterion, None, scaler, device, top1, top5)
        val_top1 = top1.compute().item()
        val_top5 = top5.compute().item()

        scheduler.step()

        print(
            f"  Epoch {epoch:>2}/{num_epochs} | "
            f"train loss {train_loss:.4f} top1 {train_top1:.3f} top5 {train_top5:.3f} | "
            f"val loss {val_loss:.4f} top1 {val_top1:.3f} top5 {val_top5:.3f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            model.save(BEST_MODEL_PATH)
            print(f"    → New best val Top-1: {best_val_top1:.3f}  (saved)")

    return best_val_top1


def evaluate(model: BirdClassifier, test_loader: DataLoader, num_classes: int, device: torch.device) -> None:
    """Evaluate the best model on the test set and print a summary."""
    print(f"\n{'=' * 60}")
    print("Test set evaluation")
    print(f"{'=' * 60}")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)
    f1   = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    model.eval()
    top1.reset(); top5.reset(); f1.reset()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="test"):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                logits = model(images)
            top1.update(logits, labels)
            top5.update(logits, labels)
            f1.update(logits, labels)

    print(f"  Top-1 accuracy : {top1.compute().item():.4f}")
    print(f"  Top-5 accuracy : {top5.compute().item():.4f}")
    print(f"  Macro F1 score : {f1.compute().item():.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train BioCLIP bird classifier")
    parser.add_argument("--batch-size",     type=int, default=64)
    parser.add_argument("--num-workers",    type=int, default=4)
    parser.add_argument("--phase1-epochs",  type=int, default=5)
    parser.add_argument("--phase2-epochs",  type=int, default=15)
    parser.add_argument("--phase2-blocks",  type=int, default=4,   help="ViT blocks to unfreeze in phase 2")
    parser.add_argument("--head-lr",        type=float, default=1e-3)
    parser.add_argument("--encoder-lr",     type=float, default=1e-5)
    parser.add_argument("--resume",         type=str,  default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure splits exist
    if not (SPLITS_DIR / "train.parquet").exists():
        print("Splits not found — creating them now...")
        create_splits(index_path=INDEX_PATH, splits_dir=SPLITS_DIR)

    with open(SPLITS_DIR / "label_map.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"Number of classes: {num_classes}")

    train_loader, val_loader, test_loader = make_loaders(args.batch_size, args.num_workers)

    # Build or restore model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = BirdClassifier.load(Path(args.resume), num_classes=num_classes)
    else:
        model = BirdClassifier(num_classes=num_classes, freeze_encoder=True)
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
        )

    # Final evaluation on test set using best checkpoint
    print(f"\nLoading best model from {BEST_MODEL_PATH} for test evaluation...")
    best_model = BirdClassifier.load(BEST_MODEL_PATH, num_classes=num_classes).to(device)
    evaluate(best_model, test_loader, num_classes, device)


if __name__ == "__main__":
    main()
