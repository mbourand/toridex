"""
Evaluate a trained BirdClassifier on the test set and compare against
the zero-shot BioCLIP baseline (no fine-tuning).

Outputs:
  - Side-by-side comparison: fine-tuned vs zero-shot baseline
  - Top-1 / Top-5 accuracy, Macro F1
  - Per-species accuracy (best and worst performers)
  - Most confused species pairs

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --no-baseline          # skip zero-shot comparison
    python evaluate.py --batch-size 256 --num-workers 4
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

from src.dataset import BirdDataset, get_transforms
from src.french_names import load_french_names
from src.model import BirdClassifier, BIOCLIP_MODEL_ID

SPLITS_DIR = Path("data/splits")
IMAGE_DIR = Path("data/images")
BEST_MODEL_PATH = Path("checkpoints/best_model.pt")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_test_loader(batch_size: int, num_workers: int) -> tuple[DataLoader, dict[str, int]]:
    with open(SPLITS_DIR / "label_map.json") as f:
        label_map = json.load(f)
    ds = BirdDataset(SPLITS_DIR / "test.parquet", IMAGE_DIR, get_transforms("val"), label_map)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, label_map


def build_idx_to_sciname(label_map: dict[str, int]) -> dict[int, str]:
    """Map label int → full scientific name using the test split metadata."""
    df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    # scientificName is the full binomial (e.g. "Parus major")
    epithet_to_sciname = df.groupby("species")["scientificName"].first().to_dict()
    idx_to_species = {v: k for k, v in label_map.items()}
    return {
        idx: epithet_to_sciname.get(epithet, epithet)
        for idx, epithet in idx_to_species.items()
    }


# ---------------------------------------------------------------------------
# Inference loop (shared by both models)
# ---------------------------------------------------------------------------

def collect_metrics(logits_fn, loader, num_classes: int, device: torch.device) -> dict:
    """
    Run inference using logits_fn(images) → logit tensor, collect all metrics.
    Returns a dict with top1, top5, f1, per_class_correct, per_class_total, confusion.
    """
    top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)
    f1   = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    class_correct: dict[int, int] = defaultdict(int)
    class_total:   dict[int, int] = defaultdict(int)
    confusion:     dict[tuple[int, int], int] = defaultdict(int)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                logits = logits_fn(images)

            preds = logits.argmax(dim=1)
            top1.update(logits, labels)
            top5.update(logits, labels)
            f1.update(logits, labels)

            for true, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                class_total[true] += 1
                if true == pred:
                    class_correct[true] += 1
                else:
                    confusion[(true, pred)] += 1

    return {
        "top1": top1.compute().item(),
        "top5": top5.compute().item(),
        "f1":   f1.compute().item(),
        "class_correct": dict(class_correct),
        "class_total":   dict(class_total),
        "confusion":     dict(confusion),
    }


# ---------------------------------------------------------------------------
# Fine-tuned model evaluation
# ---------------------------------------------------------------------------

def eval_finetuned(
    checkpoint_path: Path,
    loader: DataLoader,
    label_map: dict[str, int],
    device: torch.device,
) -> dict:
    num_classes = len(label_map)
    print(f"\nLoading fine-tuned model from {checkpoint_path}...")
    model = BirdClassifier.load(checkpoint_path, num_classes=num_classes).to(device)
    model.eval()

    results = collect_metrics(
        logits_fn=model,
        loader=tqdm(loader, desc="Fine-tuned model"),
        num_classes=num_classes,
        device=device,
    )

    # Free GPU memory before loading the baseline
    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Zero-shot BioCLIP baseline
# ---------------------------------------------------------------------------

def eval_zero_shot(
    loader: DataLoader,
    label_map: dict[str, int],
    idx_to_sciname: dict[int, str],
    device: torch.device,
) -> dict:
    """
    Zero-shot evaluation using cosine similarity between image and text embeddings.
    Text prompt: "a photo of {scientific name}"
    """
    import open_clip

    num_classes = len(label_map)
    print(f"\nLoading base BioCLIP model for zero-shot baseline...")
    clip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL_ID)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL_ID)

    # Build and encode text embeddings for all classes (done once)
    species_order = sorted(label_map.items(), key=lambda x: x[1])  # sorted by label int
    prompts = [f"a photo of {idx_to_sciname[idx]}" for _, idx in species_order]
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def zero_shot_logits(images: torch.Tensor) -> torch.Tensor:
        img_features = clip_model.encode_image(images)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        # Scale by 100 to get logit-like magnitudes for torchmetrics
        return (img_features @ text_features.T) * 100.0

    results = collect_metrics(
        logits_fn=zero_shot_logits,
        loader=tqdm(loader, desc="Zero-shot baseline"),
        num_classes=num_classes,
        device=device,
    )

    del clip_model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _display(idx: int, idx_to_sciname: dict[int, str], french: dict[str, str]) -> str:
    """Format a species label as 'Nom français (Scientific name)', width-padded."""
    sciname = idx_to_sciname.get(idx, str(idx))
    french_name = french.get(sciname.lower(), "")
    if french_name:
        return f"{french_name} ({sciname})"
    return sciname


def print_comparison(finetuned: dict, baseline: dict) -> None:
    def delta(a, b):
        sign = "+" if a >= b else ""
        return f"({sign}{(a - b)*100:.2f}pp)"

    print(f"\n{'=' * 70}")
    print(f"{'Metric':<20} {'Fine-tuned':>14} {'Zero-shot':>14} {'Delta':>14}")
    print(f"{'─' * 70}")
    print(f"{'Top-1 accuracy':<20} {finetuned['top1']*100:>13.2f}% {baseline['top1']*100:>13.2f}% {delta(finetuned['top1'], baseline['top1']):>14}")
    print(f"{'Top-5 accuracy':<20} {finetuned['top5']*100:>13.2f}% {baseline['top5']*100:>13.2f}% {delta(finetuned['top5'], baseline['top5']):>14}")
    print(f"{'Macro F1':<20} {finetuned['f1']*100:>13.2f}% {baseline['f1']*100:>13.2f}% {delta(finetuned['f1'], baseline['f1']):>14}")
    print(f"{'=' * 70}")


def print_detailed(
    results: dict,
    label_map: dict[str, int],
    idx_to_sciname: dict[int, str],
    french: dict[str, str],
    title: str,
) -> None:
    class_correct = results["class_correct"]
    class_total   = results["class_total"]
    confusion     = results["confusion"]

    per_class_acc = {
        label: class_correct.get(label, 0) / class_total[label]
        for label in class_total
    }
    sorted_acc = sorted(per_class_acc.items(), key=lambda x: x[1])

    print(f"\n{'─' * 75}")
    print(f"{title} — 10 worst-performing species")
    print(f"{'─' * 75}")
    for label, acc in sorted_acc[:10]:
        name = _display(label, idx_to_sciname, french)
        print(f"  {acc*100:5.1f}%  {name:<50}  (n={class_total[label]})")

    print(f"\n{'─' * 75}")
    print(f"{title} — 10 best-performing species")
    print(f"{'─' * 75}")
    for label, acc in sorted_acc[-10:][::-1]:
        name = _display(label, idx_to_sciname, french)
        print(f"  {acc*100:5.1f}%  {name:<50}  (n={class_total[label]})")

    top_confused = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n{'─' * 75}")
    print(f"{title} — 10 most confused pairs  (true → predicted)")
    print(f"{'─' * 75}")
    for (true, pred), count in top_confused:
        total = class_total[true]
        true_name = _display(true, idx_to_sciname, french)
        pred_name = _display(pred, idx_to_sciname, french)
        print(f"  {count:>4}x  {true_name:<45} → {pred_name:<45}  ({count/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BirdClassifier and compare to zero-shot baseline")
    parser.add_argument("--checkpoint",   type=str,  default=str(BEST_MODEL_PATH))
    parser.add_argument("--batch-size",   type=int,  default=128)
    parser.add_argument("--num-workers",  type=int,  default=4)
    parser.add_argument("--no-baseline",  action="store_true", help="Skip zero-shot baseline comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run 'python train.py' first to produce a checkpoint."
        )

    loader, label_map = build_test_loader(args.batch_size, args.num_workers)
    idx_to_sciname = build_idx_to_sciname(label_map)

    # Load French common names (fetches from GBIF on first run, ~2 min; instant after)
    french = load_french_names()

    # 1. Fine-tuned model
    finetuned = eval_finetuned(checkpoint_path, loader, label_map, device)

    if not args.no_baseline:
        # 2. Zero-shot BioCLIP baseline (loads fresh model after freeing fine-tuned)
        baseline = eval_zero_shot(loader, label_map, idx_to_sciname, device)

        # 3. Comparison table
        print_comparison(finetuned, baseline)

        # 4. Detailed breakdown for both
        print_detailed(finetuned, label_map, idx_to_sciname, french, "Fine-tuned")
        print_detailed(baseline,  label_map, idx_to_sciname, french, "Zero-shot")
    else:
        print_detailed(finetuned, label_map, idx_to_sciname, french, "Fine-tuned")


if __name__ == "__main__":
    main()
