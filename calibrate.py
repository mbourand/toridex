"""
Find the optimal temperature T for post-hoc calibration of the bird classifier.

Temperature scaling divides logits by T before softmax, making confidence
scores better reflect true probabilities. T > 1 softens overconfident
predictions; T < 1 sharpens under-confident ones.

The optimal T is found by minimizing negative log-likelihood (NLL) on the
validation set. This does NOT change model weights or accuracy — only the
confidence distribution.

Usage:
    python calibrate.py
    python calibrate.py --checkpoint checkpoints/best_model.pt

Output:
    Prints the optimal T value to use in the JS inference code.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BirdDataset, get_transforms
from src.model import BirdClassifier

SPLITS_DIR = Path("data/splits")
IMAGE_DIR = Path("data/images")
BEST_MODEL_PATH = Path("checkpoints/best_model.pt")


def collect_logits(
    model: BirdClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect all logits and labels from the validation set."""
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting logits"):
            images, labels = batch[0].to(device), batch[1].to(device)
            with torch.cuda.amp.autocast():
                logits = model(images)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def find_optimal_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Find T that minimizes NLL on the given logits/labels via grid search + refinement."""
    best_t = 1.0
    best_nll = float("inf")

    # Coarse grid search
    for t in [x * 0.1 for x in range(1, 100)]:
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_t = t

    # Fine-grained refinement around best_t
    for t in [best_t + x * 0.01 for x in range(-20, 21)]:
        if t <= 0:
            continue
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_t = t

    return round(best_t, 3)


def show_calibration_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> None:
    """Print before/after calibration statistics."""
    probs_before = F.softmax(logits, dim=1)
    probs_after = F.softmax(logits / temperature, dim=1)

    top1_conf_before = probs_before.max(dim=1).values
    top1_conf_after = probs_after.max(dim=1).values

    preds = logits.argmax(dim=1)
    correct = preds == labels

    # Accuracy (same before/after — temperature doesn't change rankings)
    acc = correct.float().mean().item()

    # Mean confidence on correct vs incorrect predictions
    correct_conf_before = top1_conf_before[correct].mean().item()
    wrong_conf_before = top1_conf_before[~correct].mean().item()
    correct_conf_after = top1_conf_after[correct].mean().item()
    wrong_conf_after = top1_conf_after[~correct].mean().item()

    print(f"\n{'=' * 55}")
    print(f"  Top-1 accuracy: {acc:.4f} (unchanged by temperature)")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<35} {'Before':>8} {'After':>8}")
    print(f"  {'-' * 51}")
    print(f"  {'Mean confidence (all)':<35} {top1_conf_before.mean().item():>7.1%} {top1_conf_after.mean().item():>7.1%}")
    print(f"  {'Mean confidence (correct preds)':<35} {correct_conf_before:>7.1%} {correct_conf_after:>7.1%}")
    print(f"  {'Mean confidence (wrong preds)':<35} {wrong_conf_before:>7.1%} {wrong_conf_after:>7.1%}")
    print(f"  {'NLL':<35} {F.cross_entropy(logits, labels).item():>8.4f} {F.cross_entropy(logits / temperature, labels).item():>8.4f}")
    print(f"{'=' * 55}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal temperature for calibration")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(SPLITS_DIR / "label_map.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading model from {checkpoint_path}...")
    model = BirdClassifier.load(checkpoint_path, num_classes=num_classes).to(device)

    # Load validation set
    val_ds = BirdDataset(SPLITS_DIR / "val.parquet", IMAGE_DIR, get_transforms("val"), label_map)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Collect logits
    logits, labels = collect_logits(model, val_loader, device)
    print(f"Collected {len(logits)} validation samples")

    # Find optimal temperature
    temperature = find_optimal_temperature(logits, labels)

    print(f"\n  Optimal temperature: T = {temperature}")
    show_calibration_stats(logits, labels, temperature)

    print(f"\n  Use this value in JS inference: logits.map(x => x / {temperature})")


if __name__ == "__main__":
    main()
