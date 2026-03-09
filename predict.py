"""
Run the trained BirdClassifier on your own photos.

For each image, a Faster R-CNN detector first locates the bird in the frame,
then the classifier runs on the detected crop. If no bird is detected the full
image is used as a fallback (with a large resize to preserve context).

For each image, prints the top-5 predicted species with confidence scores
and French common names.

Usage:
    python predict.py "F:/Photos/Processed/2026-02-28"
    python predict.py "F:/Photos/Processed/2026-02-28" --top 3
    python predict.py "F:/Photos/Processed/2026-02-28/DSC_0042.jpg"
    python predict.py "F:/Photos/Processed/2026-02-28" --no-detect   # skip detection
    python predict.py "F:/Photos/Processed/2026-02-28" --detect-thresh 0.3
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from tqdm import tqdm

from src.dataset import INPUT_SIZE, _CLIP_MEAN, _CLIP_STD
from src.french_names import load_french_names
from src.model import BirdClassifier

SPLITS_DIR = Path("data/splits")
BEST_MODEL_PATH = Path("checkpoints/best_model.pt")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

# COCO class index for "bird" (0 = background, 16 = bird in torchvision)
_COCO_BIRD_CLASS = 16

# Padding added around the detected bounding box (fraction of box size)
_BBOX_PADDING = 0.15

# Fallback resize when no bird is detected
_FALLBACK_RESIZE = 1024

# Transform for a tight detection crop.
# We pad to a square first so CenterCrop doesn't slice off the head or tail
# of a tall/narrow bird — then resize to 224.
_crop_transform = transforms.Compose([
    transforms.Lambda(lambda img: _pad_to_square(img)),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])

# Transform for the fallback (no detection) — full image with large resize.
_fallback_transform = transforms.Compose([
    transforms.Resize(_FALLBACK_RESIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])


def _pad_to_square(img: Image.Image) -> Image.Image:
    """Pad the shorter side with neutral grey so the bird isn't cropped."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (128, 128, 128))
    padded.paste(img, ((side - w) // 2, (side - h) // 2))
    return padded


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def load_detector(device: torch.device):
    """Load Faster R-CNN pre-trained on COCO. No new packages required."""
    print("Loading bird detector (Faster R-CNN, COCO)...")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    detector.eval()
    # The detector expects float tensors in [0, 1]
    det_transform = weights.transforms()
    return detector, det_transform


def detect_bird(
    img: Image.Image,
    detector,
    det_transform,
    device: torch.device,
    threshold: float,
) -> tuple[list[float], float] | tuple[None, None]:
    """
    Return ([x1, y1, x2, y2], score) of the highest-confidence bird,
    or (None, None) if no bird passes the threshold.
    """
    tensor = det_transform(img).to(device)
    with torch.no_grad():
        preds = detector([tensor])[0]

    bird_mask = (preds["labels"] == _COCO_BIRD_CLASS) & (preds["scores"] >= threshold)
    if not bird_mask.any():
        return None, None

    scores = preds["scores"][bird_mask]
    boxes  = preds["boxes"][bird_mask]
    best   = scores.argmax()
    return boxes[best].cpu().tolist(), scores[best].item()


def crop_to_bbox(img: Image.Image, bbox: list[float], padding: float) -> Image.Image:
    """Crop to bbox with relative padding, clamped to image bounds."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * padding)
    y1 = max(0, y1 - bh * padding)
    x2 = min(w, x2 + bw * padding)
    y2 = min(h, y2 + bh * padding)
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Prediction loop
# ---------------------------------------------------------------------------

def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images = sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No supported images found in {path}\n"
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return images


def predict_batch(
    image_paths: list[Path],
    classifier: BirdClassifier,
    detector,
    det_transform,
    device: torch.device,
    top_k: int,
    det_threshold: float,
) -> list[tuple[Path, torch.Tensor, torch.Tensor, str]]:
    """
    Returns list of (path, top_k_probs, top_k_indices, crop_info) where
    crop_info is a human-readable string describing what was cropped.
    """
    results = []
    for path in tqdm(image_paths, desc="Predicting", unit="img"):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {path.name}: {e}")
            continue

        # 1. Detect bird
        if detector is not None:
            bbox, score = detect_bird(img, detector, det_transform, device, det_threshold)
        else:
            bbox, score = None, None

        # 2. Crop or fallback
        if bbox is not None:
            crop = crop_to_bbox(img, bbox, _BBOX_PADDING)
            bw = int(bbox[2] - bbox[0])
            bh = int(bbox[3] - bbox[1])
            crop_info = f"detected bird  {bw}×{bh}px  conf={score:.2f}"
        else:
            crop = img
            crop_info = "no bird detected — using full image (fallback)"

        # 3. Classify — tight crop vs. fallback get different transforms
        transform = _crop_transform if bbox is not None else _fallback_transform
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = classifier(tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)
        results.append((path, top_probs[0].cpu(), top_indices[0].cpu(), crop_info))

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(
    results: list[tuple[Path, torch.Tensor, torch.Tensor, str]],
    idx_to_label: dict[int, str],
    french: dict[str, str],
    idx_to_sciname: dict[int, str],
) -> None:
    for path, probs, indices, crop_info in results:
        print(f"\n{'─' * 65}")
        print(f"  {path.name}")
        print(f"  [{crop_info}]")
        print(f"{'─' * 65}")
        for rank, (prob, idx) in enumerate(zip(probs.tolist(), indices.tolist()), 1):
            sciname = idx_to_sciname.get(idx, idx_to_label.get(idx, str(idx)))
            french_name = french.get(sciname.lower(), "")
            display = f"{french_name} ({sciname})" if french_name else sciname
            bar = "█" * int(prob * 30)
            print(f"  {rank}. {display:<45}  {prob*100:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict bird species in your photos")
    parser.add_argument("path",             type=str,   help="Image file or folder of images")
    parser.add_argument("--checkpoint",     type=str,   default=str(BEST_MODEL_PATH))
    parser.add_argument("--top",            type=int,   default=5,   help="Number of top predictions")
    parser.add_argument("--no-detect",      action="store_true",     help="Skip bird detection, use full image")
    parser.add_argument("--detect-thresh",  type=float, default=0.5, help="Detection confidence threshold (0–1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label map
    label_map_path = SPLITS_DIR / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"{label_map_path} not found. Run 'python train.py' first.")
    with open(label_map_path) as f:
        label_map: dict[str, int] = json.load(f)

    num_classes   = len(label_map)
    idx_to_label  = {v: k for k, v in label_map.items()}

    # label_map keys are now full binomials (e.g. "Parus major"),
    # so idx_to_label already gives scientific names.
    idx_to_sciname = idx_to_label

    french = load_french_names()

    input_path = Path(args.path)
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    image_paths = collect_images(input_path)
    print(f"Found {len(image_paths)} image(s) in {input_path}")

    # Load models
    classifier = BirdClassifier.load(Path(args.checkpoint), num_classes).to(device)
    classifier.eval()

    if args.no_detect:
        detector, det_transform = None, None
    else:
        detector, det_transform = load_detector(device)

    results = predict_batch(
        image_paths, classifier, detector, det_transform,
        device, args.top, args.detect_thresh,
    )

    print_results(results, idx_to_label, french, idx_to_sciname)
    print(f"\n{'─' * 65}")
    print(f"Done. {len(results)}/{len(image_paths)} images processed.")


if __name__ == "__main__":
    main()
