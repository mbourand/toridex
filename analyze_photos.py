"""
Run ML inference on a folder of photos and write results to JSON.

For each image: detects bird (Faster R-CNN), classifies species (BirdClassifier),
extracts EXIF date and GPS. Prints PROGRESS:N:TOTAL lines to stdout for
real-time progress tracking by the Tauri app.

Usage:
    python analyze_photos.py --folder "F:/Photos" --output data/scan_results.json
    python analyze_photos.py --folder "F:/Photos" --output data/scan_results.json --no-detect
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from src.dataset import INPUT_SIZE, _CLIP_MEAN, _CLIP_STD
from src.model import BirdClassifier

SPLITS_DIR = Path("data/splits")
BEST_MODEL_PATH = Path("checkpoints/best_model.pt")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

_COCO_BIRD_CLASS = 16
_BBOX_PADDING = 0.15
_FALLBACK_RESIZE = 1024

_crop_transform = transforms.Compose([
    transforms.Lambda(lambda img: _pad_to_square(img)),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])

_fallback_transform = transforms.Compose([
    transforms.Resize(_FALLBACK_RESIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])


def _pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (128, 128, 128))
    padded.paste(img, ((side - w) // 2, (side - h) // 2))
    return padded


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _dms_to_decimal(dms, ref: str) -> float | None:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    try:
        d, m, s = dms
        decimal = float(d) + float(m) / 60 + float(s) / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def extract_exif(img: Image.Image) -> dict:
    result: dict = {}
    try:
        raw = img._getexif()
        if not raw:
            return result
        tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}

        # Date
        date_str = tags.get("DateTimeOriginal") or tags.get("DateTime")
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                result["exif_date"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        # GPS
        gps_info = tags.get("GPSInfo")
        if gps_info and isinstance(gps_info, dict):
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"))
            lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
            if lat is not None:
                result["exif_lat"] = lat
            if lon is not None:
                result["exif_lon"] = lon
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def load_detector(device: torch.device):
    print("Loading bird detector...", flush=True)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    detector.eval()
    return detector, weights.transforms()


def detect_bird(img, detector, det_transform, device, threshold):
    tensor = det_transform(img).to(device)
    with torch.no_grad():
        preds = detector([tensor])[0]
    mask = (preds["labels"] == _COCO_BIRD_CLASS) & (preds["scores"] >= threshold)
    if not mask.any():
        return None, None
    scores = preds["scores"][mask]
    boxes = preds["boxes"][mask]
    best = scores.argmax()
    return boxes[best].cpu().tolist(), scores[best].item()


def crop_to_bbox(img, bbox, padding):
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * padding)
    y1 = max(0, y1 - bh * padding)
    x2 = min(w, x2 + bw * padding)
    y2 = min(h, y2 + bh * padding)
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", action="append", required=True,
                        help="Folder to scan (can be repeated for multiple folders)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_PATH))
    parser.add_argument("--no-detect", action="store_true")
    parser.add_argument("--detect-thresh", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum top-1 confidence to assign a species. Below this, "
                             "the photo is kept but marked as __unknown__.")
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--incremental", action="store_true",
                        help="Skip files unchanged since last scan (by mtime+size)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label map
    with open(SPLITS_DIR / "label_map.json") as f:
        label_map: dict[str, int] = json.load(f)
    idx_to_sciname = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    # Collect images from all folders
    folders = [Path(f) for f in args.folder]
    image_paths: list[Path] = []
    for folder in folders:
        image_paths.extend(collect_images(folder))
    image_paths.sort()

    # Incremental: load previous results, skip unchanged files
    carried_results: dict[str, dict] = {}
    if args.incremental and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        previous_photos = prev_data.get("photos", {})

        to_process: list[Path] = []
        for path in image_paths:
            path_key = str(path.resolve())
            prev = previous_photos.get(path_key)
            if prev is not None:
                try:
                    stat = path.stat()
                    if (prev.get("file_mtime") == stat.st_mtime
                            and prev.get("file_size") == stat.st_size):
                        carried_results[path_key] = prev
                        continue
                except OSError:
                    pass
            to_process.append(path)
    else:
        to_process = image_paths

    total = len(to_process)
    carried = len(carried_results)
    print(f"Found {len(image_paths)} images, {carried} unchanged, {total} to analyze", flush=True)

    if total == 0:
        # Nothing new to analyze — write carried results and exit
        output_data = {
            "folders": [str(f) for f in folders],
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "photos": carried_results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"DONE:0", flush=True)
        return

    # Load models
    print("Loading classifier...", flush=True)
    classifier = BirdClassifier.load(Path(args.checkpoint), num_classes).to(device)
    classifier.eval()

    detector, det_transform = (None, None) if args.no_detect else load_detector(device)

    results: dict[str, dict] = {}

    for n, path in enumerate(to_process, 1):
        print(f"PROGRESS:{n}:{total}", flush=True)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        # Detection
        if detector is not None:
            bbox, _ = detect_bird(img, detector, det_transform, device, args.detect_thresh)
            if bbox is None:
                # Record as skipped so incremental won't re-process
                entry: dict = {"species_idx": -1, "scientificName": "__skipped__", "confidence": 0.0}
                try:
                    stat = path.stat()
                    entry["file_mtime"] = stat.st_mtime
                    entry["file_size"] = stat.st_size
                except OSError:
                    pass
                results[str(path.resolve())] = entry
                continue
        else:
            bbox = None

        # Classify
        if bbox is not None:
            crop = crop_to_bbox(img, bbox, _BBOX_PADDING)
            transform = _crop_transform
        else:
            crop = img
            transform = _fallback_transform

        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = classifier(tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(args.top, dim=1)

        top1_idx = top_indices[0][0].item()
        top1_prob = top_probs[0][0].item()

        uncertain = top1_prob < args.min_confidence
        entry: dict = {
            "species_idx": top1_idx,
            "scientificName": "__unknown__" if uncertain else idx_to_sciname.get(top1_idx, str(top1_idx)),
            "confidence": round(top1_prob, 4),
        }
        entry.update(extract_exif(img))

        # Include top-5 for detail view
        if args.top > 1:
            entry["top_k"] = [
                {"scientificName": idx_to_sciname.get(i.item(), str(i.item())), "confidence": round(p.item(), 4)}
                for p, i in zip(top_probs[0], top_indices[0])
            ]

        # Store file metadata for incremental scanning
        try:
            stat = path.stat()
            entry["file_mtime"] = stat.st_mtime
            entry["file_size"] = stat.st_size
        except OSError:
            pass

        results[str(path.resolve())] = entry

    # Merge carried + new results
    all_results = {**carried_results, **results}

    output_data = {
        "folders": [str(f) for f in folders],
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "photos": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"DONE:{total}", flush=True)


if __name__ == "__main__":
    main()
