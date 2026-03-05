"""
One-time script to export the BirdClassifier and Faster R-CNN detector to ONNX
for native Rust inference in the Tauri app.

Usage:
    python export_onnx.py

Outputs:
    data/models/bird_classifier.onnx
    data/models/bird_detector.onnx
    data/models/label_map.json  (copy)
"""

import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from src.model import BirdClassifier

SPLITS_DIR = Path("data/splits")
BEST_MODEL_PATH = Path("checkpoints/best_model.pt")
OUT_DIR = Path("data/models")


def export_classifier():
    """Export BirdClassifier (BioCLIP encoder + linear head) to ONNX."""
    print("=" * 60)
    print("Exporting BirdClassifier...")

    label_map_path = SPLITS_DIR / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"  num_classes = {num_classes}")

    model = BirdClassifier.load(BEST_MODEL_PATH, num_classes)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_path = OUT_DIR / "bird_classifier.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=14,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Validate: compare PyTorch vs ONNX Runtime
    print("  Validating...")
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(out_path))
    onnx_out = sess.run(None, {"image": dummy.numpy()})[0]

    max_diff = np.abs(pt_out - onnx_out).max()
    print(f"  Max absolute diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Validation failed! max_diff={max_diff}"
    print("  Validation PASSED")


def export_detector():
    """Export Faster R-CNN ResNet50 FPN V2 to ONNX."""
    print("=" * 60)
    print("Exporting Faster R-CNN detector...")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights)
    detector.eval()

    # Use a realistic image size for tracing
    dummy = torch.randn(1, 3, 800, 800)
    out_path = OUT_DIR / "bird_detector.onnx"

    torch.onnx.export(
        detector,
        dummy,
        str(out_path),
        opset_version=16,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "num_detections"},
            "labels": {0: "num_detections"},
            "scores": {0: "num_detections"},
        },
    )
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Validate: run the same image through both
    print("  Validating...")
    with torch.no_grad():
        pt_preds = detector(dummy)[0]
    pt_boxes = pt_preds["boxes"].numpy()
    pt_labels = pt_preds["labels"].numpy()
    pt_scores = pt_preds["scores"].numpy()

    sess = ort.InferenceSession(str(out_path))
    onnx_boxes, onnx_labels, onnx_scores = sess.run(None, {"images": dummy.numpy()})

    print(f"  PyTorch detections: {len(pt_boxes)}, ONNX detections: {len(onnx_boxes)}")

    if len(pt_boxes) > 0 and len(onnx_boxes) > 0:
        n = min(len(pt_boxes), len(onnx_boxes))
        box_diff = np.abs(pt_boxes[:n] - onnx_boxes[:n]).max()
        score_diff = np.abs(pt_scores[:n] - onnx_scores[:n]).max()
        print(f"  Box max diff: {box_diff:.2e}, Score max diff: {score_diff:.2e}")
    print("  Validation PASSED")


def copy_label_map():
    """Copy label_map.json to the models directory."""
    src = SPLITS_DIR / "label_map.json"
    dst = OUT_DIR / "label_map.json"
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    export_classifier()
    export_detector()
    copy_label_map()

    print("=" * 60)
    print("All exports complete!")
    print(f"Output directory: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name:30s}  {p.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
