"""
One-time script to export models to ONNX for the Tauri app.

- Classifier: BioCLIP encoder + linear head (exported from PyTorch)
- Detector: YOLOv11m (exported via ultralytics at 1280x1280)

Usage:
    python export_onnx.py
    python export_onnx.py --classifier-only
    python export_onnx.py --detector-only

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

from src.model import BirdClassifier


def _patch_for_onnx_export():
    """Monkey-patch ops unsupported in PyTorch 2.0 ONNX export (unflatten, SDPA)."""
    import math
    import torch.nn.functional as F

    originals = {
        "unflatten": torch.Tensor.unflatten,
        "sdpa": F.scaled_dot_product_attention,
    }

    # unflatten → reshape
    def _unflatten_as_reshape(self, dim, sizes):
        shape = list(self.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = shape[:dim] + list(sizes) + shape[dim + 1:]
        return self.reshape(new_shape)

    # SDPA → manual attention (matmul → scale → mask → softmax → matmul)
    def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        scale = 1.0 / math.sqrt(query.size(-1))
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, value)

    torch.Tensor.unflatten = _unflatten_as_reshape
    F.scaled_dot_product_attention = _manual_sdpa
    return originals


def _restore_patches(originals):
    """Restore monkey-patched functions."""
    import torch.nn.functional as F
    torch.Tensor.unflatten = originals["unflatten"]
    F.scaled_dot_product_attention = originals["sdpa"]

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

    # Patch unsupported ops for PyTorch 2.0 ONNX compat
    originals = _patch_for_onnx_export()

    dummy = torch.randn(1, 3, 224, 224)
    out_path = OUT_DIR / "bird_classifier.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=17,
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

    # Restore patched ops
    _restore_patches(originals)


def export_detector():
    """Export YOLOv11m to ONNX at 1280x1280."""
    print("=" * 60)
    print("Exporting YOLOv11m detector...")

    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")
    model.export(format="onnx", imgsz=1280, opset=16, simplify=True, dynamic=False)

    exported = Path("yolo11m.onnx")
    out_path = OUT_DIR / "bird_detector.onnx"
    shutil.move(str(exported), str(out_path))
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Validate: check the model loads and has expected inputs/outputs
    print("  Validating...")
    sess = ort.InferenceSession(str(out_path))
    inputs = {inp.name: inp.shape for inp in sess.get_inputs()}
    outputs = {out.name: out.shape for out in sess.get_outputs()}
    print(f"  Inputs:  {inputs}")
    print(f"  Outputs: {outputs}")
    assert "images" in inputs, "Expected input 'images'"
    assert "output0" in outputs, "Expected output 'output0'"
    print("  Validation PASSED")


def copy_label_map():
    """Copy label_map.json to the models directory."""
    src = SPLITS_DIR / "label_map.json"
    dst = OUT_DIR / "label_map.json"
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--classifier-only", action="store_true",
                        help="Only export classifier (skip detector)")
    parser.add_argument("--detector-only", action="store_true",
                        help="Only export detector (skip classifier)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.detector_only:
        export_classifier()
        copy_label_map()
    if not args.classifier_only:
        export_detector()

    print("=" * 60)
    print("All exports complete!")
    print(f"Output directory: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name:30s}  {p.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
