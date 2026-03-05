"""
Export YOLOv8n (COCO pretrained) to ONNX for bird detection.
Replaces the 175MB Faster R-CNN with a ~6MB YOLOv8n model.

Usage:
    pip install ultralytics
    python export_yolo_detector.py
"""
from pathlib import Path
from ultralytics import YOLO

def main():
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    print("Exporting to ONNX (640x640 fixed input)...")
    model.export(format="onnx", imgsz=640, simplify=True, opset=17)

    src = Path("yolov8n.onnx")
    dst = models_dir / "bird_detector.onnx"
    if dst.exists():
        dst.unlink()
    src.rename(dst)

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"Saved to {dst} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
