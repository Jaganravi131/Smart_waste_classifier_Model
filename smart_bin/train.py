"""
Smart Bin Trash Classifier — YOLOv8 Classification Training

Uses the class folders prepared by prepare_dataset.py (10 classes by default).
Note that smart_bin/data.yaml declares only 6 classes - data.yaml is for the
*detection* dataset and is not read by this script; the class list here comes from
the folder names.

Prefer the verified pipeline (dataset.py / train_real.py / evaluate_real.py) for
anything that will be quoted in the report: it reproduces a published split and
records every metric.

Run:
  python train.py                          # default YOLOv8s, 50 epochs
  python train.py --model yolov8n          # smaller/faster (CPU-only)
  python train.py --epochs 100             # more epochs for higher accuracy
  python train.py --batch 32               # increase if you have >8GB VRAM
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets" / "cls"


def count_images():
    train_dir = DATASET_DIR / "train"
    val_dir   = DATASET_DIR / "val"
    if not train_dir.exists():
        return 0, 0, []
    classes   = [d.name for d in train_dir.iterdir() if d.is_dir()]
    train_n   = sum(len(list((train_dir / c).glob("*.*"))) for c in classes)
    val_n     = sum(len(list((val_dir   / c).glob("*.*"))) for c in classes)
    return train_n, val_n, classes


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 smart bin classifier")
    parser.add_argument("--model",  default="yolov8s",  help="yolov8n / yolov8s / yolov8m")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=224)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--name",   default="smart_bin_v1")
    args = parser.parse_args()

    train_n, val_n, classes = count_images()

    print("=" * 58)
    print("  Smart Bin Classifier — YOLOv8 Classification Training")
    print("=" * 58)
    print(f"  Model      : {args.model}-cls.pt  (classification)")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}x{args.imgsz}")
    print(f"  Batch      : {args.batch}")
    print(f"  Dataset    : {DATASET_DIR}")
    print(f"  Train imgs : {train_n}")
    print(f"  Val imgs   : {val_n}")
    print(f"  Classes    : {classes}")
    print("=" * 58)

    if train_n == 0:
        print("\nERROR: No training images found.")
        print("  Run these first:")
        print("    python download_dataset.py")
        print("    python prepare_dataset.py")
        return

    # YOLOv8 classification models use a -cls suffix. Accept either spelling
    # ("yolov8s" or "yolov8s-cls") instead of building "yolov8s-cls-cls.pt".
    weights = args.model if args.model.endswith("-cls") else f"{args.model}-cls"
    model = YOLO(f"{weights}.pt")

    results = model.train(
        data=str(DATASET_DIR),   # folder with train/ and val/ subdirs
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=15,
        augment=True,
        cache=False,             # set True if you have >=16GB RAM
        project=str(BASE_DIR / "runs"),
        exist_ok=False,   # never silently overwrite an earlier run
    )

    best_weights = BASE_DIR / "runs" / args.name / "weights" / "best.pt"
    print("\n" + "=" * 58)
    print("  Training complete!")
    print(f"  Best weights : {best_weights}")
    print(f"  Results      : {BASE_DIR / 'runs' / args.name / 'results.png'}")
    print("\n  Next step: python detect_smartbin.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
