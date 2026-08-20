"""
Prepare the downloaded garbage dataset for YOLOv8 classification training.

Input:  datasets/raw/original/<class>/*.jpg
Output: datasets/cls/train/<class>/*.jpg
        datasets/cls/val/<class>/*.jpg

Run:  python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

RAW_DIR  = Path(__file__).parent / "datasets" / "raw" / "original"
OUT_DIR  = Path(__file__).parent / "datasets" / "cls"
VAL_SPLIT = 0.2   # 20% of images go to validation
SEED      = 42

# Classes relevant to smart bin (keep all — more classes = smarter bin)
# clothes/shoes go to BIN_5 (general trash)
CLASSES = ["battery", "biological", "cardboard", "glass", "metal",
           "paper", "plastic", "shoes", "clothes", "trash"]

random.seed(SEED)

total_train = 0
total_val   = 0

print(f"Preparing dataset: {RAW_DIR} → {OUT_DIR}\n")
print(f"{'Class':<12} {'Total':>6} {'Train':>6} {'Val':>5}")
print("-" * 34)

for cls in CLASSES:
    src = RAW_DIR / cls
    if not src.exists():
        print(f"  WARNING: {cls} folder not found, skipping.")
        continue

    images = [f for f in src.iterdir()
              if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    random.shuffle(images)

    n_val   = max(1, int(len(images) * VAL_SPLIT))
    n_train = len(images) - n_val
    val_imgs   = images[:n_val]
    train_imgs = images[n_val:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dest = OUT_DIR / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dest / img.name)

    total_train += n_train
    total_val   += n_val
    print(f"  {cls:<12} {len(images):>6} {n_train:>6} {n_val:>5}")

print("-" * 34)
print(f"  {'TOTAL':<12} {total_train+total_val:>6} {total_train:>6} {total_val:>5}")
print(f"\n✓ Dataset ready at: {OUT_DIR}")
print(f"\nNext step: python train.py")
