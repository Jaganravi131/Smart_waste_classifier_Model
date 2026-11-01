"""
Prepare a downloaded garbage dataset for YOLOv8 classification training.

Input:  datasets/raw/original/<class>/*.jpg
Output: datasets/cls/{train,val,test}/<class>/*.jpg

FIXED IN THIS REVISION
  * a **test split** is produced. The previous version made only train/val, so no
    held-out partition existed and the "never used in training" claim in the
    report could not be verified;
  * the source images are *hard-linked* when possible instead of copied, so a
    12,000-image dataset does not double in size on every run;
  * classes with too few images are reported and skipped, instead of silently
    putting one image into validation;
  * the class list is checked against data.yaml and the difference is printed -
    data.yaml declares 6 detection classes, this script builds 10.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --test-split 0.1
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "datasets" / "raw" / "original"
OUT_DIR = BASE_DIR / "datasets" / "cls"

# Classes relevant to the smart bin. clothes/shoes route to the residual stream;
# see routing.py for the class -> compartment map.
CLASSES = ["battery", "biological", "cardboard", "glass", "metal",
           "paper", "plastic", "shoes", "clothes", "trash"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_PER_CLASS = 20  # anything smaller cannot be split three ways meaningfully


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)          # no extra disk, still a real file for the trainer
    except OSError:
        shutil.copy2(src, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train/val/test folders for classification")
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"ERROR: {args.raw} does not exist.")
        print("  Fetch a dataset first. For an account-free, reproducible option:")
        print("      python dataset.py")
        return 1

    random.seed(args.seed)
    if args.val_split + args.test_split >= 0.9:
        print("ERROR: val-split + test-split must leave room for a training set.")
        return 1

    print(f"Preparing dataset: {args.raw} -> {args.out}\n")
    print(f"{'Class':<12} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print("-" * 40)

    totals = {"train": 0, "val": 0, "test": 0}
    skipped = []
    for cls in CLASSES:
        src = args.raw / cls
        if not src.exists():
            skipped.append(cls)
            print(f"  {cls:<12} {'-':>6} {'-':>6} {'-':>5} {'-':>5}   (folder missing)")
            continue

        images = [f for f in src.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES]
        if len(images) < MIN_PER_CLASS:
            skipped.append(cls)
            print(f"  {cls:<12} {len(images):>6}   too few images to split, skipped")
            continue

        random.shuffle(images)
        n_total = len(images)
        n_test = max(1, int(n_total * args.test_split))
        n_val = max(1, int(n_total * args.val_split))
        n_train = n_total - n_val - n_test

        splits = {
            "train": images[n_test + n_val:],
            "val": images[n_test:n_test + n_val],
            "test": images[:n_test],
        }
        for split, files in splits.items():
            for image in files:
                link_or_copy(image, args.out / split / cls / image.name)
            totals[split] += len(files)

        print(f"  {cls:<12} {n_total:>6} {n_train:>6} {n_val:>5} {n_test:>5}")

    print("-" * 40)
    print(f"  {'TOTAL':<12} {sum(totals.values()):>6} {totals['train']:>6} "
          f"{totals['val']:>5} {totals['test']:>5}")
    if skipped:
        print(f"\n  skipped: {', '.join(skipped)}")

    yaml_path = BASE_DIR / "data.yaml"
    if yaml_path.exists():
        import yaml

        declared = list((yaml.safe_load(yaml_path.read_text()) or {}).get("names", {}).values())
        if declared and set(declared) != set(CLASSES):
            print(f"\n  NOTE: data.yaml declares {len(declared)} classes {declared},")
            print(f"        this script built {len(CLASSES)}: {CLASSES}")
            print("        data.yaml is only used by the *detection* trainer.")

    print(f"\nDataset ready at {args.out}")
    print("Next: python train.py            (YOLOv8-cls)")
    print("  or: python train_real.py       (verified pipeline, records real metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
