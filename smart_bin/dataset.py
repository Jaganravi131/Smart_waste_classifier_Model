"""
BinGenie - dataset acquisition and reproducible splitting.

The previous version of the project pointed at two different datasets that do
not agree (data.yaml says 6 YOLO classes, prepare_dataset.py builds 10
ImageFolder classes, the report claims 7), and nothing in the repository could
be reproduced from a clean clone because `datasets/` is git-ignored and there
was no download step that actually worked without a Roboflow key.

This module fixes that: one command fetches a public dataset and lays it out
with the *official* train/val/test split published with it.

Usage
-----
    python dataset.py                 # download + build the default dataset
    python dataset.py --check         # report what is on disk
    python dataset.py --raw DIR       # use a dataset you already have

Default dataset: TrashNet (garythung/trashnet), 2,527 images, 6 classes
(cardboard, glass, metal, paper, plastic, trash) - the same six classes that
smart_bin/data.yaml declares.  It is downloaded as a repository tarball so no
Kaggle/Roboflow credentials are needed, and the split files shipped in the
repository are used so results are comparable with the published literature.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "datasets"
TRASHNET_DIR = DATA_ROOT / "trashnet"

TRASHNET_TARBALL = "https://codeload.github.com/garythung/trashnet/tar.gz/refs/heads/master"
SPLIT_FILES = {
    "train": "one-indexed-files-notrash_train.txt",
    "val": "one-indexed-files-notrash_val.txt",
    "test": "one-indexed-files-notrash_test.txt",
}


def _download(url: str, dest: Path, timeout: int = 300) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] downloading {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec - fixed URL
        payload = response.read()
    dest.write_bytes(payload)
    print(f"[dataset] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_and_extract(dest: Path = DATA_ROOT / "raw") -> Path:
    """Download the TrashNet tarball and return the folder holding the images."""
    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / "trashnet.tar.gz"
    if not tarball.exists():
        _download(TRASHNET_TARBALL, tarball)

    extract_dir = dest / "trashnet-master"
    if not extract_dir.exists():
        with tarfile.open(tarball) as tar:
            tar.extractall(dest)  # nosec - tarball comes from the fixed URL above

    zip_path = extract_dir / "data" / "dataset-resized.zip"
    images_root = extract_dir / "data" / "images"
    if not images_root.exists() and zip_path.exists():
        import zipfile

        zipfile.ZipFile(zip_path).extractall(extract_dir / "data" / "images")
        images_root = extract_dir / "data" / "images" / "dataset-resized"
    if not images_root.exists():
        raise FileNotFoundError("Could not find the TrashNet image folder after extraction.")
    return images_root


def build_trashnet_split(images_root: Path, split_root: Path = TRASHNET_DIR) -> Dict[str, Dict[str, int]]:
    """Copy images into split_root/{train,val,test}/<class>/ using the official lists."""
    index = {p.name: p.parent.name for p in images_root.glob("*/*.jpg")}
    if not index:
        raise FileNotFoundError(f"No .jpg files under {images_root}")

    source_data = images_root.parent
    counts: Dict[str, Dict[str, int]] = {}
    for split, filename in SPLIT_FILES.items():
        listing = source_data / filename
        if not listing.exists():
            raise FileNotFoundError(
                f"Missing official split file {listing}. "
                "Re-download the dataset with `python dataset.py --force`."
            )
        per_class: Dict[str, int] = {}
        for line in listing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name = line.split()[0]
            cls = index.get(name)
            if cls is None:
                continue
            target = split_root / split / cls
            target.mkdir(parents=True, exist_ok=True)
            if not (target / name).exists():
                shutil.copy2(images_root / cls / name, target / name)
            per_class[cls] = per_class.get(cls, 0) + 1
        counts[split] = dict(sorted(per_class.items()))
    return counts


def describe_split(split_root: Path = TRASHNET_DIR) -> None:
    if not split_root.exists():
        print(f"[dataset] nothing built yet at {split_root}")
        return
    print(f"[dataset] {split_root}")
    for split in ("train", "val", "test"):
        folder = split_root / split
        if not folder.exists():
            continue
        per_class = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(folder.iterdir()) if d.is_dir()}
        print(f"  {split:<5} total={sum(per_class.values()):<5} {per_class}")


def ensure_dataset(split_root: Path = TRASHNET_DIR) -> Path:
    """Build the dataset if it is not present yet. Returns the split root."""
    if (split_root / "train").exists():
        return split_root
    images_root = download_and_extract()
    counts = build_trashnet_split(images_root, split_root)
    print(f"[dataset] built splits: {counts}")
    return split_root


def dataset_summary(split_root: Path) -> Dict[str, object]:
    """Counts + imbalance ratio, used in reports and the model card."""
    summary: Dict[str, object] = {}
    for split in ("train", "val", "test"):
        folder = split_root / split
        if folder.exists():
            summary[split] = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(folder.iterdir()) if d.is_dir()}
    train = summary.get("train") or {}
    if train:
        biggest, smallest = max(train.values()), min(train.values())
        summary["train_imbalance_ratio"] = round(biggest / smallest, 2)
        summary["train_total"] = sum(train.values())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the BinGenie classification dataset")
    parser.add_argument("--check", action="store_true", help="only report what is on disk")
    parser.add_argument("--force", action="store_true", help="rebuild the split folders")
    parser.add_argument("--raw", type=Path, default=None, help="use an existing image folder")
    args = parser.parse_args()

    if args.check:
        describe_split()
        return 0

    if args.force and TRASHNET_DIR.exists():
        shutil.rmtree(TRASHNET_DIR)

    if args.raw:
        counts = build_trashnet_split(args.raw, TRASHNET_DIR)
        print(f"[dataset] built splits from {args.raw}: {counts}")
    else:
        ensure_dataset()
    describe_split()
    return 0


if __name__ == "__main__":
    sys.exit(main())
