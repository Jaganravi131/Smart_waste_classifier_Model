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


def _has_class_images(folder: Path) -> bool:
    """True when `folder` looks like ImageFolder layout: <folder>/<class>/*.jpg."""
    return folder.is_dir() and any(folder.glob("*/*.jpg"))


def download_and_extract(dest: Path = DATA_ROOT / "raw") -> Path:
    """Download the TrashNet tarball and return the folder holding the images.

    Deliberately idempotent and tolerant of a half-finished previous attempt:
    the image folder is chosen by *looking for class sub-folders that contain
    images*, not by testing whether a directory path exists. The first version
    tested `Path.exists()` on `data/images`, which is created by the zip
    extraction itself, so a second run returned an empty folder and the split
    builder failed with "No .jpg files under .../data/images".
    """
    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / "trashnet.tar.gz"
    if not tarball.exists():
        _download(TRASHNET_TARBALL, tarball)

    extract_dir = dest / "trashnet-master"
    if not extract_dir.exists():
        with tarfile.open(tarball) as tar:
            tar.extractall(dest)  # nosec - tarball comes from the fixed URL above

    # Candidate locations, shallowest expected layout last.
    candidates = [
        extract_dir / "data" / "images" / "dataset-resized",
        extract_dir / "data" / "dataset-resized",
        extract_dir / "data" / "images",
        extract_dir / "data",
    ]
    for candidate in candidates:
        if _has_class_images(candidate):
            return candidate

    # Nothing usable yet: unpack the resized archive (dataset-resized.zip).
    zip_path = extract_dir / "data" / "dataset-resized.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"No images and no {zip_path.name} in {extract_dir}")
    import zipfile

    target = extract_dir / "data" / "images"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] unpacking {zip_path.name} (42 MB, ~2,500 images)")
    zipfile.ZipFile(zip_path).extractall(target)

    for candidate in candidates:
        if _has_class_images(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a <class>/<image>.jpg layout under "
        f"{extract_dir / 'data'}. Delete smart_bin/datasets and re-run."
    )


def _find_split_dir(images_root: Path) -> Path:
    """Locate the folder that holds the official split lists.

    The images can live at either of two depths depending on whether the zip had
    to be unpacked (dataset-resized.zip) or was already expanded:

        <repo>/data/images/<class>/*.jpg              -> lists are in data/
        <repo>/data/images/dataset-resized/<class>/*  -> lists are in data/

    The first version of this function assumed the first layout, so a fresh clone
    crashed with "Missing official split file .../data/images/one-indexed-...".
    """
    for candidate in (images_root, images_root.parent, images_root.parent.parent):
        if (candidate / SPLIT_FILES["train"]).exists():
            return candidate
    searched = " | ".join(str(c) for c in (images_root, images_root.parent, images_root.parent.parent))
    raise FileNotFoundError(
        f"Could not find {SPLIT_FILES['train']} next to the images. Searched: {searched}. "
        "Re-download with `python dataset.py --force`."
    )


def build_trashnet_split(images_root: Path, split_root: Path = TRASHNET_DIR) -> Dict[str, Dict[str, int]]:
    """Copy images into split_root/{train,val,test}/<class>/ using the official lists."""
    index = {p.name: p.parent.name for p in images_root.glob("*/*.jpg")}
    if not index:
        raise FileNotFoundError(f"No .jpg files under {images_root}")

    source_data = _find_split_dir(images_root)
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
