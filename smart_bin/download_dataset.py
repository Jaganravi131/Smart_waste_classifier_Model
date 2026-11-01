"""
Download a garbage *detection* dataset from Roboflow Universe (optional path).

This needs a Roboflow account and internet access. For classification - which is
what the report's accuracy numbers describe - use `dataset.py` instead: it needs
no account, fetches TrashNet from a public repository tarball, and rebuilds the
published train/val/test split, so a result can be reproduced.

FIXED IN THIS REVISION
  * the API-key check used to run at *import* time (`raise` on line 15), so the
    module could not be imported or inspected without a key;
  * a failed download now exits with a code and a clear message instead of
    `sys.exit()` from deep inside the script;
  * the destination folder is checked for images before it is reported as ready.

Usage:
    ROBOFLOW_API_KEY=... python download_dataset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

API_KEY_ENV = "ROBOFLOW_API_KEY"
DEST = Path(__file__).resolve().parent / "datasets" / "trash"

# (workspace, project, version) - each entry is a public Roboflow Universe project.
CANDIDATES = [
    ("roboflow-universe-projects", "garbage-classification-3", 3),
    ("roboflow-universe-projects", "garbage-detection-3", 1),
    ("material-identification", "garbage-classification-3", 2),
    ("tannergi", "garbage_detection-wvzwv", 9),
]

MANUAL_INSTRUCTIONS = f"""
Manual download (2 minutes):
  1. Open https://universe.roboflow.com/ and search "garbage detection"
  2. Pick a dataset with 1000+ images and 5+ classes
  3. Download Dataset -> YOLOv8 -> "show download code"
  4. Paste the rf.workspace(...).project(...).version(...).download("yolov8")
     line into CANDIDATES-style form in this file
  5. Re-run:  python download_dataset.py

Or skip Roboflow entirely:
     python dataset.py            # TrashNet, no account, reproducible split
"""


def count_images(root: Path) -> int:
    total = 0
    for split in ("train", "val", "test"):
        folder = root / "images" / split
        if folder.exists():
            total += len(list(folder.glob("*.*")))
    return total


def try_download(api_key: str, workspace: str, project: str, version: int, dest: Path) -> bool:
    from roboflow import Roboflow

    print(f"  trying {workspace}/{project} v{version} ...", end=" ", flush=True)
    try:
        Roboflow(api_key=api_key).workspace(workspace).project(project).version(version).download(
            "yolov8", location=str(dest)
        )
    except Exception as exc:  # noqa: BLE001 - any download failure means "try the next one"
        print(f"failed ({exc.__class__.__name__})")
        return False
    images = count_images(dest)
    print(f"ok ({images} images)")
    return images > 0


def main() -> int:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"ERROR: {API_KEY_ENV} is not set.")
        print()
        print(MANUAL_INSTRUCTIONS)
        return 2

    print(f"Downloading a garbage detection dataset into {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)

    for workspace, project, version in CANDIDATES:
        if try_download(api_key, workspace, project, version, DEST):
            yaml_path = DEST / "data.yaml"
            print()
            print("Dataset ready.")
            print(f"  images : {count_images(DEST)}")
            print(f"  root   : {DEST}")
            if yaml_path.exists():
                print(f"  data.yaml : {yaml_path}  <- pass this file to your trainer")
            print("  next   : python train.py   (or the verified pipeline, train_real.py)")
            return 0
        DEST.mkdir(parents=True, exist_ok=True)

    print()
    print("Every candidate failed.")
    print(MANUAL_INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
