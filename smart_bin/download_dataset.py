"""
Download a pre-annotated garbage detection dataset from Roboflow Universe.

This script tries two well-known public datasets. If both fail, it prints manual
instructions so you can pick any dataset from Roboflow Universe yourself.

Steps:
  1. Set the ROBOFLOW_API_KEY environment variable
  2. Run:  python download_dataset.py
"""

import os
import sys
from pathlib import Path

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
  raise RuntimeError("Set the ROBOFLOW_API_KEY environment variable before downloading.")

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "trash")

# Ordered list of public datasets to try (workspace, project, version)
CANDIDATES = [
  ("roboflow-universe-projects", "garbage-classification-3",      3),
  ("roboflow-universe-projects", "garbage-detection-3",           1),
  ("material-identification",    "garbage-classification-3",      2),
  ("tannergi",                   "garbage_detection-wvzwv",       9),
]


def try_download(rf, workspace, project_name, version, dest):
  try:
    print(f"  Trying: {workspace}/{project_name} v{version} ...", end=" ", flush=True)
    proj    = rf.workspace(workspace).project(project_name)
    dataset = proj.version(version).download("yolov8", location=dest)
    print("OK")
    return dataset
  except Exception as e:
    print(f"FAILED ({e.__class__.__name__})")
    return None


def count_images(dest):
  total = 0
  for split in ("train", "val", "test"):
    folder = Path(dest) / "images" / split
    if folder.exists():
      total += len(list(folder.glob("*.*")))
  return total


from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)

print("Downloading garbage detection dataset from Roboflow Universe...")
print(f"Destination: {DEST}\n")

dataset = None
for ws, proj, ver in CANDIDATES:
  dataset = try_download(rf, ws, proj, ver, DEST)
  if dataset:
    imgs = count_images(DEST)
    if imgs > 0:
      break
    else:
      print("  → Downloaded but no images found, trying next...")
      dataset = None

if dataset and count_images(DEST) > 0:
  imgs = count_images(DEST)
  # Find the data.yaml roboflow created
  yaml_path = Path(DEST) / "data.yaml"
  print(f"\n✓ Dataset ready!")
  print(f"  Images downloaded : {imgs}")
  print(f"  Location          : {DEST}")
  if yaml_path.exists():
    print(f"  data.yaml         : {yaml_path}")
  print(f"\nNext step: python train.py")
else:
  print("\n" + "=" * 65)
  print("  AUTO-DOWNLOAD FAILED — Manual download required (2 min)")
  print("=" * 65)
  print("""
  1. Open: https://universe.roboflow.com/
  2. Search: "garbage detection"
  3. Pick any dataset with 1000+ images and 5+ classes
   Recommended: "Garbage Classification" by material-identification
  4. Click  Download Dataset  →  YOLOv8  →  show download code
  5. Copy the Python snippet shown, paste ONLY the rf.workspace(...)
   lines into this file replacing the CANDIDATES loop,
   and change location to:
""")
  print(f'     location=r"{DEST}"')
  print("""
  6. Re-run:  python download_dataset.py
""")
  sys.exit(1)
