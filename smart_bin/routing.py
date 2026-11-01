"""
BinGenie - the class -> compartment map, in one place, with no heavy imports.

Why this file exists
--------------------
The previous code base had the routing table in three incompatible places:
`detect_smartbin.py` invented seven bins (BIN_1 ... BIN_7), the report describes
four colour-coded compartments, and Appendix I's fusion layer hard-coded class
*indices* (0, 1, 2) with no names attached.  Any change to the class list silently
changed where an item physically went.

Now everything - training, evaluation, the CLI, the web demo and the YOLOv8 script
that the team still uses - imports this module.  It depends on nothing but the
standard library, so it can be imported on a Raspberry Pi without PyTorch.
"""

from __future__ import annotations

from typing import Dict

# The four physical compartments, in carousel slot order.
STREAMS: Dict[str, Dict[str, object]] = {
    "biodegradable": {"bin": "GREEN", "slot": 0, "label": "Wet / Organic (green)"},
    "recyclable": {"bin": "YELLOW", "slot": 1, "label": "Dry Recyclable (yellow)"},
    "electronic": {"bin": "ORANGE", "slot": 2, "label": "Electronic / Hazardous (orange)"},
    "residual": {"bin": "RED", "slot": 3, "label": "Residual / Hazardous (red)"},
}

# Dataset class -> stream.  Two vocabularies are covered because the project has
# used two datasets; extend the table, never hard-code a bin at a call site.
CLASS_TO_STREAM: Dict[str, str] = {
    # TrashNet (6 classes) - the dataset data.yaml and the shipped model use
    "cardboard": "recyclable",
    "glass": "recyclable",
    "metal": "recyclable",       # metal is the most valuable dry recyclable, not e-waste
    "paper": "recyclable",
    "plastic": "recyclable",
    "trash": "residual",
    # 10-class Kaggle garbage set - what prepare_dataset.py builds
    "biological": "biodegradable",
    "battery": "electronic",
    "clothes": "residual",
    "shoes": "residual",
    "e-waste": "electronic",
    "ewaste": "electronic",
    "electronic waste": "electronic",
    "medical waste": "residual",
    "organic": "biodegradable",
    "organic waste": "biodegradable",
}

DEFAULT_STREAM = "residual"


def stream_for(class_name: str, fallback: str = DEFAULT_STREAM) -> Dict[str, object]:
    """Return {'stream', 'bin', 'slot', 'label'} for a predicted class name."""
    key = CLASS_TO_STREAM.get(str(class_name).strip().lower(), fallback)
    info = dict(STREAMS[key])
    info["stream"] = key
    return info


def bin_label(class_name: str) -> str:
    """Human-readable compartment label, e.g. 'YELLOW - Dry Recyclable (yellow)'."""
    info = stream_for(class_name)
    return f"{info['bin']} - {info['label']}"


def unknown_classes(class_names) -> list[str]:
    """Classes in a dataset that have no compartment yet - fail loudly, not silently."""
    return [name for name in class_names if name.strip().lower() not in CLASS_TO_STREAM]


if __name__ == "__main__":
    for name in ("plastic", "battery", "biological", "trash", "mystery"):
        info = stream_for(name)
        print(f"{name:<12} -> slot {info['slot']}  {info['bin']:<6} {info['stream']}")
