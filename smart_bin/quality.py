"""Frame-quality guard: refuse a photograph the classifier was never trained on.

The robustness sweep showed the failure the confidence threshold cannot see. At a
third of the normal brightness the model drops from 78.65 % to 33.18 % top-1 while
refusing only 5.8 % of items: it stays confidently wrong, because a softmax is not a
measure of whether the input looks like the training set. A three-line check on the
frame itself - how bright it is, how much contrast it has, how sharp it is - catches
that case before the model speaks.

Deliberately simple. There is no reference model, no histogram comparison and no
extra dependency: three scalars computed with numpy on a 256-pixel thumbnail, so the
guard costs well under a millisecond and can also run on the Pi.

Calibration, measured on the validation split (328 images):

    condition      luminance (mean, p1-p99)   contrast   sharpness
    clean          0.646  [0.422-0.792]       0.177      0.0285
    dark  x0.50    0.322  [0.210-0.395]       0.089      0.0150
    dark  x0.30    0.192  [0.125-0.236]       0.053      0.0094
    blur radius 2  0.646  [0.422-0.793]       0.173      0.0109

The limits below are set so that clean frames pass with margin and severe
under-lighting is refused. What it does *not* catch, and the model card says so:
over-exposure (TrashNet's background is already white, so a clipping test is
uninformative), sensor noise (noise inflates the sharpness metric) and a cluttered
background at normal brightness. This is a tripwire for the lighting failure mode,
not a general out-of-distribution detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

#: A frame below this mean luminance has lost most of the signal the model trained on.
DEFAULT_LIMITS: Dict[str, float] = {
    "luminance_min": 0.30,      # clean validation frames never fall below 0.344
    "contrast_min": 0.07,       # clean 5th percentile is 0.098; 0.10 refused 5 % of good frames
    "sharpness_min": 0.005,     # a blur-2 frame averages 0.011, so this only trips on real blur
}


@dataclass
class QualityReport:
    ok: bool
    metrics: Dict[str, float]
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "metrics": self.metrics, "reasons": self.reasons}


def frame_metrics(image: Image.Image, max_side: int = 256) -> Dict[str, float]:
    """Luminance, contrast and gradient energy of an RGB frame, cheaply."""
    grey = image.convert("L")
    if max(grey.size) > max_side:
        grey = grey.copy()
        grey.thumbnail((max_side, max_side), Image.BILINEAR)
    array = np.asarray(grey, dtype=np.float32) / 255.0
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 3:
        return {"luminance": 0.0, "contrast": 0.0, "sharpness": 0.0}

    # discrete Laplacian: four neighbours against the centre pixel
    laplacian = (4.0 * array[1:-1, 1:-1]
                 - array[:-2, 1:-1] - array[2:, 1:-1]
                 - array[1:-1, :-2] - array[1:-1, 2:])
    return {
        "luminance": float(array.mean()),
        "contrast": float(array.std()),
        "sharpness": float(np.abs(laplacian).mean()),
    }


def assess_quality(image: Image.Image, limits: Optional[Dict[str, float]] = None) -> QualityReport:
    """Return whether this frame is worth classifying, and why not."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    metrics = {name: round(value, 4) for name, value in frame_metrics(image).items()}
    reasons: List[str] = []

    if metrics["luminance"] < limits["luminance_min"]:
        reasons.append(
            f"frame too dark (luminance {metrics['luminance']:.2f} < {limits['luminance_min']:.2f})"
        )
    if metrics["contrast"] < limits["contrast_min"]:
        reasons.append(
            f"frame has too little contrast ({metrics['contrast']:.2f} < {limits['contrast_min']:.2f})"
        )
    if metrics["sharpness"] < limits["sharpness_min"]:
        reasons.append(
            f"frame is out of focus (sharpness {metrics['sharpness']:.3f} < {limits['sharpness_min']:.3f})"
        )
    return QualityReport(ok=not reasons, metrics=metrics, reasons=reasons)


def self_test() -> None:
    """Cheap invariants - the guard must not fire on a normal frame."""
    bright = Image.new("RGB", (64, 64), (200, 200, 200))
    for x in range(0, 64, 8):                      # add structure so it is not flat
        for y in range(64):
            bright.putpixel((x, y), (40, 40, 40))
    assert assess_quality(bright).ok, "a well-lit, sharp frame must pass"

    black = Image.new("RGB", (64, 64), (8, 8, 8))
    report = assess_quality(black)
    assert not report.ok and any("dark" in reason for reason in report.reasons)

    flat = Image.new("RGB", (64, 64), (128, 128, 128))
    assert not assess_quality(flat).ok, "a flat grey frame has no contrast"
    print("quality guard self-test: ok")


if __name__ == "__main__":
    self_test()
