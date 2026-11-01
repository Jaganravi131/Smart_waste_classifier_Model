"""
BinGenie - sensor fusion and fail-safe routing.

Rewritten from the version quoted in Appendix I.2 of the report, which could
never have worked:

    def fusion_override(label, s):
        if s['moisture'] > 600 and s['inductive'] == 0:
            return 0                       # biodegradable
        if s['inductive'] == 1:
            return 2                       # electronic / metal
        if s['moisture'] > 600:
            return 0
        return 1                           # dry recyclable

  * `return 1` fires for every input, so the "override" replaced the vision
    verdict on 100 % of items, not on the ~6 % of ambiguous ones the report
    claims.  The four-stream bin becomes a three-stream bin that always answers.
  * metal was routed to the *electronic* compartment although metal is the most
    valuable dry recyclable and the report's own Table 5.1/6.1 treat it as a
    recyclable class;
  * the first and third rules are logically identical (the `inductive == 0`
    guard is dead code);
  * `fusion_agrees()` referenced `CLASS_INDEX`, which is defined nowhere, and
    the moisture threshold was a raw ADC count (600) with no unit or scaling.

The design rule used here instead: a sensor may only *contradict* the camera
when it is actually carrying evidence, and when it abstains the camera decides.
Every decision records why it was taken, so the log is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from model import CLASS_TO_STREAM, STREAMS, stream_for

FUSION_AVAILABLE = True

# Moisture is expressed as normalised wetness in [0, 1] where 0 = dry,
# 1 = dripping.  The device normalises the raw resistive reading against a
# dry reference captured during self-test, so the number has a meaning and is
# comparable between units.  A recyclable item above this level is assumed to
# be contaminated and is diverted to the wet stream (this is what protects the
# recyclate price, which is the product's main claim).
MOISTURE_WET_THRESHOLD = 0.65

# Fill level in percent at which a compartment is declared full.
COMPARTMENT_FULL_PERCENT = 95

# Gas readings do not route anything - they raise a hygiene alert.  Present in
# the code so that a reading cannot silently change a routing decision.
GAS_ALERT_PPM = 400


@dataclass
class SensorReading:
    """One snapshot of the physical evidence. `None` means 'no evidence'."""

    moisture: Optional[float] = None      # 0..1 normalised wetness
    inductive: Optional[bool] = None      # metal detected in the throat
    gas_ppm: Optional[float] = None       # MQ-135, advisory only
    fill_percent: Dict[str, int] = field(default_factory=dict)  # stream -> % full
    weight_g: Optional[float] = None


@dataclass
class Decision:
    label: str
    stream: str
    confidence: float
    margin: float
    used_fusion: bool = False
    to_residual: bool = False
    reasons: List[str] = field(default_factory=list)

    @property
    def compartment(self) -> str:
        return str(STREAMS[self.stream]["bin"])

    @property
    def slot(self) -> int:
        return int(STREAMS[self.stream]["slot"])

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "stream": self.stream,
            "compartment": self.compartment,
            "slot": self.slot,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "used_fusion": self.used_fusion,
            "to_residual": self.to_residual,
            "reasons": self.reasons,
        }


def _residual_label(class_names: Sequence[str]) -> str:
    for name in class_names:
        if CLASS_TO_STREAM.get(name.lower()) == "residual":
            return name
    return "residual"


def decide(
    class_names: Sequence[str],
    probabilities: np.ndarray,
    sensor: Optional[SensorReading] = None,
    threshold: float = 0.40,
    margin_threshold: float = 0.0,
) -> Decision:
    """Turn a probability vector (+ optional sensor snapshot) into a routing decision."""
    sensor = sensor or SensorReading()
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if probabilities.size != len(class_names):
        raise ValueError(f"got {probabilities.size} probabilities for {len(class_names)} classes")

    order = np.argsort(-probabilities)
    top1_idx = int(order[0])
    top2_idx = int(order[1]) if len(order) > 1 else top1_idx
    confidence = float(probabilities[top1_idx])
    margin = float(probabilities[top1_idx] - probabilities[top2_idx])
    label = str(class_names[top1_idx])
    reasons: List[str] = []

    # --- 1. fail safe: never guess ----------------------------------------
    if confidence < threshold:
        reasons.append(f"confidence {confidence:.2f} < threshold {threshold:.2f}")
        return Decision(
            label=_residual_label(class_names),
            stream="residual",
            confidence=confidence,
            margin=margin,
            to_residual=True,
            reasons=reasons,
        )
    if margin_threshold > 0 and margin < margin_threshold:
        reasons.append(f"top1-top2 margin {margin:.2f} < {margin_threshold:.2f}")
        return Decision(
            label=_residual_label(class_names),
            stream="residual",
            confidence=confidence,
            margin=margin,
            to_residual=True,
            reasons=reasons,
        )

    stream = str(stream_for(label)["stream"])
    used_fusion = False

    # --- 2. metal overrides an optical verdict ----------------------------
    # An inductive sensor is a *direct* measurement of metal; the camera is a
    # proxy.  Only plastic/paper/cardboard/trash verdicts are contradicted -
    # glass and metal already agree with the sensor.
    if sensor.inductive is True and stream != "recyclable":
        reasons.append("inductive sensor: metal present -> dry recyclable")
        stream, label, used_fusion = "recyclable", label, True

    # --- 3. moisture overrides a "dry" verdict ----------------------------
    if sensor.moisture is not None and sensor.moisture >= MOISTURE_WET_THRESHOLD and stream == "recyclable":
        reasons.append(
            f"moisture {sensor.moisture:.2f} >= {MOISTURE_WET_THRESHOLD:.2f} -> contaminated, wet stream"
        )
        stream, used_fusion = "biodegradable", True

    # --- 4. gas is advisory, never routable -------------------------------
    if sensor.gas_ppm is not None and sensor.gas_ppm >= GAS_ALERT_PPM:
        reasons.append(f"gas {sensor.gas_ppm:.0f} ppm -> hygiene alert (does not change routing)")

    # --- 5. a full compartment cannot accept the item ---------------------
    fill = sensor.fill_percent.get(stream)
    if fill is not None and fill >= COMPARTMENT_FULL_PERCENT:
        reasons.append(f"compartment {stream} at {fill}% -> diverted to residual")
        return Decision(
            label=_residual_label(class_names),
            stream="residual",
            confidence=confidence,
            margin=margin,
            used_fusion=used_fusion,
            to_residual=True,
            reasons=reasons,
        )

    if not reasons:
        reasons.append("vision verdict accepted, no contradicting sensor evidence")
    return Decision(
        label=label,
        stream=stream,
        confidence=confidence,
        margin=margin,
        used_fusion=used_fusion,
        to_residual=False,
        reasons=reasons,
    )


def _self_test() -> None:
    names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    probs = lambda i, p=0.9: np.eye(len(names))[i] * p + (1 - p) / len(names)  # noqa: E731

    # 1. clean vision verdict, no sensor evidence -> vision is kept, not overridden
    d = decide(names, probs(4))
    assert d.stream == "recyclable" and not d.used_fusion, d

    # 2. low confidence -> residual (fail safe)
    d = decide(names, np.full(len(names), 1 / len(names)))
    assert d.to_residual and d.stream == "residual", d

    # 3. wet plastic -> wet stream
    d = decide(names, probs(4), SensorReading(moisture=0.9))
    assert d.stream == "biodegradable" and d.used_fusion, d

    # 4. metal detected while camera says "trash" -> rescued into recyclable
    d = decide(names, probs(5), SensorReading(inductive=True))
    assert d.stream == "recyclable" and d.used_fusion, d

    # 4b. camera already agrees with the sensor -> no override recorded
    d = decide(names, probs(4), SensorReading(inductive=True))
    assert d.stream == "recyclable" and not d.used_fusion, d

    # 5. full compartment -> residual
    d = decide(names, probs(1), SensorReading(fill_percent={"recyclable": 99}))
    assert d.to_residual, d

    # 6. gas alone does not move an item
    d = decide(names, probs(1), SensorReading(gas_ppm=900))
    assert d.stream == "recyclable", d

    print("fusion self-test: 7/7 checks passed")


if __name__ == "__main__":
    _self_test()
