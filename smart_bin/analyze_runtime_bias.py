"""Per-class threshold analysis.

The thresholds below are a *study*, not a calibration: keep them in this file and
copy the accepted values into one place (routing.py / predict.py) once measured -
the previous version had a second hard-coded copy inside detect_smartbin.py, and
the two drifted apart.
"""
from pathlib import Path
import random
from ultralytics import YOLO

BASE = Path(__file__).parent
model = YOLO(str(BASE / "runs" / "smart_bin_v1" / "weights" / "best.pt"))
val_dir = BASE / "datasets" / "cls" / "val"

random.seed(42)
max_per_class = 60

# Proposed stricter thresholds per class (can be tuned)
class_min_conf = {
    "battery": 0.65,
    "biological": 0.65,
    "cardboard": 0.70,
    "clothes": 0.70,
    "glass": 0.72,
    "metal": 0.72,
    "paper": 0.70,
    "plastic": 0.78,
    "shoes": 0.70,
    "trash": 0.75,
}
margin_min = 0.12

raw_total = 0
raw_correct = 0
raw_plastic_nonplastic = 0

smart_total = 0
smart_correct = 0
smart_accepted = 0
smart_plastic_nonplastic = 0

for cls_dir in sorted([d for d in val_dir.iterdir() if d.is_dir()]):
    true_cls = cls_dir.name
    imgs = list(cls_dir.glob("*.*"))
    if len(imgs) > max_per_class:
        imgs = random.sample(imgs, max_per_class)
    for img in imgs:
        r = model(str(img), verbose=False)[0]
        probs = r.probs.data.tolist()
        top1 = int(r.probs.top1)
        top1_conf = float(r.probs.top1conf)
        pred = model.names[top1]

        # raw rule: global threshold 0.6 only
        raw_total += 1
        if top1_conf >= 0.60:
            if pred == true_cls:
                raw_correct += 1
            if pred == "plastic" and true_cls != "plastic":
                raw_plastic_nonplastic += 1

        # smart rule: per-class threshold + top2 margin
        smart_total += 1
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top2_conf = float(sorted_probs[1][1]) if len(sorted_probs) > 1 else 0.0
        margin = top1_conf - top2_conf
        needed = class_min_conf.get(pred, 0.70)

        accept = (top1_conf >= needed) and (margin >= margin_min)
        if accept:
            smart_accepted += 1
            if pred == true_cls:
                smart_correct += 1
            if pred == "plastic" and true_cls != "plastic":
                smart_plastic_nonplastic += 1

print("RAW_RULE")
print(f"samples={raw_total} accepted={raw_total} correct={raw_correct} acc={raw_correct/raw_total:.4f}")
print(f"false_plastic_on_nonplastic={raw_plastic_nonplastic}")

print("\nSMART_RULE")
print(f"samples={smart_total} accepted={smart_accepted}")
print(f"correct={smart_correct} acc_on_accepted={(smart_correct/smart_accepted if smart_accepted else 0):.4f}")
print(f"false_plastic_on_nonplastic={smart_plastic_nonplastic}")
