from pathlib import Path
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).parent
weights = BASE / "runs" / "smart_bin_v1" / "weights" / "best.pt"
val_dir = BASE / "datasets" / "cls" / "val"

if not weights.exists():
    raise FileNotFoundError(f"Weights not found: {weights}")
if not val_dir.exists():
    raise FileNotFoundError(f"Validation folder not found: {val_dir}")

model = YOLO(str(weights))
classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
idx = {c: i for i, c in enumerate(classes)}
cm = np.zeros((len(classes), len(classes)), dtype=int)

total = 0
correct = 0

for true_class in classes:
    for img in (val_dir / true_class).glob("*.*"):
        total += 1
        r = model(str(img), verbose=False)[0]
        pred_idx = int(r.probs.top1)
        pred_name = model.names[pred_idx]
        if pred_name not in idx:
            continue
        ti, pi = idx[true_class], idx[pred_name]
        cm[ti, pi] += 1
        if ti == pi:
            correct += 1

print("CLASSES:", ", ".join(classes))
print(f"TOTAL: {total}")
print(f"TOP1_ACC: {correct/total:.4f}")
print("\nPER_CLASS_METRICS")
print("class,support,precision,recall,f1")

for c in classes:
    i = idx[c]
    tp = cm[i, i]
    support = cm[i, :].sum()
    fp = cm[:, i].sum() - tp
    rec = tp / support if support else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    print(f"{c},{support},{prec:.4f},{rec:.4f},{f1:.4f}")

print("\nCONFUSION_MATRIX_ROWS=true_COLS=pred")
print("classes:", classes)
print(cm)
