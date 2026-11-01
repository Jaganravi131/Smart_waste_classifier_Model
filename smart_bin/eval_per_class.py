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

# Use the checkpoint's own class order. The previous version indexed model.names
# with an index built from the *sorted folder names*, so a single missing folder
# silently mislabelled every prediction.
names = model.names
if isinstance(names, dict):
    classes = [names[i] for i in sorted(names)]
else:
    classes = list(names)
folders = {d.name for d in val_dir.iterdir() if d.is_dir()}
missing = sorted(set(folders) - set(classes))
if missing:
    print(f"WARNING: validation folders not present in the model: {missing}")
idx = {c: i for i, c in enumerate(classes)}
skipped = 0
cm = np.zeros((len(classes), len(classes)), dtype=int)

total = 0
correct = 0

for true_class in classes:
    folder = val_dir / true_class
    if not folder.exists():
        continue
    for img in folder.glob("*.*"):
        r = model(str(img), verbose=False)[0]
        pred_idx = int(r.probs.top1)
        pred_name = names[pred_idx] if isinstance(names, dict) else names[pred_idx]
        if pred_name not in idx:
            # Count it, but as a miss rather than dropping it from the denominator.
            skipped += 1
            total += 1
            continue
        total += 1
        ti, pi = idx[true_class], idx[pred_name]
        cm[ti, pi] += 1
        if ti == pi:
            correct += 1

print("CLASSES:", ", ".join(classes))
print(f"TOTAL: {total}")
print(f"TOP1_ACC: {correct/total:.4f}")
print(f"SKIPPED_UNKNOWN_PREDICTIONS: {skipped}")
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
