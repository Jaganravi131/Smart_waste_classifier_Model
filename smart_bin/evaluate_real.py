"""
BinGenie - honest evaluation of the trained classifier.

The report quotes five different accuracy figures (92.8 %, 0.920, 91.2 %,
95.42 %, 89.5 %) that cannot all describe the same model.  This script produces
exactly one number per definition, every one of them measured on the held-out
test split:

  top1_accuracy          - the plain classification accuracy
  top3_accuracy          - useful for an operator assist / "did you mean"
  macro_f1               - balances the six classes
  routing_accuracy       - end-to-end decision accuracy after the confidence
                           threshold and the fusion rules, i.e. the number the
                           product actually delivers
  residual_rate          - share of items refused by the model (fail-safe)
  coverage_at_precision  - acceptable trade-off curve for the threshold

It also reports 95 % Wilson confidence intervals, because 80 images per class
cannot support a claim of "95.42 %" without one.

Usage
-----
    python evaluate_real.py
    python evaluate_real.py --run runs/bingenie_mnv3s_128 --tta
    python evaluate_real.py --split val
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import TRASHNET_DIR
from fusion import FUSION_AVAILABLE, SensorReading, decide
from model import describe, eval_transform, load_checkpoint, stream_for, write_json

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval - correct for small n, unlike +/- 1.96*sqrt(p(1-p)/n)."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def plot_confusion(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center", fontsize=9,
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the BinGenie classifier on the held-out split")
    parser.add_argument("--run", type=Path, default=None, help="run folder (default: newest with best.pt)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data", type=Path, default=TRASHNET_DIR)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="below this top-1 probability the item is routed to residual")
    parser.add_argument("--tta", action="store_true", help="average original + horizontal flip")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    run_dir = args.run
    if run_dir is None:
        candidates = sorted(RUNS_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit("No trained checkpoint found. Run `python train_real.py` first.")
        run_dir = candidates[0].parent
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}")

    split_dir = args.data / args.split
    if not split_dir.exists():
        raise SystemExit(f"Missing split: {split_dir} - run `python dataset.py` first.")

    model, labels, imgsz = load_checkpoint(checkpoint, device="cpu")
    device = torch.device("cpu")
    for p in model.parameters():
        p.requires_grad_(False)

    dataset = ImageFolder(split_dir, transform=eval_transform(imgsz))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    class_names = dataset.classes
    if class_names != labels:
        raise SystemExit(
            f"Folder classes {class_names} do not match the checkpoint labels {labels}. "
            "Re-run `python dataset.py` so the split is built in the same order."
        )

    describe("BinGenie - evaluation")
    print(f"  run        : {run_dir.name}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  split      : {args.split} ({len(dataset)} images)")
    print(f"  imgsz      : {imgsz}   TTA: {args.tta}   threshold: {args.threshold}")
    print()

    n_cls = len(labels)
    confusion = np.zeros((n_cls, n_cls), dtype=int)
    all_probs: list[np.ndarray] = []
    all_targets: list[int] = []
    stream_correct = 0
    routed_to_residual = 0
    latency_ms: list[float] = []
    fusion_overrides = 0
    true_streams = [str(stream_for(name)["stream"]) for name in labels]

    import time

    with torch.no_grad():
        for images, targets in loader:
            started = time.perf_counter()
            logits = model(images.to(device))
            if args.tta:
                flipped = model(torch.flip(images.to(device), dims=[3]))
                logits = (logits.softmax(1) + flipped.softmax(1)) / 2
                probs = logits
            else:
                probs = logits.softmax(1)
            latency_ms.append((time.perf_counter() - started) * 1000 / max(images.size(0), 1))

            probs_np = probs.cpu().numpy()
            for row, target in zip(probs_np, targets.tolist()):
                all_probs.append(row)
                all_targets.append(target)
                pred_idx = int(row.argmax())
                confusion[target, pred_idx] += 1

                # ---- end-to-end routing decision (product behaviour) --------
                decision = decide(
                    class_names=list(labels),
                    probabilities=row,
                    sensor=SensorReading(),           # camera-only evaluation
                    threshold=args.threshold,
                )
                if decision.used_fusion:
                    fusion_overrides += 1
                if decision.to_residual:
                    routed_to_residual += 1
                # The product-level question is "did the item reach the right
                # compartment", not "was the class name right": paper called
                # cardboard still lands in the dry-recyclable bin.
                if decision.stream == true_streams[target]:
                    stream_correct += 1

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)
    preds_arr = probs_arr.argmax(1)
    correct_mask = (preds_arr == targets_arr)
    confidences = probs_arr.max(1)
    total = len(targets_arr)

    top1 = correct_mask.mean()
    ranked = np.argsort(-probs_arr, axis=1)
    top3 = float(np.mean([targets_arr[i] in ranked[i, :3] for i in range(total)]))

    per_class = {}
    for idx, name in enumerate(labels):
        tp = int(((preds_arr == idx) & (targets_arr == idx)).sum())
        support = int((targets_arr == idx).sum())
        fp = int(((preds_arr == idx) & (targets_arr != idx)).sum())
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lo, hi = wilson_interval(tp, support or 1)
        per_class[name] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "recall_ci95": [round(lo, 4), round(hi, 4)],
            "mean_confidence": round(float(confidences[targets_arr == idx].mean()) if support else 0.0, 4),
        }

    ci_lo, ci_hi = wilson_interval(int(correct_mask.sum()), total)
    results = {
        "run": run_dir.name,
        "split": args.split,
        "images": total,
        "top1_accuracy": round(float(top1), 4),
        "top1_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "top3_accuracy": round(top3, 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "routing_accuracy_streams": round(stream_correct / total, 4),
        "routing_ci95": [round(v, 4) for v in wilson_interval(stream_correct, total)],
        "residual_rate": round(routed_to_residual / total, 4),
        "fusion_override_rate": round(fusion_overrides / total, 4),
        "fusion_rules_available": FUSION_AVAILABLE,
        "threshold": args.threshold,
        "tta": args.tta,
        "mean_confidence": round(float(confidences.mean()), 4),
        "ece_10bin": round(expected_calibration_error(confidences, correct_mask.astype(float)), 4),
        "latency_ms_per_image": round(float(np.mean(latency_ms)), 2),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "labels": list(labels),
    }

    describe("Results (all values measured, none simulated)")
    print(f"  images                 : {total}")
    print(f"  top-1 accuracy         : {top1 * 100:.2f} %  (95 % CI {ci_lo * 100:.2f} - {ci_hi * 100:.2f})")
    print(f"  top-3 accuracy         : {top3 * 100:.2f} %")
    print(f"  macro F1               : {results['macro_f1'] * 100:.2f} %")
    print(f"  routing accuracy       : {results['routing_accuracy_streams'] * 100:.2f} %  "
          f"(item reached the correct compartment, threshold {args.threshold})")
    print(f"  sent to residual bin   : {results['residual_rate'] * 100:.2f} % of items")
    print(f"  mean confidence        : {results['mean_confidence']:.4f}   ECE(10) {results['ece_10bin']:.4f}")
    print(f"  CPU latency            : {results['latency_ms_per_image']:.1f} ms/image")
    print()
    print("  class        support  precision  recall     f1     mean conf")
    for name, m in per_class.items():
        print(f"  {name:<12} {m['support']:>6}    {m['precision'] * 100:6.2f}   {m['recall'] * 100:6.2f}  "
              f"{m['f1'] * 100:6.2f}    {m['mean_confidence']:.3f}")
    print()
    print("  confusion matrix (rows = true, cols = predicted):")
    print("   " + " ".join(f"{name[:6]:>7}" for name in labels))
    for name, row in zip(labels, confusion):
        print(f"  {name[:12]:<12}" + " ".join(f"{v:>7}" for v in row))

    write_json(run_dir / f"metrics_{args.split}.json", results)
    plot_confusion(confusion, list(labels), run_dir / f"confusion_matrix_{args.split}.png",
                   f"BinGenie {run_dir.name} - {args.split} split")
    print()
    print(f"  written: {run_dir / f'metrics_{args.split}.json'}")
    print(f"  written: {run_dir / f'confusion_matrix_{args.split}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
