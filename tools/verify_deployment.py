#!/usr/bin/env python3
"""
verify_deployment.py - score the artefact that actually ships.

`evaluate_real.py` measures the PyTorch checkpoint. That is the right thing to do
while training, but it is not the thing the demo runs: the demo loads the ONNX
export and pre-processes with PIL + NumPy. A viva examiner is entitled to ask
"your 78.89 % was measured on the training stack - what does the shipped model
actually score?", and until this script existed the answer was "trust me".

This script therefore imports **no PyTorch at all**. It loads whatever
`backends.make_backend()` would load at serve time, replays the held-out split
through it, and reports the same metric definitions as `evaluate_real.py` so the
two JSON files can be compared line by line.

    python tools/verify_deployment.py                       # newest .onnx in runs/
    python tools/verify_deployment.py --onnx path/to.onnx --split val
    python tools/verify_deployment.py --compare             # diff vs metrics_<split>.json

The dependency list is numpy + pillow + onnxruntime: this script runs happily in
the 48 MB runtime-only environment, which is also how you prove that the demo
needs no training stack.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SMART_BIN = REPO / "smart_bin"
sys.path.insert(0, str(SMART_BIN))

from backends import find_artifacts, make_backend  # noqa: E402
from fusion import SensorReading, decide  # noqa: E402
from routing import stream_for  # noqa: E402

DATA = SMART_BIN / "datasets" / "trashnet"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval - the same one evaluate_real.py uses."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidences)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.sum():
            error += mask.sum() / total * abs(correct[mask].mean() - confidences[mask].mean())
    return float(error)


def load_split(root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    items: list[tuple[Path, int]] = []
    for index, name in enumerate(labels):
        for image in sorted((root / name).glob("*.jpg")):
            items.append((image, index))
    return items, labels


def evaluate(backend, items, labels, threshold: float) -> dict:
    probabilities, targets, latencies = [], [], []
    stream_correct = residual = fusion_overrides = 0

    for image_path, target in items:
        image = Image.open(image_path).convert("RGB")
        started = time.perf_counter()
        probs = backend.probabilities(image)
        latencies.append((time.perf_counter() - started) * 1000.0)

        probabilities.append(probs)
        targets.append(target)

        # The product question is "did the item reach the right compartment?",
        # not "was the class name right" - paper called cardboard still belongs
        # in the dry stream.
        decision = decide(class_names=list(labels), probabilities=probs,
                          sensor=SensorReading(), threshold=threshold)
        if decision.used_fusion:
            fusion_overrides += 1
        if decision.to_residual:
            residual += 1
        if decision.stream == stream_for(labels[target])["stream"]:
            stream_correct += 1

    probs_arr = np.asarray(probabilities)
    targets_arr = np.asarray(targets)
    preds = probs_arr.argmax(1)
    correct = (preds == targets_arr)
    confidences = probs_arr.max(1)
    total = len(targets_arr)

    ranked = np.argsort(-probs_arr, axis=1)
    top3 = float(np.mean([targets_arr[i] in ranked[i, :3] for i in range(total)]))

    per_class = {}
    for index, name in enumerate(labels):
        true_positive = int(((preds == index) & (targets_arr == index)).sum())
        support = int((targets_arr == index).sum())
        false_positive = int(((preds == index) & (targets_arr != index)).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        low, high = wilson(true_positive, support or 1)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "recall_ci95": [round(low, 4), round(high, 4)],
            "mean_confidence": round(float(confidences[targets_arr == index].mean()), 4) if support else 0.0,
        }

    top1 = float(correct.mean())
    low, high = wilson(int(correct.sum()), total)
    route_low, route_high = wilson(stream_correct, total)

    return {
        "images": total,
        "top1_accuracy": round(top1, 4),
        "top1_ci95": [round(low, 4), round(high, 4)],
        "top3_accuracy": round(top3, 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "routing_accuracy_streams": round(stream_correct / total, 4),
        "routing_ci95": [round(route_low, 4), round(route_high, 4)],
        "residual_rate": round(residual / total, 4),
        "fusion_override_rate": round(fusion_overrides / total, 4),
        "threshold": threshold,
        "mean_confidence": round(float(confidences.mean()), 4),
        "ece_10bin": round(expected_calibration_error(confidences, correct.astype(float)), 4),
        "latency_ms_per_image": round(float(np.mean(latencies)), 2),
        "per_class": per_class,
        "labels": list(labels),
    }


def compare(results: dict, reference_path: Path) -> None:
    if not reference_path.exists():
        print(f"\n[compare] {reference_path.relative_to(REPO)} not found - nothing to compare against")
        return
    reference = json.loads(reference_path.read_text())
    print(f"\n[compare] deployed vs {reference_path.name} (PyTorch training stack)")
    print(f"  {'metric':<26}{'deployed':>10}{'pytorch':>10}{'delta':>10}")
    for key, label in (("top1_accuracy", "top-1 accuracy"),
                       ("top3_accuracy", "top-3 accuracy"),
                       ("macro_f1", "macro F1"),
                       ("routing_accuracy_streams", "routing accuracy"),
                       ("residual_rate", "sent to residual"),
                       ("mean_confidence", "mean confidence")):
        if key not in reference:
            continue
        deployed_value = results[key]
        reference_value = reference[key]
        delta = (deployed_value - reference_value) * 100
        print(f"  {label:<26}{deployed_value * 100:>9.2f}%{reference_value * 100:>9.2f}%{delta:>+9.2f} pp")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the shipped ONNX artefact (no PyTorch required)")
    parser.add_argument("--onnx", type=Path, default=None, help="export to score (default: newest in runs/)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--data", type=Path, default=DATA, help="dataset root holding the split folders")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None,
                        help="JSON to write (default: <run>/metrics_deployed_<split>.json)")
    parser.add_argument("--compare", action="store_true", help="diff against metrics_<split>.json")
    args = parser.parse_args()

    onnx_path = args.onnx or find_artifacts()[1]
    if onnx_path is None:
        raise SystemExit("[deploy] no .onnx export found - run smart_bin/onnx_export.py first")
    root = args.data / args.split
    if not root.exists():
        raise SystemExit(f"[deploy] no split at {root} - run smart_bin/dataset.py first")

    items, labels = load_split(root)
    backend = make_backend("onnx", onnx_path=onnx_path, threads=args.threads)
    print("=" * 70)
    print(f"deployed artefact : {onnx_path.relative_to(REPO) if onnx_path.is_relative_to(REPO) else onnx_path}")
    print(f"backend           : {backend.describe()}   ({onnx_path.stat().st_size / 1e6:.2f} MB)")
    print(f"split             : {args.split} ({len(items)} images)   threshold {args.threshold}")
    if sorted(labels) != sorted(backend.labels):
        raise SystemExit(f"[deploy] label mismatch: dataset {labels} vs model {backend.labels}")
    print("=" * 70)

    started = time.perf_counter()
    results = evaluate(backend, items, labels, args.threshold)
    results.update({
        "split": args.split,
        "onnx": onnx_path.name,
        "backend": backend.describe(),
        "onnx_bytes": onnx_path.stat().st_size,
        "runtime": "numpy + pillow + onnxruntime (no PyTorch)",
    })

    print(f"  top-1 accuracy         : {results['top1_accuracy'] * 100:.2f} %  "
          f"(95 % CI {results['top1_ci95'][0] * 100:.2f} - {results['top1_ci95'][1] * 100:.2f})")
    print(f"  top-3 accuracy         : {results['top3_accuracy'] * 100:.2f} %")
    print(f"  macro F1               : {results['macro_f1'] * 100:.2f} %")
    print(f"  routing accuracy       : {results['routing_accuracy_streams'] * 100:.2f} %  "
          f"(correct compartment, threshold {args.threshold})")
    print(f"  sent to residual bin   : {results['residual_rate'] * 100:.2f} % of items")
    print(f"  mean confidence        : {results['mean_confidence']:.4f}   ECE(10) {results['ece_10bin']:.4f}")
    print(f"  latency                : {results['latency_ms_per_image']:.1f} ms/image "
          f"({time.perf_counter() - started:.1f} s wall clock)")

    print(f"\n  {'class':<11}{'support':>8}{'precision':>11}{'recall':>9}{'f1':>8}{'recall 95 % CI':>18}")
    for name, values in results["per_class"].items():
        low, high = values["recall_ci95"]
        print(f"  {name:<11}{values['support']:>8}{values['precision'] * 100:>10.2f}%"
              f"{values['recall'] * 100:>8.2f}%{values['f1'] * 100:>7.2f}%"
              f"{low * 100:>10.1f}-{high * 100:<7.1f}")

    if args.compare:
        compare(results, onnx_path.parent / f"metrics_{args.split}.json")

    report = args.report or onnx_path.parent / f"metrics_deployed_{args.split}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n  written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
