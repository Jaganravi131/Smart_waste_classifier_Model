#!/usr/bin/env python3
"""Score the *deployed* ONNX artefact, with the same fusion rules as the app.

`evaluate_real.py` measures the PyTorch checkpoint.  The thing that ships is the
quantised ONNX file, and int8 quantisation changes both the accuracy and the
confidence distribution - the confidence part matters because the residual
threshold sits on the top-1 probability.  This script therefore repeats the
routing measurement on `backend.probabilities()` directly, so a threshold picked
here is a threshold that means the same thing in the demo.

    python tools/eval_onnx.py --run bingenie_mnv3s_128 --split val --sweep
    python tools/eval_onnx.py --run bingenie_mnv3s_128 --split test --threshold 0.30 --tta

Horizontal-flip TTA is averaged in probability space, matching `--tta` in
`evaluate_real.py`.  Latency is per image and includes decoding, so it is the
honest end-to-end number rather than the ONNX session time alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "smart_bin"))

from backends import OnnxBackend  # noqa: E402
from fusion import decide  # noqa: E402
from routing import stream_for  # noqa: E402

RUNS_DIR = ROOT / "smart_bin" / "runs"
DATA_DIR = ROOT / "smart_bin" / "datasets" / "trashnet"


def collect(root: Path, split: str) -> tuple[list[Path], list[int], list[str]]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise SystemExit(f"missing split directory: {split_dir}")
    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    paths: list[Path] = []
    targets: list[int] = []
    for index, name in enumerate(classes):
        for path in sorted((split_dir / name).glob("*.jpg")):
            paths.append(path)
            targets.append(index)
    return paths, targets, classes


def probabilities(backend, path: Path, tta: bool) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        probs = backend.probabilities(image)
        if tta:
            probs = (probs + backend.probabilities(image.transpose(Image.FLIP_LEFT_RIGHT))) / 2
    return probs


def score(backend, paths, targets, classes, threshold: float, tta: bool):
    true_streams = [str(stream_for(name)["stream"]) for name in classes]
    correct = routed = refused = 0
    confidence: list[float] = []
    latency: list[float] = []
    confusion = np.zeros((len(classes), len(classes)), dtype=int)

    for path, target in zip(paths, targets):
        started = time.perf_counter()
        probs = probabilities(backend, path, tta)
        latency.append((time.perf_counter() - started) * 1000)

        pred = int(probs.argmax())
        confusion[target, pred] += 1
        correct += pred == target
        confidence.append(float(probs.max()))

        decision = decide(
            class_names=classes,
            probabilities=probs,
            sensor=None,
            threshold=threshold,
        )
        refused += bool(decision.to_residual)
        routed += decision.stream == true_streams[target]

    total = len(targets)
    recall = {}
    for index, name in enumerate(classes):
        support = int(confusion[index].sum())
        recall[name] = round(float(confusion[index, index] / support) if support else 0.0, 4)

    return {
        "images": total,
        "threshold": threshold,
        "tta": tta,
        "top1_accuracy": round(correct / total, 4),
        "routing_accuracy_streams": round(routed / total, 4),
        "residual_rate": round(refused / total, 4),
        "coverage": round(1 - refused / total, 4),
        "mean_confidence": round(float(np.mean(confidence)), 4),
        "latency_ms_per_image": round(float(np.mean(latency)), 2),
        "per_class_recall": recall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="bingenie_mnv3s_128", help="run folder name under smart_bin/runs")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--sweep", action="store_true", help="report routing across thresholds 0.20-0.65")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run
    if not run_dir.is_dir():
        raise SystemExit(f"no such run: {run_dir}")

    wanted_int8 = args.precision == "int8"
    candidates = [p for p in sorted(run_dir.glob("*.onnx")) if p.stem.endswith("_int8") == wanted_int8]
    if not candidates:
        raise SystemExit(f"no {args.precision} .onnx in {run_dir} - export it first")
    onnx_path = candidates[0]
    backend = OnnxBackend(onnx_path, threads=args.threads)
    paths, targets, classes = collect(args.data, args.split)
    if classes != backend.labels:
        raise SystemExit(f"folder classes {classes} do not match sidecar labels {backend.labels}")

    print("=" * 70)
    print("BinGenie - deployed ONNX evaluation")
    print("=" * 70)
    print(f"  artefact  : {onnx_path.name}  ({backend.describe()})")
    print(f"  split     : {args.split} ({len(paths)} images)  TTA: {args.tta}")
    print()

    result = score(backend, paths, targets, classes, args.threshold, args.tta)
    print(f"  top-1            : {result['top1_accuracy'] * 100:.2f} %")
    print(f"  routing          : {result['routing_accuracy_streams'] * 100:.2f} %  (threshold {args.threshold:.2f})")
    print(f"  residual rate    : {result['residual_rate'] * 100:.2f} %")
    print(f"  mean confidence  : {result['mean_confidence']:.3f}")
    print(f"  latency          : {result['latency_ms_per_image']:.1f} ms/image")
    print("  per-class recall : " + ", ".join(f"{k} {v * 100:.1f}" for k, v in result["per_class_recall"].items()))

    if args.sweep:
        sweep = []
        for candidate in [x / 100 for x in range(20, 66, 5)]:
            row = score(backend, paths, targets, classes, candidate, args.tta)
            sweep.append({k: row[k] for k in ("threshold", "routing_accuracy_streams", "residual_rate", "coverage")})
        result["threshold_sweep"] = sweep
        best = max(sweep, key=lambda r: (r["routing_accuracy_streams"], r["coverage"]))
        print("\n  threshold sweep (routing, measured on this split):")
        print(f"    {'threshold':>9}{'routing':>10}{'refused':>10}{'coverage':>10}")
        for row in sweep:
            marker = "  <- best" if row is best else ""
            print(
                f"    {row['threshold']:>9.2f}{row['routing_accuracy_streams'] * 100:>9.2f}%"
                f"{row['residual_rate'] * 100:>9.2f}%{row['coverage'] * 100:>9.2f}%{marker}"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
