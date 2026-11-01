#!/usr/bin/env python3
"""Pick each model's routing threshold on validation, then score it on test.

Comparing architectures at a single fixed confidence threshold is not a fair
comparison: an undertrained (or heavily regularised) model is less confident, so a
threshold that suits the shipped model silently converts its extra *correctness*
into extra `residual` refusals. This was measured twice in this project - once
against the 160 px run and once against a 30-epoch run - so the sweep is now part
of the protocol rather than a one-off.

For every run on the command line, in one inference pass per split:

  1. sweep the threshold on the VALIDATION split, keep the best operating point
     (highest routing accuracy, ties broken by higher coverage);
  2. score the TEST split at that threshold and report it next to top-1, macro-F1,
     parameter count and per-image latency.

    python tools/arch_sweep.py arch30_mobilenet_v3_small arch30_resnet18 arch30_efficientnet_b0
    python tools/arch_sweep.py --runs arch30_* --markdown --out smart_bin/runs/arch_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "smart_bin"))

from dataset import TRASHNET_DIR  # noqa: E402
from fusion import SensorReading, decide  # noqa: E402
from model import eval_transform, load_checkpoint  # noqa: E402
from routing import stream_for  # noqa: E402

RUNS = REPO / "smart_bin" / "runs"
GRID = [x / 100 for x in range(15, 71, 5)]


def evaluate(model, dataset, tta: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """One pass over a split -> (probabilities, targets, ms per image)."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    probabilities, targets, latency = [], [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            started = time.perf_counter()
            probs = torch.softmax(model(images), 1)
            if tta:
                probs = 0.5 * (probs + torch.softmax(model(torch.flip(images, dims=[3])), 1))
            latency.append((time.perf_counter() - started) * 1000 / max(len(images), 1))
            probabilities.append(probs.numpy())
            targets.append(batch_targets.numpy())
    return np.concatenate(probabilities), np.concatenate(targets), float(np.mean(latency))


def routing_at(probs: np.ndarray, targets: np.ndarray, labels, threshold: float):
    true_streams = [stream_for(name)["stream"] for name in labels]
    routed = refused = 0
    for row, target in zip(probs, targets):
        decision = decide(list(labels), row, sensor=SensorReading(), threshold=threshold)
        refused += decision.to_residual
        routed += decision.stream == true_streams[target]
    total = len(targets)
    return routed / total, refused / total


def macro_f1(probs: np.ndarray, targets: np.ndarray, n_classes: int) -> float:
    predictions = probs.argmax(1)
    scores = []
    for index in range(n_classes):
        true_positive = int(((predictions == index) & (targets == index)).sum())
        predicted = int((predictions == index).sum())
        support = int((targets == index).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def prune(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def sweep(probs: np.ndarray, targets: np.ndarray, labels):
    rows = []
    for threshold in GRID:
        routing, refused = routing_at(probs, targets, labels, threshold)
        rows.append({"threshold": threshold, "routing": round(routing, 4),
                     "residual_rate": round(refused, 4), "coverage": round(1 - refused, 4)})
    # Among the thresholds that are within noise of the best routing accuracy,
    # keep the HIGHEST one: it refuses at least as much as the optimum requires,
    # so the fail-safe keeps doing useful work instead of being quietly disabled.
    ceiling = max(row["routing"] for row in rows)
    plateau = [row for row in rows if row["routing"] >= ceiling - 0.005]
    best = max(plateau, key=lambda row: row["threshold"])
    return rows, best


def run_one(name: str, root: Path, tta: bool, val_probs_cache: dict) -> dict:
    run_dir = RUNS / name
    model, labels, imgsz = load_checkpoint(run_dir / "best.pt")
    model.eval()

    if name not in val_probs_cache:
        val = ImageFolder(root / "val", transform=eval_transform(imgsz))
        if val.classes != labels:
            raise SystemExit(f"{name}: val classes {val.classes} != checkpoint labels {labels}")
        val_probs_cache[name] = (*evaluate(model, val, tta)[:2],)
    val_probs, val_targets = val_probs_cache[name]

    grid, best = sweep(val_probs, val_targets, labels)

    test = ImageFolder(root / "test", transform=eval_transform(imgsz))
    test_probs, test_targets, latency = evaluate(model, test, tta)

    threshold = best["threshold"]
    test_routing, test_refused = routing_at(test_probs, test_targets, labels, threshold)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    return prune(
        run=name,
        imgsz=imgsz,
        params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        epochs=summary.get("epochs"),
        minutes=summary.get("minutes"),
        seconds_per_epoch=summary.get("seconds_per_epoch"),
        val_best_threshold=threshold,
        val_routing=best["routing"],
        val_routing_at_040=next(r["routing"] for r in grid if r["threshold"] == 0.40),
        threshold_for_test=threshold,
        test_top1=round(float((test_probs.argmax(1) == test_targets).mean()), 4),
        test_macro_f1=round(macro_f1(test_probs, test_targets, len(labels)), 4),
        test_routing=round(test_routing, 4),
        test_residual_rate=round(test_refused, 4),
        test_latency_ms=round(latency, 2),
        test_routing_at_040=round(routing_at(test_probs, test_targets, labels, 0.40)[0], 4),
        val_sweep=grid,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run folder names under smart_bin/runs")
    parser.add_argument("--runs-from", default=None, help="glob pattern, e.g. 'arch30_*'")
    parser.add_argument("--data", type=Path, default=TRASHNET_DIR)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    names = list(args.runs)
    if args.runs_from:
        names += sorted(path.name for path in RUNS.glob(args.runs_from))
    if not names:
        parser.error("give run names or --runs-from")

    torch.set_num_threads(args.threads)
    cache: dict = {}
    results = []
    for name in names:
        if not (RUNS / name / "best.pt").exists():
            print(f"[skip] {name}: no best.pt")
            continue
        print(f"[sweep] {name} ...", flush=True)
        results.append(run_one(name, args.data, args.tta, cache))

    if args.markdown:
        print()
        print("| run | params | test top-1 | macro-F1 | val-tuned threshold | test routing | routing @ 0.40 |")
        print("|---|---|---|---|---|---|---|")
        for r in results:
            print(f"| `{r['run']}` | {r['params_m']:.2f} M | {r['test_top1'] * 100:.2f} % | "
                  f"{r['test_macro_f1'] * 100:.2f} % | {r['val_best_threshold']:.2f} | "
                  f"{r['test_routing'] * 100:.2f} % | {r['test_routing_at_040'] * 100:.2f} % |")
    else:
        print()
        print(f"  {'run':<30}{'params':>9}{'top-1':>9}{'macro-F1':>10}"
              f"{'tau(val)':>10}{'routing':>9}{'@0.40':>8}{'ms/img':>8}")
        for r in results:
            print(f"  {r['run']:<30}{r['params_m']:>7.2f} M{r['test_top1'] * 100:>8.2f}%"
                  f"{r['test_macro_f1'] * 100:>9.2f}%{r['val_best_threshold']:>10.2f}"
                  f"{r['test_routing'] * 100:>8.2f}%{r['test_routing_at_040'] * 100:>7.2f}%"
                  f"{r['test_latency_ms']:>8.1f}")
        print("\n  routing is reported at each model's own validation-tuned threshold, and again"
              "\n  at the shipped 0.40 so the cost of not re-tuning is visible.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"tta": args.tta, "results": results}, indent=2) + "\n")
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
