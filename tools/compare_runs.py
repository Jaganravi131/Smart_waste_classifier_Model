#!/usr/bin/env python3
"""
compare_runs.py - is the new run actually better than the shipped one?

`evaluate_real.py` scores one checkpoint. Twice in this project that was not
enough: a retrain can post a higher headline accuracy while getting *worse* at the
thing the product does (routing), and with 431 test images a 1-2 point difference
is inside the noise. This script scores two checkpoints on the identical split,
reports both metric sets with confidence intervals, and runs a McNemar test on the
paired predictions so "better" means better beyond the sampling noise.

    python tools/compare_runs.py --baseline bingenie_mnv3s_128 --candidate bingenie_mnv3s_160
    python tools/compare_runs.py --baseline ... --candidate ... --split val --tta

McNemar is exact (binomial) for small discordant counts, which is what 431 test
images produce: it only looks at the images where the two models disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "smart_bin"))

from dataset import TRASHNET_DIR  # noqa: E402
from fusion import SensorReading, decide  # noqa: E402
from model import eval_transform, load_checkpoint  # noqa: E402
from routing import stream_for  # noqa: E402


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(baseline_correct: np.ndarray, candidate_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired binary outcomes. Returns (b, c, two-sided p)."""
    only_baseline = int(np.sum(baseline_correct & ~candidate_correct))
    only_candidate = int(np.sum(~baseline_correct & candidate_correct))
    discordant = only_baseline + only_candidate
    if discordant == 0:
        return only_baseline, only_candidate, 1.0
    smaller = min(only_baseline, only_candidate)
    # two-sided exact binomial test with p = 0.5
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2 ** discordant
    return only_baseline, only_candidate, min(1.0, 2 * tail)


def score_checkpoint(run: Path, root: Path, split: str, threshold: float, tta: bool):
    checkpoint = run / "best.pt"
    model, labels, imgsz = load_checkpoint(checkpoint)
    model.eval()

    dataset = ImageFolder(root / split, transform=eval_transform(imgsz))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    probabilities, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            logits = model(images)
            probs = torch.softmax(logits, 1)
            if tta:
                probs = 0.5 * (probs + torch.softmax(model(torch.flip(images, dims=[3])), 1))
            probabilities.append(probs.numpy())
            targets.append(batch_targets.numpy())

    probs = np.concatenate(probabilities)
    targets = np.concatenate(targets)
    predictions = probs.argmax(1)

    correct = predictions == targets
    routed_correct = np.zeros(len(targets), dtype=bool)
    for index, (row, target) in enumerate(zip(probs, targets)):
        decision = decide(list(labels), row, sensor=SensorReading(), threshold=threshold)
        routed_correct[index] = decision.stream == stream_for(labels[target])["stream"]

    per_class = {}
    for index, name in enumerate(labels):
        support = int((targets == index).sum())
        hits = int(((predictions == index) & (targets == index)).sum())
        per_class[name] = {"recall": hits / support if support else 0.0, "support": support}

    return {
        "run": run.name,
        "imgsz": imgsz,
        "images": int(len(targets)),
        "top1": float(correct.mean()),
        "top1_ci": wilson(int(correct.sum()), len(targets)),
        "routing": float(routed_correct.mean()),
        "routing_ci": wilson(int(routed_correct.sum()), len(targets)),
        "per_class": per_class,
        "correct": correct,
        "routed_correct": routed_correct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two runs on the same split")
    parser.add_argument("--baseline", default="bingenie_mnv3s_128", help="run folder name under smart_bin/runs")
    parser.add_argument("--candidate", required=True, help="run folder name under smart_bin/runs")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--data", type=Path, default=TRASHNET_DIR)
    parser.add_argument("--tta", action="store_true", help="average original + horizontal flip")
    parser.add_argument("--json", type=Path, default=None, help="write the comparison as JSON")
    args = parser.parse_args()

    runs = REPO / "smart_bin" / "runs"
    baseline = score_checkpoint(runs / args.baseline, args.data, args.split, args.threshold, args.tta)
    candidate = score_checkpoint(runs / args.candidate, args.data, args.split, args.threshold, args.tta)

    print("=" * 78)
    print(f"split {args.split} ({baseline['images']} images)   threshold {args.threshold}"
          f"{'   TTA on' if args.tta else ''}")
    print("=" * 78)
    print(f"  {'':<22}{baseline['run']:>22}{candidate['run']:>22}")
    print(f"  {'input size':<22}{baseline['imgsz']:>19} px{candidate['imgsz']:>19} px")
    for label, key, ci_key in (("top-1 accuracy", "top1", "top1_ci"),
                               ("routing accuracy", "routing", "routing_ci")):
        b, c = baseline[key], candidate[key]
        low_b, high_b = baseline[ci_key]
        low_c, high_c = candidate[ci_key]
        print(f"  {label:<22}{b * 100:>15.2f} %  {c * 100:>15.2f} %   "
              f"({(c - b) * 100:+.2f} pp)")
        print(f"  {'  95 % CI':<22}{low_b * 100:>12.1f}-{high_b * 100:<4.1f}"
              f"{low_c * 100:>10.1f}-{high_c * 100:<4.1f}")

    print(f"\n  {'class':<12}{'support':>8}{'recall (base)':>15}{'recall (new)':>16}")
    for name, values in baseline["per_class"].items():
        new = candidate["per_class"][name]["recall"]
        print(f"  {name:<12}{values['support']:>8}{values['recall'] * 100:>14.1f} %{new * 100:>15.1f} %")

    for label, key in (("top-1", "correct"), ("routing", "routed_correct")):
        b_only, c_only, p_value = mcnemar_exact(baseline[key], candidate[key])
        verdict = "significant" if p_value < 0.05 else "not significant"
        sign = "better" if c_only > b_only else "worse" if c_only < b_only else "tied"
        print(f"\n  McNemar on {label} correctness: only-baseline {b_only}, "
              f"only-candidate {c_only}  ->  p = {p_value:.4f}  ({verdict}, candidate {sign})")

    if args.json:
        payload = {
            "split": args.split,
            "threshold": args.threshold,
            "tta": args.tta,
            "baseline": {k: v for k, v in baseline.items() if k not in ("correct", "routed_correct")},
            "candidate": {k: v for k, v in candidate.items() if k not in ("correct", "routed_correct")},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n  written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
