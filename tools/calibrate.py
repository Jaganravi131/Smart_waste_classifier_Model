#!/usr/bin/env python3
"""Is the model's confidence telling the truth, and where should the gate sit?

Two questions the report could not answer before. First, softmax output is not a
probability: the shipped model has an expected calibration error of 0.055, fitted by
temperature scaling on the validation split. Second, the 0.40 residual threshold was
chosen by hand - this script replaces that guess with a risk-coverage curve, and
reports the threshold that holds the wrong-compartment rate at a chosen level.

Everything is measured on the int8 ONNX artefact by default, because quantisation
shifts the confidence distribution: a temperature fitted on fp32 logits is not the
temperature the deployed model needs.

    python tools/calibrate.py                          # val-fit, test-report
    python tools/calibrate.py --target-wrong 0.01 --figure smart_bin/runs/calibration.png
    python tools/calibrate.py --precision fp32 --out smart_bin/runs/calibration_fp32.json

Temperature is fitted by minimising validation NLL with a coarse grid followed by a
golden-section search - no scipy, no torch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "smart_bin"))

from backends import OnnxBackend  # noqa: E402
from model import load_checkpoint  # noqa: E402
from routing import stream_for  # noqa: E402

RUNS = REPO / "smart_bin" / "runs"
DATA = REPO / "smart_bin" / "datasets" / "trashnet"


# --------------------------------------------------------------------------- #
# calibration primitives
# --------------------------------------------------------------------------- #
def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Divide the log-probabilities by T and renormalise (temperature scaling)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(probabilities: np.ndarray, targets: np.ndarray) -> float:
    picked = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    return float(-np.mean(np.log(picked)))


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray,
                               bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - confidence[mask].mean())
        error += gap * mask.sum() / total
    return float(error)


def fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Return (temperature, nll_after) by minimising NLL on the given split."""
    grid = np.arange(0.50, 3.01, 0.05)
    losses = [nll(apply_temperature(probabilities, t), targets) for t in grid]
    best_index = int(np.argmin(losses))

    # golden-section refinement around the best grid point
    low = max(0.05, grid[max(best_index - 1, 0)])
    high = grid[min(best_index + 1, len(grid) - 1)]
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a = high - phi * (high - low)
        b = low + phi * (high - low)
        if nll(apply_temperature(probabilities, a), targets) < nll(apply_temperature(probabilities, b), targets):
            high = b
        else:
            low = a
    temperature = float(round((low + high) / 2, 3))
    return temperature, nll(apply_temperature(probabilities, temperature), targets)


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def collect(backend, split_dir: Path, labels: list[str]):
    images, targets = [], []
    for index, name in enumerate(labels):
        for path in sorted((split_dir / name).glob("*.jpg")):
            with Image.open(path) as handle:
                images.append(handle.convert("RGB").copy())
            targets.append(index)
    probabilities = np.array([backend.probabilities(image) for image in images])
    return probabilities, np.array(targets)


def risk_coverage(probabilities: np.ndarray, targets: np.ndarray, labels: list[str]) -> list[dict]:
    """Selective accuracy and wrong-compartment rate as the gate tightens."""
    confidence = probabilities.max(axis=1)
    predicted_stream = np.array([str(stream_for(str(labels[i]))["stream"])
                                 for i in probabilities.argmax(axis=1)])
    true_stream = np.array([str(stream_for(str(labels[int(t)]))["stream"]) for t in targets])
    correct = probabilities.argmax(axis=1) == targets
    routed_right = predicted_stream == true_stream

    curve = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        accepted = confidence >= threshold
        if not accepted.any():
            continue
        curve.append({
            "threshold": round(float(threshold), 2),
            "coverage": round(float(accepted.mean()), 4),
            "selective_accuracy": round(float(correct[accepted].mean()), 4),
            "wrong_bin_rate": round(float((~routed_right[accepted]).mean()), 4),
        })
    return curve


def pick_threshold(curve: list[dict], target_wrong: float) -> dict | None:
    """Lowest threshold whose wrong-compartment rate meets the target."""
    feasible = [row for row in curve if row["wrong_bin_rate"] <= target_wrong]
    if not feasible:
        return None
    return max(feasible, key=lambda row: (row["coverage"], -row["threshold"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="bingenie_mnv3s_128")
    parser.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--target-wrong", type=float, default=0.01,
                        help="acceptable rate of wrong-compartment decisions among accepted items")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    run_dir = RUNS / args.run
    files = [p for p in sorted(run_dir.glob("*.onnx"))
             if p.stem.endswith("_int8") == (args.precision == "int8")]
    if not files:
        raise SystemExit(f"no {args.precision} ONNX in {run_dir}")
    backend = OnnxBackend(files[0], threads=args.threads)
    labels = backend.labels

    print("=" * 78)
    print("BinGenie - calibration and risk-coverage")
    print("=" * 78)
    print(f"  artefact : {files[0].name}")
    print()

    val_probs, val_targets = collect(backend, DATA / "val", labels)
    test_probs, test_targets = collect(backend, DATA / "test", labels)

    temperature, val_nll_after = fit_temperature(val_probs, val_targets)
    calibrated_val = apply_temperature(val_probs, temperature)
    calibrated_test = apply_temperature(test_probs, temperature)

    rows = {
        "val": (val_probs, calibrated_val, val_targets),
        "test": (test_probs, calibrated_test, test_targets),
    }
    summary = {"run": args.run, "artefact": files[0].name, "temperature": temperature}
    print(f"  {'split':<6}{'ECE before':>12}{'ECE after':>11}{'NLL before':>12}{'NLL after':>11}")
    for split, (raw, calibrated, targets) in rows.items():
        before, after = expected_calibration_error(raw, targets), expected_calibration_error(calibrated, targets)
        nll_before, nll_after = nll(raw, targets), nll(calibrated, targets)
        print(f"  {split:<6}{before:>12.4f}{after:>11.4f}{nll_before:>12.4f}{nll_after:>11.4f}")
        summary[split] = {"ece_before": round(before, 4), "ece_after": round(after, 4),
                          "nll_before": round(nll_before, 4), "nll_after": round(nll_after, 4)}

    curve_calibrated = risk_coverage(calibrated_test, test_targets, labels)
    curve_raw = risk_coverage(test_probs, test_targets, labels)
    chosen_calibrated = pick_threshold(curve_calibrated, args.target_wrong)
    chosen_raw = pick_threshold(curve_raw, args.target_wrong)

    print(f"\n  target: wrong-compartment rate <= {args.target_wrong * 100:.1f} % among accepted items")
    for name, chosen in (("raw", chosen_raw), ("calibrated", chosen_calibrated)):
        if chosen is None:
            print(f"  {name:<12} no threshold reaches the target")
            continue
        print(f"  {name:<12} threshold {chosen['threshold']:.2f}  coverage "
              f"{chosen['coverage'] * 100:5.1f}%  selective accuracy "
              f"{chosen['selective_accuracy'] * 100:5.2f}%  wrong-bin {chosen['wrong_bin_rate'] * 100:.2f}%")
    summary["risk_coverage_calibrated"] = curve_calibrated
    summary["risk_coverage_raw"] = curve_raw
    summary["chosen_calibrated"] = chosen_calibrated
    summary["chosen_raw"] = chosen_raw

    print(f"\n  risk-coverage (calibrated, test split)")
    print(f"    {'tau':>6}{'coverage':>10}{'selective acc':>15}{'wrong-bin':>11}")
    for row in curve_calibrated:
        if row["threshold"] in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            print(f"    {row['threshold']:>6.2f}{row['coverage'] * 100:>9.1f}%"
                  f"{row['selective_accuracy'] * 100:>14.2f}%{row['wrong_bin_rate'] * 100:>10.2f}%")

    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))

        # reliability diagram: confidence vs accuracy, raw and calibrated
        for name, probs, colour in (("raw", test_probs, "#94a3b8"),
                                    (f"T = {temperature:.2f}", calibrated_test, "#2563eb")):
            confidence = probs.max(axis=1)
            correct = (probs.argmax(axis=1) == test_targets).astype(float)
            edges = np.linspace(0, 1, 11)
            centres, accuracies = [], []
            for low, high in zip(edges[:-1], edges[1:]):
                mask = (confidence > low) & (confidence <= high)
                if mask.sum() < 5:
                    continue
                centres.append(confidence[mask].mean())
                accuracies.append(correct[mask].mean())
            left.plot(centres, accuracies, marker="o", markersize=4, color=colour, label=name)
        left.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1, label="perfect")
        left.set_xlabel("predicted confidence")
        left.set_ylabel("observed accuracy")
        left.set_title("Reliability, test split")
        left.legend(fontsize=8)
        left.grid(alpha=0.3)

        ordered = sorted(curve_calibrated, key=lambda row: row["coverage"])
        coverage = [row["coverage"] * 100 for row in ordered]
        selective = [row["selective_accuracy"] * 100 for row in ordered]
        wrong = [row["wrong_bin_rate"] * 100 for row in ordered]
        line = right.plot(coverage, selective, color="#2563eb", label="accuracy on accepted")[0]
        right.set_xlabel("coverage (% of items accepted)")
        right.set_ylabel("accuracy on accepted (%)")
        right.grid(alpha=0.3)

        # the wrong-compartment rate is an order of magnitude smaller - give it its own axis
        twin = right.twinx()
        twin.plot(coverage, wrong, color="#dc2626", label="wrong-compartment rate")
        twin.set_ylabel("wrong-compartment rate (%)")
        lines = line.get_label() and [line] + twin.get_lines()
        right.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower left")
        right.set_title("Risk-coverage, calibrated")

        figure.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.figure, dpi=160)
        print(f"\n  written: {args.figure}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
