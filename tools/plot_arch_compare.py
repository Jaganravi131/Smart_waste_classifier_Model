#!/usr/bin/env python3
"""Overlay the validation curves of several runs on one axis.

The report's architecture-selection section used to quote a ranking of eleven
networks that no run in this repository supported. This script produces the
figure that replaces it: every curve here is read from a `history.json` written
by `train_real.py`, so the figure cannot show a model that was never trained.

    python tools/plot_arch_compare.py arch30_mobilenet_v3_small arch30_resnet18 \\
        arch30_efficientnet_b0 --out smart_bin/runs/arch30_compare.png

Curves are compared against *epochs* (equal training budget) and the legend
carries the wall-clock cost per epoch, because on a 2-core CPU the interesting
question is accuracy per minute, not accuracy per parameter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "smart_bin" / "runs"


def load(run: Path) -> dict:
    history = json.loads((run / "history.json").read_text())
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {"history": history, "summary": summary, "name": run.name}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot several runs' validation curves together")
    parser.add_argument("runs", nargs="+", help="run folder names under smart_bin/runs")
    parser.add_argument("--out", type=Path, default=RUNS / "architecture_compare.png")
    parser.add_argument("--title", default="Validation accuracy at equal epoch budget")
    args = parser.parse_args()

    payloads = []
    for name in args.runs:
        run_dir = RUNS / name
        if not (run_dir / "history.json").exists():
            print(f"[skip] {name}: no history.json")
            continue
        payloads.append(load(run_dir))
    if not payloads:
        raise SystemExit("no runs with history.json found")

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for payload in payloads:
        history = payload["history"]
        epochs = history["epoch"]
        val = [value * 100 for value in history["val_acc"]]
        train = [value * 100 for value in history["train_acc"]]
        seconds = payload["summary"].get("seconds_per_epoch")
        label = payload["name"].replace("arch30_", "")
        if seconds:
            label += f"  ({seconds:.0f} s/epoch)"
        line = left.plot(epochs, val, linewidth=1.9, label=label)[0]
        left.plot(epochs, train, linewidth=1.0, linestyle=":", color=line.get_color(), alpha=0.8)

    left.set_xlabel("epoch")
    left.set_ylabel("accuracy (%)")
    left.set_title(args.title)
    left.grid(alpha=0.3)
    left.legend(fontsize=8, loc="lower right")
    left.text(0.02, 0.97, "solid = validation, dotted = train", transform=left.transAxes,
              fontsize=7, va="top", color="#555555")

    # right panel: does the extra compute buy accuracy per minute?
    for payload in payloads:
        summary = payload["summary"]
        seconds = summary.get("seconds_per_epoch")
        history = payload["history"]
        if not seconds:
            continue
        minutes = [epoch * seconds / 60 for epoch in history["epoch"]]
        left_color = right.plot(minutes, [value * 100 for value in history["val_acc"]],
                                linewidth=1.9)[0].get_color()
        right.plot(minutes, [value * 100 for value in history["val_acc"]],
                   linewidth=1.9, color=left_color,
                   label=payload["name"].replace("arch30_", ""))
    right.set_xlabel("CPU minutes spent (2 cores)")
    right.set_title("Accuracy against wall clock")
    right.grid(alpha=0.3)
    right.legend(fontsize=8, loc="lower right")

    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
