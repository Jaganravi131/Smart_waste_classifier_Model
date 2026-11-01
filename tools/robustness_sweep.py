#!/usr/bin/env python3
"""How far does the shipped model fall when the camera stops being kind?

Section 6.5 of the journal report asks for accuracy under dim light, blur, motion,
occlusion and edge-of-frame placement. A real capture campaign needs the production
rig, which does not exist in this repository - so this script does the honest
half of the job: it degrades the *held-out test split* deterministically, scores
the deployed int8 ONNX artefact on every variant, and labels the result for what
it is. Synthetic degradation is not a substitute for field images; it is an upper
bound on how much these factors can hurt, because the degradation is applied to
photographs whose subject is already centred and well lit.

    python tools/robustness_sweep.py                     # all conditions, int8
    python tools/robustness_sweep.py --conditions clean,blur2,dark
    python tools/robustness_sweep.py --out smart_bin/runs/robustness.json --figure ...

Each condition is applied to all images before inference, so the comparison is
paired: the script also runs an exact McNemar test of every condition against the
clean baseline, which is what stops a two-point drop being reported as a finding.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "smart_bin"))

from backends import OnnxBackend  # noqa: E402
from fusion import decide  # noqa: E402
from quality import assess_quality  # noqa: E402
from routing import stream_for  # noqa: E402

RUNS = REPO / "smart_bin" / "runs"
DATA = REPO / "smart_bin" / "datasets" / "trashnet"


# --------------------------------------------------------------------------- #
# degradations - deterministic, and each one is a plausible bin-side failure
# --------------------------------------------------------------------------- #
def _occlude(image: Image.Image, fraction: float) -> Image.Image:
    """A grey square (a hand, a sleeve, the edge of the funnel) over the subject."""
    width, height = image.size
    side = int(math.sqrt(fraction) * min(width, height))
    canvas = image.copy()
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    left = (width - side) // 2
    top = (height - side) // 3          # upper third: covers most material cues
    draw.rectangle([left, top, left + side, top + side], fill=(128, 128, 128))
    return canvas


def _noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.clip(array + rng.normal(0.0, sigma, array.shape), 0, 1)
    return Image.fromarray((array * 255).astype(np.uint8))


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _downscale(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    small = image.resize((max(1, int(width * factor)), max(1, int(height * factor))),
                         Image.BICUBIC)
    return small.resize((width, height), Image.BICUBIC)


CONDITIONS = {
    # name              description                                  transform
    "clean": ("as captured", lambda im: im),
    "dark": ("brightness ×0.50 (dim room, LEDs only)", lambda im: ImageEnhance.Brightness(im).enhance(0.50)),
    "very_dark": ("brightness ×0.30", lambda im: ImageEnhance.Brightness(im).enhance(0.30)),
    "bright": ("brightness ×1.60 (specular highlights)", lambda im: ImageEnhance.Brightness(im).enhance(1.60)),
    "low_contrast": ("contrast ×0.60", lambda im: ImageEnhance.Contrast(im).enhance(0.60)),
    "blur1": ("gaussian blur, radius 1", lambda im: im.filter(ImageFilter.GaussianBlur(1.0))),
    "blur2": ("gaussian blur, radius 2 (out of focus)", lambda im: im.filter(ImageFilter.GaussianBlur(2.0))),
    "noise": ("sensor noise σ = 0.03", lambda im: _noise(im, 0.03, 0)),
    "heavy_noise": ("sensor noise σ = 0.08", lambda im: _noise(im, 0.08, 1)),
    "jpeg_q30": ("JPEG quality 30", lambda im: _jpeg(im, 30)),
    "jpeg_q10": ("JPEG quality 10", lambda im: _jpeg(im, 10)),
    "rot10": ("rotated 10°", lambda im: im.rotate(10, fillcolor=(255, 255, 255))),
    "rot20": ("rotated 20° (item sliding)", lambda im: im.rotate(20, fillcolor=(255, 255, 255))),
    "occl10": ("10 % occluded (hand at the edge)", lambda im: _occlude(im, 0.10)),
    "occl25": ("25 % occluded", lambda im: _occlude(im, 0.25)),
    "small_item": ("item at half resolution (far / small)", lambda im: _downscale(im, 0.5)),
    "worst_realistic": ("dim + blur + JPEG + tilt together", lambda im: _jpeg(
        ImageEnhance.Brightness(im.filter(ImageFilter.GaussianBlur(1.0))).enhance(0.60).rotate(
            10, fillcolor=(255, 255, 255)), 40)),
}


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(baseline: np.ndarray, candidate: np.ndarray) -> tuple[int, int, float]:
    only_base = int(np.sum(baseline & ~candidate))
    only_cand = int(np.sum(~baseline & candidate))
    discordant = only_base + only_cand
    if discordant == 0:
        return only_base, only_cand, 1.0
    smaller = min(only_base, only_cand)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2 ** discordant
    return only_base, only_cand, min(1.0, 2 * tail)


def load_split(split_dir: Path, labels: list[str]) -> tuple[list[Image.Image], list[int], list[str]]:
    images, targets, paths = [], [], []
    for index, name in enumerate(labels):
        for path in sorted((split_dir / name).glob("*.jpg")):
            with Image.open(path) as handle:
                images.append(handle.convert("RGB").copy())
            targets.append(index)
            paths.append(str(path.relative_to(REPO)))
    return images, targets, paths


def score(backend, images, targets, labels, threshold: float, guard: bool = False):
    """Score one condition.

    With `guard=True` the frame-quality tripwire runs first: a rejected frame is
    counted as refused, so the *selective* accuracy (accuracy on the frames the
    system chose to act on) is the number that shows what the guard buys.
    """
    true_streams = [stream_for(name)["stream"] for name in labels]
    correct = []
    routed = []
    refused = []
    confidence = []
    latency = []
    for image, target in zip(images, targets):
        report = assess_quality(image) if guard else None
        started = time.perf_counter()
        probs = backend.probabilities(image)
        latency.append((time.perf_counter() - started) * 1000)
        prediction = int(probs.argmax())
        correct.append(prediction == target)
        confidence.append(float(probs.max()))
        decision = decide(labels, probs, sensor=None, threshold=threshold)
        refused.append(bool(decision.to_residual) or bool(report is not None and not report.ok))
        routed.append(decision.stream == true_streams[target])
    return (np.array(correct), np.array(routed), np.array(confidence),
            float(np.mean(latency)), np.array(refused), np.array(latency))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="bingenie_mnv3s_128")
    parser.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--conditions", default=None,
                        help="comma-separated subset of: " + ", ".join(CONDITIONS))
    parser.add_argument("--guard", action="store_true",
                        help="also run the frame-quality tripwire and report selective accuracy")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    run_dir = RUNS / args.run
    wanted_int8 = args.precision == "int8"
    files = [p for p in sorted(run_dir.glob("*.onnx")) if p.stem.endswith("_int8") == wanted_int8]
    if not files:
        raise SystemExit(f"no {args.precision} ONNX in {run_dir}")
    backend = OnnxBackend(files[0], threads=args.threads)

    images, targets, _ = load_split(DATA / args.split, backend.labels)
    names = [n.strip() for n in args.conditions.split(",")] if args.conditions else list(CONDITIONS)
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s): {unknown}")

    print("=" * 78)
    print("BinGenie - robustness to simulated capture conditions")
    print("=" * 78)
    print(f"  artefact : {files[0].name}   split {args.split} ({len(images)} images)"
          f"   threshold {args.threshold}")
    print()

    results = []
    baseline_correct = None
    for name in names:
        description, transform = CONDITIONS[name]
        started = time.time()
        degraded = [transform(image) for image in images]
        correct, routed, confidence, latency, refused, _ = score(
            backend, degraded, targets, backend.labels, args.threshold, guard=args.guard)
        accepted = ~refused
        entry = {
            "condition": name,
            "description": description,
            "top1": round(float(correct.mean()), 4),
            "top1_ci": [round(v, 4) for v in wilson(int(correct.sum()), len(correct))],
            "routing": round(float(routed.mean()), 4),
            "refused_rate": round(float(refused.mean()), 4),
            "mean_confidence": round(float(confidence.mean()), 4),
            "latency_ms": round(latency, 2),
            "seconds": round(time.time() - started, 1),
        }
        if args.guard:
            entry["coverage"] = round(float(accepted.mean()), 4)
            entry["selective_top1"] = (round(float(correct[accepted].mean()), 4)
                                       if accepted.any() else None)
            entry["wrong_but_accepted"] = round(float(((~correct) & accepted).mean()), 4)
        if name == "clean":
            baseline_correct = correct
        elif baseline_correct is not None:
            only_base, only_cond, p_value = mcnemar_exact(baseline_correct, correct)
            entry.update({"vs_clean_pp": round((entry["top1"] - float(baseline_correct.mean())) * 100, 2),
                          "mcnemar_only_clean": only_base, "mcnemar_only_condition": only_cond,
                          "p_value": round(p_value, 4)})
        results.append(entry)
        delta = entry.get("vs_clean_pp")
        marker = f"{delta:+6.2f} pp" if delta is not None else "   base "
        p_text = f"p={entry['p_value']:.4f}" if "p_value" in entry else ""
        print(f"  {name:<16}{entry['top1'] * 100:>7.2f}%  {marker}   routing "
              f"{entry['routing'] * 100:>6.2f}%   refused {entry['refused_rate'] * 100:>5.2f}%   {p_text}")
        if args.guard and entry.get("selective_top1") is not None:
            print(f"  {'':<16}  guard: coverage {entry['coverage'] * 100:5.1f}%   "
                  f"accuracy on accepted {entry['selective_top1'] * 100:5.2f}%   "
                  f"wrong-but-accepted {entry['wrong_but_accepted'] * 100:5.2f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"run": args.run, "precision": args.precision,
                                        "split": args.split, "threshold": args.threshold,
                                        "conditions": results}, indent=2) + "\n")
        print(f"\n  written: {args.out}")

    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ordered = sorted(results, key=lambda r: r["top1"])
        figure, axis = plt.subplots(figsize=(8.2, 0.34 * len(ordered) + 1.6))
        values = [r["top1"] * 100 for r in ordered]
        errors = [[(r["top1"] - r["top1_ci"][0]) * 100 for r in ordered],
                  [(r["top1_ci"][1] - r["top1"]) * 100 for r in ordered]]
        colours = ["#2563eb" if r["condition"] == "clean" else "#94a3b8" for r in ordered]
        axis.barh([r["condition"] for r in ordered], values, xerr=errors, color=colours,
                  error_kw={"ecolor": "#475569", "elinewidth": 1, "capsize": 3})
        clean_value = next(r["top1"] for r in results if r["condition"] == "clean") * 100
        axis.axvline(clean_value, color="#2563eb", linestyle="--", linewidth=1)
        axis.set_xlabel("top-1 accuracy on the held-out test split (%)")
        axis.set_title("Synthetic capture degradation (int8 model, 431 test images)")
        axis.grid(axis="x", alpha=0.3)
        figure.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.figure, dpi=160)
        print(f"  written: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
