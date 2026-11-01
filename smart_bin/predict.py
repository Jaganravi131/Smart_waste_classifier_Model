"""
BinGenie - inference: image, folder, video file or camera.

This is the deployable version of detections: one image in, one routing
decision out, with the compartment, the confidence, the reason for the
decision and an optional CSV row.  It replaces `detect_smartbin.py`, which

  * mapped classes to seven bins (BIN_1 ... BIN_7) although the product has
    four compartments,
  * logged every frame instead of every decision (the very problem §6.7 of the
    report diagnoses),
  * hard-coded per-class confidence floors that were never calibrated,
  * and crashed on a headless machine as soon as cv2.imshow was called.

Usage
-----
    python predict.py --image path/to/item.jpg
    python predict.py --folder datasets/trashnet/test/glass
    python predict.py --camera 0 --save-annotated out/
    python predict.py --video clip.mp4 --every 10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from fusion import SensorReading, decide
from model import STREAMS, eval_transform, load_checkpoint, stream_for

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
DEFAULT_LOG = BASE_DIR.parent / "logs" / "bingenie_decisions.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STREAM_COLORS = {  # BGR, matches the physical compartment colours
    "biodegradable": (60, 180, 75),
    "recyclable": (0, 200, 255),
    "electronic": (0, 140, 255),
    "residual": (40, 40, 220),
}


def newest_checkpoint() -> Path:
    candidates = sorted(RUNS_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("No trained checkpoint found. Run `python train_real.py` first.")
    return candidates[0]


class Classifier:
    """Wraps the checkpoint so that every entry point pre-processes identically."""

    def __init__(self, checkpoint: Optional[Path] = None, threads: int = 2, tta: bool = False):
        torch.set_num_threads(threads)
        self.checkpoint = checkpoint or newest_checkpoint()
        self.model, self.labels, self.imgsz = load_checkpoint(self.checkpoint)
        self.transform = eval_transform(self.imgsz)
        self.tta = tta
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def probabilities(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        logits = self.model(tensor).softmax(1)
        if self.tta:
            logits = (logits + self.model(torch.flip(tensor, dims=[3])).softmax(1)) / 2
        return logits.squeeze(0).numpy()

    def classify(self, image: Image.Image, threshold: float = 0.40, margin: float = 0.0,
                 sensor: Optional[SensorReading] = None):
        started = time.perf_counter()
        probs = self.probabilities(image)
        decision = decide(self.labels, probs, sensor=sensor, threshold=threshold, margin_threshold=margin)
        decision.latency_ms = (time.perf_counter() - started) * 1000  # type: ignore[attr-defined]
        ranking = sorted(zip(self.labels, probs.tolist()), key=lambda kv: -kv[1])
        return decision, ranking


def annotate(image: Image.Image, decision, ranking, latency_ms: float) -> Image.Image:
    """Draw the banner + a compact result card onto a copy of the frame."""
    from PIL import ImageDraw, ImageFont

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, img.width // 26))
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, img.width // 42))
    except OSError:  # pragma: no cover
        font_big = font = ImageFont.load_default()

    colour = STREAM_COLORS.get(decision.stream, (200, 200, 200))
    bar_h = int(img.height * 0.22)
    draw.rectangle([0, 0, img.width, bar_h], fill=(18, 18, 18, 215))
    draw.rectangle([0, 0, img.width, bar_h], outline=colour, width=4)
    draw.text((14, 8), f"{decision.label.upper()}  {decision.confidence * 100:.1f}%", font=font_big, fill=colour)
    draw.text(
        (14, 8 + int(font_big.size * 1.25)),
        f"-> {STREAMS[decision.stream]['label']}  |  {latency_ms:.0f} ms",
        font=font,
        fill=(235, 235, 235),
    )
    draw.text((14, 14 + int(font_big.size * 2.4)), decision.reasons[0][:64], font=font, fill=(190, 190, 190))

    y = img.height - 6 * (font.size + 2)
    for name, prob in ranking[:3]:
        draw.text((14, y), f"{name:<12} {prob * 100:5.1f}%", font=font, fill=(240, 240, 240))
        y += font.size + 2
    return img


class DecisionLog:
    """One CSV row per routed item - not per frame."""

    FIELDS = ["timestamp", "label", "stream", "compartment", "slot", "confidence",
              "margin", "used_fusion", "to_residual", "source", "image"]

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._new = not path.exists()
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if self._new:
            self._writer.writeheader()

    def write(self, decision, source: str, image: str = "") -> None:
        row = decision.as_dict()
        row.update(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
            image=image,
        )
        self._writer.writerow({key: row.get(key) for key in self.FIELDS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def iter_folder(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def run_images(classifier: Classifier, paths: Iterable[Path], args) -> int:
    log = DecisionLog(Path(args.log)) if not args.no_log else None
    out_dir = Path(args.save_annotated) if args.save_annotated else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    correct = 0
    total = 0
    for path in paths:
        image = Image.open(path)
        decision, ranking = classifier.classify(
            image, threshold=args.threshold, margin=args.margin, sensor=SensorReading()
        )
        latency = getattr(decision, "latency_ms", 0.0)
        counts[decision.stream] = counts.get(decision.stream, 0) + 1
        total += 1
        if path.parent.name and path.parent.name in classifier.labels:
            total_label = path.parent.name
            correct += int(stream_for(total_label)["stream"] == decision.stream)

        print(
            f"{path.name:<28} {decision.label:<12} {decision.confidence * 100:5.1f}%  "
            f"-> {decision.compartment:<6} ({decision.stream})  {latency:5.0f} ms"
            + ("  [FUSION]" if decision.used_fusion else "")
            + ("  [RESIDUAL]" if decision.to_residual else "")
        )
        if log:
            log.write(decision, source="image", image=str(path))
        if out_dir:
            annotate(image, decision, ranking, latency).save(out_dir / f"{path.stem}_annotated.jpg", quality=92)

    if log:
        log.close()
    print()
    print(f"processed {total} image(s); compartments {counts}")
    if total:
        print(f"stream-level accuracy on this folder: {correct / total * 100:.1f}%")
    return 0


def run_camera(classifier: Classifier, args) -> int:
    import cv2

    capture = cv2.VideoCapture(int(args.camera) if str(args.camera).isdigit() else args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open camera/video: {args.camera}")

    log = DecisionLog(Path(args.log)) if not args.no_log else None
    recent: deque[str] = deque(maxlen=max(3, args.vote_frames))
    show = not args.no_display
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if args.every > 1 and frame_index % args.every:
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            decision, ranking = classifier.classify(image, threshold=args.threshold, margin=args.margin)
            recent.append(decision.stream)
            stable = recent.count(decision.stream) >= max(2, args.vote_frames - 1)
            latency = getattr(decision, "latency_ms", 0.0)

            if stable and not decision.to_residual:
                print(
                    f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {decision.label:<12} "
                    f"{decision.confidence * 100:5.1f}% -> {decision.compartment} "
                    f"({decision.stream}) {latency:.0f} ms"
                )
                if log:
                    log.write(decision, source="camera")

            if show:
                annotated = np.array(annotate(image, decision, ranking, latency))[:, :, ::-1]
                cv2.imshow("BinGenie", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if log:
            log.close()
        if show:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BinGenie inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--folder", type=Path)
    source.add_argument("--camera", default=None, help="camera index or video path")
    source.add_argument("--video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--every", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--vote-frames", type=int, default=3)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--save-annotated", default=None, help="folder for annotated images")
    parser.add_argument("--json", action="store_true", help="emit decisions as JSON lines")
    args = parser.parse_args()

    classifier = Classifier(args.checkpoint, threads=args.threads, tta=args.tta)

    if args.image:
        return run_images(classifier, [args.image], args)
    if args.folder:
        return run_images(classifier, iter_folder(args.folder), args)
    if args.video:
        args.camera = str(args.video)
    return run_camera(classifier, args)


if __name__ == "__main__":
    raise SystemExit(main())
