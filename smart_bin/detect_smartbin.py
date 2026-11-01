"""
Smart Bin — real-time classification with the local YOLOv8 classification model.

FIXED IN THIS REVISION
----------------------
1. **Routing table.** This script used to define BIN_1 ... BIN_7 (seven bins)
   while the product has four colour-coded compartments, so the printed bin never
   matched the machine. It now imports the single routing table from `routing.py`.
2. **Decision logging.** It logged one CSV row per *stable frame*. The report
   itself (§6.7) diagnoses this as the reason the field log is useless. It now
   logs one row per *routed item*: a decision is written when the class is
   confident, stable, and different from the last logged decision within
   `--cooldown` seconds.
3. **Headless safety.** `cv2.imshow` used to abort the process on a headless Pi.
   `--no-display` is now honoured and display errors fall back to headless.
4. **Resource safety.** Camera and CSV handles are released in a `finally:` block.
5. **Camera source.** On a Pi the CSI camera is not `/dev/video0` unless
   `libcamera`'s V4L2 compatibility layer is enabled; the message now says so.

Run:
  python detect_smartbin.py                       # webcam / CSI camera
  python detect_smartbin.py --source clip.mp4
  python detect_smartbin.py --no-display --cooldown 5
  python detect_smartbin.py --weights runs/smart_bin_v1/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from routing import STREAMS, stream_for, unknown_classes  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = BASE_DIR.parent / "logs" / "smartbin_decisions.csv"

# Overlay colour per compartment (BGR)
STREAM_COLORS = {
    "biodegradable": (60, 180, 75),
    "recyclable": (0, 200, 255),
    "electronic": (0, 140, 255),
    "residual": (40, 40, 220),
}


def find_best_weights() -> str:
    candidates = sorted(BASE_DIR.glob("runs/*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        print(f"Found trained model: {candidates[0]}")
        return str(candidates[0])
    print("WARNING: no trained model under smart_bin/runs/*/weights/best.pt.")
    print("         Run train.py first, or use the verified pipeline (train_real.py).")
    print("         Falling back to the ImageNet-pretrained yolov8s-cls (un-tuned).")
    return "yolov8s-cls.pt"


def draw_overlay(frame, cls_name, conf, stream, reason):
    import cv2

    info = STREAMS[stream]
    colour = STREAM_COLORS.get(stream, (200, 200, 200))
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 74), (28, 28, 28), -1)
    cv2.putText(frame, f"{cls_name.upper()}  {conf * 100:.1f}%", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, colour, 2)
    cv2.putText(frame, f"-> {info['bin']} / {info['stream']}   ({reason})", (14, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), colour, 5)


class DecisionWriter:
    """One CSV row per routed item, not per frame."""

    FIELDS = ["timestamp", "class", "confidence", "margin", "bin", "stream", "slot", "reason"]

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self.path = path
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new:
            self._writer.writeheader()

    def write(self, class_name, conf, margin, stream, reason):
        info = STREAMS[stream]
        self._writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "class": class_name,
                "confidence": f"{conf:.4f}",
                "margin": f"{margin:.4f}",
                "bin": info["bin"],
                "stream": stream,
                "slot": info["slot"],
                "reason": reason,
            }
        )
        self._fh.flush()

    def close(self):
        self._fh.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Bin real-time classifier")
    parser.add_argument("--source", default=0, help="0 = camera, or a video path")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=0.60, help="minimum top-1 probability")
    parser.add_argument("--margin", type=float, default=0.12, help="minimum top1-top2 gap")
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=4.0,
                        help="seconds before the same class can be logged again")
    parser.add_argument("--no-display", action="store_true", help="headless (Pi without a screen)")
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.weights or find_best_weights())
    names = model.names
    missing = unknown_classes(list(names.values()) if isinstance(names, dict) else list(names))
    if missing:
        print(f"WARNING: classes with no compartment mapped: {missing} -> they route to residual.")

    recent = deque(maxlen=max(1, args.stable_frames))
    writer = None if args.no_log else DecisionWriter(Path(args.log))
    last_logged = {"class": None, "time": 0.0}

    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"ERROR: cannot open source {args.source!r}.")
        print("  On a Raspberry Pi the CSI camera needs libcamera: enable the")
        print("  'v4l2' compatibility layer (sudo modprobe bcm2835-v4l2) or use picamera2.")
        return 1

    show = not args.no_display
    print(f"\nSmart Bin running (conf {args.conf}, margin {args.margin}, display={show})")
    print("Press Q to quit\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model(frame, verbose=False)[0]
            if result.probs is None:  # not a classification model
                print("ERROR: this model has no classification head. Use a *-cls.pt model.")
                break

            probs = result.probs.data.tolist()
            top1 = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf)
            class_name = names[top1] if isinstance(names, dict) else names[top1]
            ranked = sorted(probs, reverse=True)
            margin = top1_conf - (ranked[1] if len(ranked) > 1 else 0.0)

            confident = top1_conf >= args.conf and margin >= args.margin
            recent.append(class_name if confident else "uncertain")
            votes = Counter(recent)
            stable = confident and votes[class_name] >= max(2, args.stable_frames - 1)
            stream = str(stream_for(class_name)["stream"])

            now = time.time()
            is_new_decision = (
                stable
                and (class_name != last_logged["class"] or now - last_logged["time"] >= args.cooldown)
            )
            if is_new_decision:
                reason = "vision"
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {class_name:<12} {top1_conf * 100:5.1f}% "
                    f"(margin {margin:.2f}) -> {stream_for(class_name)['bin']} / {stream}"
                )
                if writer:
                    writer.write(class_name, top1_conf, margin, stream, reason)
                last_logged = {"class": class_name, "time": now}

            if show:
                try:
                    draw_overlay(frame, class_name, top1_conf, stream,
                                 "confident" if stable else "confirming")
                    cv2.imshow("Smart Bin Classifier", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("No display available - switching to headless mode.")
                    show = False
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer:
            writer.close()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"\nSession log saved -> {args.log}" if writer else "\nNo log written (--no-log).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
