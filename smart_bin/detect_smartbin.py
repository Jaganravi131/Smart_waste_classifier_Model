"""
Smart Bin — Real-time Classification with local YOLOv8 model.

This uses YOLOv8 CLASSIFICATION (whole-frame, no bounding boxes).
It classifies each webcam frame and tells which bin to open.

Run:
  python detect_smartbin.py                          # webcam (default)
  python detect_smartbin.py --source video.mp4       # video file
  python detect_smartbin.py --conf 0.70 --margin 0.15
"""

import argparse
import csv
import os
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

# Which physical bin each class maps to
BIN_MAPPING = {
    "battery":    "BIN_1 (Hazardous / E-waste)",
    "biological": "BIN_2 (Wet / Organic)",
    "cardboard":  "BIN_3 (Dry Recyclable)",
    "glass":      "BIN_4 (Glass)",
    "metal":      "BIN_5 (Metal)",
    "paper":      "BIN_3 (Dry Recyclable)",
    "plastic":    "BIN_6 (Plastic)",
    "shoes":      "BIN_7 (General)",
    "clothes":    "BIN_7 (General)",
    "trash":      "BIN_7 (General)",
}

# Overlay colour per class (BGR)
CLASS_COLORS = {
    "battery":    (0,   0,   255),   # red
    "biological": (0,   180,  60),   # green
    "cardboard":  (30,  144, 255),   # dodger blue
    "glass":      (255, 255,   0),   # cyan
    "metal":      (160, 160, 160),   # grey
    "paper":      (0,   255, 255),   # yellow
    "plastic":    (0,   165, 255),   # orange
    "shoes":      (180,  90,  40),   # brown
    "clothes":    (200,   0, 200),   # magenta
    "trash":      (80,   80,  80),   # dark grey
}

# Raise threshold for classes that had more confusion during validation.
CLASS_MIN_CONF = {
    "battery": 0.65,
    "biological": 0.65,
    "cardboard": 0.70,
    "clothes": 0.70,
    "glass": 0.72,
    "metal": 0.72,
    "paper": 0.70,
    "plastic": 0.78,
    "shoes": 0.70,
    "trash": 0.75,
}


def find_best_weights():
    candidates = sorted(
        BASE_DIR.glob("runs/*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"Found trained model: {candidates[0]}")
        return str(candidates[0])
    print("WARNING: No trained model found. Run train.py first.")
    print("         Falling back to pretrained yolov8s-cls (untuned).")
    return "yolov8s-cls.pt"


def draw_overlay(frame, cls_name, conf, bin_label, color):
    h, w = frame.shape[:2]
    # Filled banner at top
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(frame, f"{cls_name.upper()}  {conf*100:.1f}%",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"→ {bin_label}",
                (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    # Coloured border around frame
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)


def main():
    parser = argparse.ArgumentParser(description="Smart Bin real-time classifier")
    parser.add_argument("--source",  default=0,      help="0=webcam, or video path")
    parser.add_argument("--weights", default=None,   help="path to best.pt (auto-detected)")
    parser.add_argument("--conf",    type=float, default=0.60,
                        help="global minimum confidence floor")
    parser.add_argument("--margin",  type=float, default=0.12,
                        help="minimum gap between top1 and top2 probabilities")
    parser.add_argument("--stable-frames", type=int, default=3,
                        help="frames used for short voting to stabilize predictions")
    parser.add_argument("--log",     default=str(BASE_DIR.parent / "logs" / "smartbin_log.csv"),
                        help="CSV log path")
    args = parser.parse_args()

    weights = args.weights or find_best_weights()
    model   = YOLO(weights)
    recent_preds = deque(maxlen=max(1, args.stable_frames))

    # CSV logging
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log_new = not os.path.exists(args.log)
    log_fh  = open(args.log, "a", newline="")
    writer  = csv.writer(log_fh)
    if log_new:
        writer.writerow(["timestamp", "class", "confidence", "top2_conf", "margin", "bin"])

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {args.source}")
        return

    print(f"\nSmart Bin running  (global conf: {args.conf}, margin: {args.margin})")
    print("Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results  = model(frame, verbose=False)
        ts       = datetime.now(timezone.utc).isoformat()
        probs    = results[0].probs
        scores   = probs.data.tolist()

        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        cls_name = model.names[top1_idx]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top2_conf = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = top1_conf - top2_conf

        class_floor = CLASS_MIN_CONF.get(cls_name, args.conf)
        needed_conf = max(args.conf, class_floor)
        confident = (top1_conf >= needed_conf) and (margin >= args.margin)

        recent_preds.append(cls_name if confident else "uncertain")
        vote_count = Counter(recent_preds)
        needed_votes = max(2, args.stable_frames - 1)
        stable = confident and vote_count[cls_name] >= needed_votes

        bin_label = BIN_MAPPING.get(cls_name, "BIN_7 (General)")
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))

        if stable:
            print(
                f"[{ts}]  {cls_name:<12}  {top1_conf*100:.1f}%  "
                f"(top2 {top2_conf*100:.1f}%, m={margin:.2f})  → {bin_label}"
            )
            writer.writerow([ts, cls_name, f"{top1_conf:.4f}", f"{top2_conf:.4f}", f"{margin:.4f}", bin_label])
            draw_overlay(frame, cls_name, top1_conf, bin_label, color)
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
            cv2.putText(
                frame,
                f"? {cls_name} {top1_conf*100:.1f}% | top2 {top2_conf*100:.1f}% | m {margin:.2f}",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (120, 120, 120),
                1,
            )

        cv2.imshow("Smart Bin Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    log_fh.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession log saved → {args.log}")


if __name__ == "__main__":
    main()
