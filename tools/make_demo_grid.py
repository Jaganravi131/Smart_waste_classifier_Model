#!/usr/bin/env python3
"""
make_demo_grid.py - the figure the report calls "sample predictions".

The original figure in the report was fabricated by `archive/simulated_plots/`.
This script replaces it with a real prediction grid: two held-out test images per
class, run through the same backend the demo serves, annotated with the true label
and the model's verdict. It prints how many of the twelve were right so the caption
can be checked instead of trusted.

    python tools/make_demo_grid.py                  # docs/demo_predictions.jpg
    python tools/make_demo_grid.py --per-class 3 --out docs/grid.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "smart_bin"))

from backends import find_artifacts, make_backend  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from routing import stream_for  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "smart_bin" / "datasets" / "trashnet" / "test"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a real prediction grid")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "demo_predictions.jpg")
    parser.add_argument("--cell", type=int, default=190)
    parser.add_argument("--backend", default="auto", choices=("auto", "onnx", "torch"))
    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"[grid] no test split at {DATA} - run smart_bin/dataset.py first")
    backend = make_backend(args.backend)

    classes = sorted(d.name for d in DATA.iterdir() if d.is_dir())
    picks = []
    for name in classes:
        images = sorted((DATA / name).glob("*.jpg"))[: args.per_class]
        picks.extend((name, path) for path in images)

    cell, label_h, header = args.cell, 38, 40
    columns = len(classes)
    rows = (len(picks) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell, header + rows * (cell + label_h) + 8), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:  # pragma: no cover
        font = bold = ImageFont.load_default()

    draw.text((12, 11), f"real predictions - {backend.describe()} - "
                        f"held-out test images, two per class", font=bold, fill=(240, 240, 240))

    correct = 0
    for index, (truth, path) in enumerate(picks):
        image = Image.open(path).convert("RGB")
        probs = backend.probabilities(image)
        top = int(probs.argmax())
        predicted = backend.labels[top]
        correct += predicted == truth
        tile = image.copy()
        tile.thumbnail((cell, cell))
        column, row = index % columns, index // columns
        box_top = header + row * (cell + label_h)
        x = column * cell + (cell - tile.width) // 2
        y = box_top + (cell - tile.height) // 2
        canvas.paste(tile, (x, y))
        ok = predicted == truth
        colour = (110, 220, 130) if ok else (240, 120, 120)
        stream = stream_for(truth)["stream"]
        same_stream = stream_for(predicted)["stream"] == stream
        caption = f"{truth} -> {predicted} {probs[top] * 100:.0f}%"
        if not ok and same_stream:
            caption += " (same bin)"
        caption_y = box_top + cell
        draw.rectangle([column * cell, caption_y, (column + 1) * cell - 1, caption_y + label_h - 1],
                       fill=(32, 32, 32))
        draw.text((column * cell + 8, caption_y + 10), caption, font=font, fill=colour)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=92)
    print(f"[grid] {correct}/{len(picks)} correct, written to {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
