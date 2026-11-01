"""
BinGenie - export a trained checkpoint to ONNX, and verify that the exported
model agrees with the PyTorch one.

Why this file exists
--------------------
Two problems it solves:

1. **Deployment.** The report claims a quantised on-device model (TensorFlow Lite,
   int8) and never shows one. An ONNX export is the same idea with a runtime that
   is available everywhere - on the Pi, `onnxruntime` is a 10 MB install and does
   not need PyTorch (which is ~1 GB and 20x slower to start). Exporting also lets
   the demo run without the training stack.

2. **The missing float-vs-deployed check.** The report asserts "0.9 percentage
   points" of quantisation loss without measuring it. This script measures it on
   the held-out test split: top-1 agreement, the accuracy of the exported model,
   and the largest probability difference per image.

Usage
-----
    python onnx_export.py                       # newest checkpoint -> fp32 ONNX
    python onnx_export.py --int8                # + dynamic int8 quantisation
    python onnx_export.py --checkpoint path/to/best.pt --out path/to/model.onnx
    python onnx_export.py --verify-only --onnx path/to/model.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


# ---------------------------------------------------------------------------
# preprocessing - the *same* function the runtime uses, so the comparison below
# measures the network, not two different resamplers
# ---------------------------------------------------------------------------
def preprocess(image: Image.Image, imgsz: int, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """`backends._resize_crop`, i.e. exactly what OnnxBackend feeds the model."""
    from backends import _resize_crop

    return _resize_crop(image, imgsz, mean, std)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export(checkpoint_path: Path, out_path: Path, opset: int = 17) -> Path:
    import torch
    from model import load_checkpoint

    model, labels, imgsz = load_checkpoint(checkpoint_path)
    model.eval()

    dummy = torch.randn(1, 3, imgsz, imgsz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )
    # torch >= 2.6 defaults to the dynamo exporter, which needs the extra
    # `onnxscript` package.  The TorchScript exporter needs nothing beyond torch,
    # so try it first and fall back only if this torch build has dropped it.
    for label, extra in (("torchscript", {"dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}}}),
                         ("dynamo", {"dynamo": True, "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}}})):
        try:
            torch.onnx.export(model, dummy, str(out_path), dynamo=(label == "dynamo"), **kwargs, **extra)
            print(f"[export] exporter: {label}")
            break
        except Exception as exc:  # noqa: BLE001 - report and try the other path
            print(f"[export] {label} exporter unavailable ({type(exc).__name__}: {exc})")
    else:
        raise SystemExit("[export] both ONNX exporters failed; install onnxscript and retry")

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "labels": labels,
                "imgsz": imgsz,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "source_checkpoint": checkpoint_path.name,
                "opset": opset,
            },
            indent=2,
        )
    )
    print(f"[export] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[export] wrote {sidecar}")
    return out_path


def quantize_int8(onnx_path: Path) -> Path:
    """Dynamic int8 quantisation - no calibration data needed for weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    sidecar = onnx_path.with_suffix(".json")
    if sidecar.exists():
        out_path.with_suffix(".json").write_text(sidecar.read_text())
    print(f"[export] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, int8 dynamic)")
    return out_path


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def verify(onnx_path: Path, checkpoint_path: Path | None, data_dir: Path, limit: int | None = None) -> dict:
    import onnxruntime as ort
    from torchvision.datasets import ImageFolder

    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    labels, imgsz = meta["labels"], meta["imgsz"]
    mean, std = meta["mean"], meta["std"]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_dir = data_dir / "test"
    dataset = ImageFolder(test_dir)
    if dataset.classes != labels:
        raise SystemExit(f"dataset classes {dataset.classes} != model labels {labels}")

    paths = [Path(p) for p, _ in dataset.samples]
    targets = np.array([t for _, t in dataset.samples])
    if limit:
        paths, targets = paths[:limit], targets[:limit]

    onnx_probs = np.vstack([
        softmax(session.run(None, {"input": preprocess(Image.open(p), imgsz, mean, std)})[0])
        for p in paths
    ])
    onnx_pred = onnx_probs.argmax(1)
    report = {
        "onnx": str(onnx_path),
        "images": len(paths),
        "onnx_top1": round(float((onnx_pred == targets).mean()), 4),
        "onnx_mean_confidence": round(float(onnx_probs.max(1).mean()), 4),
    }

    if checkpoint_path and checkpoint_path.exists():
        import torch
        from model import eval_transform, load_checkpoint

        model, torch_labels, torch_imgsz = load_checkpoint(checkpoint_path)
        assert torch_labels == labels and torch_imgsz == imgsz
        transform = eval_transform(imgsz)
        with torch.no_grad():
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])
            torch_probs = model(batch).softmax(1).numpy()
        torch_pred = torch_probs.argmax(1)
        delta = np.abs(torch_probs - onnx_probs)
        report.update(
            {
                "torch_top1": round(float((torch_pred == targets).mean()), 4),
                "top1_agreement": round(float((torch_pred == onnx_pred).mean()), 4),
                "max_probability_delta": round(float(delta.max()), 5),
                "mean_probability_delta": round(float(delta.mean()), 6),
                "flipped_predictions": int((torch_pred != onnx_pred).sum()),
            }
        )

    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and verify the ONNX model")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "datasets" / "trashnet")
    parser.add_argument("--int8", action="store_true", help="also write a dynamically quantised int8 model")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="verify on the first N test images only")
    parser.add_argument("--report", type=Path, default=None, help="write the verification JSON here")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = sorted(RUNS_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit("No checkpoint found - train first (train_real.py) or use --verify-only.")
        checkpoint = candidates[0]

    if args.verify_only:
        onnx_path = args.onnx or checkpoint.with_suffix(".onnx")
        report = verify(onnx_path, checkpoint, args.data, args.limit)
    else:
        out_path = args.out or checkpoint.with_suffix(".onnx")
        export(checkpoint, out_path)
        report = verify(out_path, checkpoint, args.data, args.limit)
        if args.int8:
            int8_path = quantize_int8(out_path)
            int8_report = verify(int8_path, checkpoint, args.data, args.limit)
            report["int8"] = int8_report

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"[export] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
