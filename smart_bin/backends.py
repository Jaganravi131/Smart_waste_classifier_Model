"""
BinGenie - two interchangeable inference backends.

    OnnxBackend   onnxruntime + numpy + pillow         (48 MB of wheels, 190 MB installed)
    TorchBackend  PyTorch + torchvision                (~1 GB, training stack)

Why both
--------
The demo, the CLI and the device only need *inference*. Requiring PyTorch for
that costs a 1 GB install and several seconds of import time before the first
prediction, which is why a cold environment used to mean a 40-minute rebuild
before the preview could come back. `OnnxBackend` removes that: if a `.onnx`
export is present, the app starts in ~1 s with no training stack at all.

Both backends expose the same tiny interface, so nothing downstream changes:

    backend.labels        list[str]
    backend.imgsz         int
    backend.describe()    str, e.g. "onnx int8 128px"
    backend.probabilities(image)  -> np.ndarray (len(labels),), sums to 1
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


_WEIGHT_CACHE: dict = {}


def _resize_weights(in_size: int, out_size: int) -> np.ndarray:
    """Weights of torch.nn.functional.interpolate(mode="bilinear", antialias=True).

    PIL's BILINEAR filter is *not* the same kernel: on a 512 px image resized to
    146 px the two differ by ~4 % per pixel, which flipped 10 of 431 predictions
    in an earlier build.  Reproducing PyTorch's filter exactly (a triangle filter
    whose support widens with the downscale factor) costs one matrix multiply per
    axis and makes the deployed pipeline numerically identical to the measured one.
    """
    key = (in_size, out_size)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    scale = in_size / out_size
    support = scale
    weights = np.zeros((out_size, in_size), dtype=np.float32)
    for j in range(out_size):
        center = (j + 0.5) * scale
        lo = max(0, int(np.floor(center - support)))
        hi = min(in_size, int(np.ceil(center + support)))
        index = np.arange(lo, hi)
        w = np.clip(1.0 - np.abs((index + 0.5 - center) / scale), 0.0, None)
        total = float(w.sum())
        if total <= 0.0:  # pragma: no cover - degenerate size
            w = np.ones_like(w)
            total = float(w.sum())
        weights[j, index] = w / total
    _WEIGHT_CACHE[key] = weights
    return weights


def _resize_antialias(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Separable bilinear downsample that matches torchvision to ~1e-5."""
    height, width = array.shape[:2]
    if height != out_h:
        rows = _resize_weights(height, out_h) @ array.reshape(height, -1)
        array = rows.reshape(out_h, width, array.shape[2])
    if width != out_w:
        columns = _resize_weights(width, out_w) @ np.ascontiguousarray(array.transpose(1, 0, 2)).reshape(width, -1)
        array = np.ascontiguousarray(columns.reshape(out_w, out_h, array.shape[2]).transpose(1, 0, 2))
    return array


def _resize_crop(image: Image.Image, imgsz: int, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """torchvision's eval_transform, reimplemented with PIL + NumPy only."""
    resize_to = int(round(imgsz / 0.875))
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        size = (resize_to, max(resize_to, int(height * resize_to / width)))
    else:
        size = (max(resize_to, int(width * resize_to / height)), resize_to)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = _resize_antialias(array, size[1], size[0])
    left = (array.shape[1] - imgsz) // 2
    top = (array.shape[0] - imgsz) // 2
    array = array[top:top + imgsz, left:left + imgsz]
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


class OnnxBackend:
    """Inference through onnxruntime - no PyTorch, no torchvision."""

    kind = "onnx"

    def __init__(self, onnx_path: Path, threads: int = 2):
        import json

        import onnxruntime as ort

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        meta_path = onnx_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing sidecar {meta_path.name} next to {onnx_path.name} - "
                "re-export with `python smart_bin/onnx_export.py`."
            )
        meta = json.loads(meta_path.read_text())
        self.path = onnx_path
        self.meta = meta
        self.labels: List[str] = list(meta["labels"])
        self.imgsz: int = int(meta["imgsz"])
        self.mean = meta.get("mean", DEFAULT_MEAN)
        self.std = meta.get("std", DEFAULT_STD)

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def describe(self) -> str:
        quantised = "int8" if self.path.stem.endswith("_int8") else "fp32"
        return f"onnx {quantised} {self.imgsz}px"

    def probabilities(self, image: Image.Image) -> np.ndarray:
        batch = _resize_crop(image, self.imgsz, self.mean, self.std)
        logits = self.session.run(None, {self.input_name: batch})[0]
        return _softmax(np.asarray(logits, dtype=np.float64))[0]


class TorchBackend:
    """Inference through PyTorch (needed for training/evaluation anyway)."""

    kind = "torch"

    def __init__(self, checkpoint_path: Path, threads: int = 2):
        import torch

        from model import load_checkpoint

        torch.set_num_threads(max(1, threads))
        self.path = checkpoint_path
        self.model, self.labels, self.imgsz = load_checkpoint(checkpoint_path)
        from model import eval_transform

        self.transform = eval_transform(self.imgsz)
        self._torch = torch
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def describe(self) -> str:
        return f"torch {self.imgsz}px"

    def probabilities(self, image: Image.Image) -> np.ndarray:
        import torch

        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = self.model(tensor).softmax(1)[0]
        return probs.numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def find_artifacts(prefer: str = "auto") -> tuple[Optional[Path], Optional[Path]]:
    """Return (checkpoint, onnx) for the newest run that has them."""
    checkpoints = sorted(RUNS_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    onnx_files = sorted(
        [p for p in RUNS_DIR.glob("*/*.onnx") if not p.stem.endswith("_fp32")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # the int8 export is what ships: 1.7 MB instead of 6.1 MB, same accuracy
    quantised = [p for p in onnx_files if p.stem.endswith("_int8")]
    checkpoint = checkpoints[0] if checkpoints else None
    onnx_path = (quantised or onnx_files)[0] if onnx_files else None
    if onnx_path is None and checkpoint is not None:
        candidate = checkpoint.with_suffix(".onnx")
        onnx_path = candidate if candidate.exists() else None

    if prefer == "onnx":
        return checkpoint, onnx_path
    if prefer == "torch":
        return checkpoint, None
    # auto: prefer the light backend when both exist
    return checkpoint, onnx_path


def make_backend(prefer: str = "auto", checkpoint: Optional[Path] = None,
                 onnx_path: Optional[Path] = None, threads: int = 2):
    """Build whichever backend is available, preferring ONNX unless asked otherwise."""
    if checkpoint is None and onnx_path is None:
        checkpoint, onnx_path = find_artifacts(prefer)

    order = ["onnx", "torch"] if prefer in ("auto", "onnx") else ["torch", "onnx"]
    errors = []
    for kind in order:
        try:
            if kind == "onnx" and onnx_path is not None:
                return OnnxBackend(onnx_path, threads=threads)
            if kind == "onnx" and checkpoint is not None:
                candidate = checkpoint.with_suffix(".onnx")
                if candidate.exists():
                    return OnnxBackend(candidate, threads=threads)
            if kind == "torch" and checkpoint is not None:
                return TorchBackend(checkpoint, threads=threads)
        except ImportError as exc:
            errors.append(f"{kind}: {exc}")
        except FileNotFoundError as exc:
            errors.append(f"{kind}: {exc}")

    detail = "; ".join(errors) if errors else "no checkpoint and no .onnx found under smart_bin/runs"
    raise SystemExit(
        "No usable inference backend.\n"
        f"  {detail}\n"
        "  Fix: run `python tools/bootstrap.py` (downloads nothing, trains only if needed)."
    )
