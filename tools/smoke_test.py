#!/usr/bin/env python3
"""
End-to-end smoke test for the BinGenie pipeline.

Checks, in order:
  1. the routing table covers every class of the trained checkpoint,
  6. the fusion layer passes its own self-test,
  3. the dataset split exists and the test partition is non-empty,
  4. an inference backend is available (ONNX export preferred, PyTorch fallback),
  5. a real test image produces a valid decision + probability simplex,
  6. the HTTP demo answers /api/info, /api/samples and /api/predict.

Run:
    python tools/smoke_test.py
    python tools/smoke_test.py --port 8099        # also boot the demo server
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SMART_BIN = REPO / "smart_bin"
sys.path.insert(0, str(SMART_BIN))

PASSED: list[str] = []
FAILED: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--backend", choices=["auto", "onnx", "torch"], default="auto")
    args = parser.parse_args()

    # 1. routing table ----------------------------------------------------
    from routing import STREAMS, stream_for, unknown_classes

    report("four compartments defined", len(STREAMS) == 4, f"{sorted(STREAMS)}")
    report(
        "metal routes to recyclable (not e-waste)",
        stream_for("metal")["stream"] == "recyclable",
    )
    report("unknown class falls back to residual", stream_for("quantum-foam")["stream"] == "residual")

    # 2. fusion self-test -------------------------------------------------
    fusion = subprocess.run([sys.executable, str(SMART_BIN / "fusion.py")], capture_output=True, text=True)
    report("fusion self-test", fusion.returncode == 0, fusion.stdout.strip().splitlines()[-1] if fusion.stdout else "")

    # 3. dataset (only needed when a model has to be trained from it) -------
    from dataset import TRASHNET_DIR, dataset_summary

    summary = dataset_summary(TRASHNET_DIR) if TRASHNET_DIR.exists() else {}
    has_splits = all(split in summary for split in ("train", "val", "test"))
    report("dataset splits present", has_splits,
           str({k: v for k, v in summary.items() if k in ("train", "val", "test")})
           if has_splits else "not downloaded (fine for serving an exported model)")

    # 4. model backend ----------------------------------------------------
    # Prefers the exported ONNX model, so this test validates the artefact that
    # actually deploys - and runs on a machine with no PyTorch at all.
    from backends import make_backend
    from PIL import Image

    try:
        backend = make_backend(args.backend, checkpoint=args.checkpoint, onnx_path=args.onnx, threads=2)
    except SystemExit as exc:
        report("model backend available", False, str(exc).replace("\n", " ")[:120])
        print("\nFix: python tools/bootstrap.py")
        return 1
    labels = backend.labels
    imgsz = backend.imgsz
    report("model backend available", True, backend.describe())
    unmapped = unknown_classes(labels)
    report("every model class has a compartment", not unmapped, f"unmapped: {unmapped}" if unmapped else "")

    # 5. inference --------------------------------------------------------
    if has_splits:
        test_image = next((TRASHNET_DIR / "test").rglob("*.jpg"))
    else:
        from app import collect_samples  # type: ignore

        test_image = None
    if test_image is None or not Path(test_image).exists():
        # fall back to an image shipped with the repo
        candidates = sorted(Path(REPO / "docs").glob("*.jpg"))
        if not candidates:
            report("a test image is available", False, "no dataset and no docs/*.jpg")
            return 1
        test_image = candidates[0]

    probs = np.asarray(backend.probabilities(Image.open(test_image)))
    report("probabilities sum to 1", abs(float(probs.sum()) - 1.0) < 1e-4, f"sum={probs.sum():.6f}")
    report("output length matches labels", len(probs) == len(labels), f"{len(probs)} outputs")

    from fusion import SensorReading, decide

    decision = decide(labels, probs, sensor=SensorReading(), threshold=0.40)
    valid = decision.stream in STREAMS and 0.0 <= decision.confidence <= 1.0
    report(
        "decision is valid",
        bool(valid),
        f"{decision.label} -> {decision.compartment} ({decision.stream}) {decision.confidence:.2f}",
    )

    # 6. HTTP demo --------------------------------------------------------
    if not args.skip_server:
        import app as demo

        from predict import Classifier

        demo.THRESHOLD = 0.40
        demo.CLASSIFIER = Classifier(args.checkpoint, backend=args.backend, onnx=args.onnx)
        demo.SAMPLES = demo.collect_samples()

        from http.server import ThreadingHTTPServer

        server = ThreadingHTTPServer(("127.0.0.1", args.port), demo.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.5)
        base = f"http://127.0.0.1:{args.port}"
        try:
            info = json.load(urllib.request.urlopen(f"{base}/api/info", timeout=20))
            report("GET /api/info", "labels" in info, ", ".join(info.get("labels", [])))
            samples = json.load(urllib.request.urlopen(f"{base}/api/samples", timeout=20))["samples"]
            report("GET /api/samples", len(samples) > 0, f"{len(samples)} images")
            payload = {
                "image": "data:image/jpeg;base64," + base64.b64encode(test_image.read_bytes()).decode(),
                "threshold": 0.4,
            }
            request = urllib.request.Request(
                f"{base}/api/predict",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            result = json.load(urllib.request.urlopen(request, timeout=30))
            report("POST /api/predict", "decision" in result, result["decision"]["compartment"])
        except Exception as exc:  # noqa: BLE001
            report("HTTP demo", False, f"{exc.__class__.__name__}: {exc}")
        finally:
            server.shutdown()
            server.server_close()

    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed checks: " + ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
