#!/usr/bin/env python3
"""
End-to-end smoke test for the BinGenie pipeline.

Checks, in order:
  1. the routing table covers every class of the trained checkpoint,
  6. the fusion layer passes its own self-test,
  3. the dataset split exists and the test partition is non-empty,
  4. the checkpoint loads and reproduces a pre-processing config,
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

    # 3. dataset ----------------------------------------------------------
    from dataset import TRASHNET_DIR, dataset_summary

    summary = dataset_summary(TRASHNET_DIR) if TRASHNET_DIR.exists() else {}
    has_splits = all(split in summary for split in ("train", "val", "test"))
    report("dataset splits present", has_splits, str({k: v for k, v in summary.items() if k in ("train", "val", "test")}))
    if not has_splits:
        print("\nRun `python smart_bin/dataset.py` first.")
        return 1

    # 4. checkpoint -------------------------------------------------------
    from model import load_checkpoint, eval_transform

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = sorted((SMART_BIN / "runs").glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            report("checkpoint found", False, "no smart_bin/runs/*/best.pt")
            return 1
        checkpoint = candidates[0]
    model, labels, imgsz = load_checkpoint(checkpoint)
    report("checkpoint loads", True, f"{checkpoint.parent.name} ({imgsz}px, {len(labels)} classes)")
    unmapped = unknown_classes(labels)
    report("every checkpoint class has a compartment", not unmapped, f"unmapped: {unmapped}" if unmapped else "")

    # 5. inference --------------------------------------------------------
    from PIL import Image
    import torch

    test_image = next((TRASHNET_DIR / "test").rglob("*.jpg"))
    tensor = eval_transform(imgsz)(Image.open(test_image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = model(tensor).softmax(1).squeeze(0).numpy()
    report("probabilities sum to 1", abs(float(probs.sum()) - 1.0) < 1e-4, f"sum={probs.sum():.6f}")
    report("output length matches labels", len(probs) == len(labels))

    from fusion import SensorReading, decide

    decision = decide(labels, probs, sensor=SensorReading(), threshold=0.40)
    valid = decision.stream in STREAMS and 0.0 <= decision.confidence <= 1.0
    report(
        "decision is valid",
        bool(valid),
        f"{decision.label} -> {decision.compartment} ({decision.stream}) "
        f"{decision.confidence:.2f}",
    )

    # 6. HTTP demo --------------------------------------------------------
    if not args.skip_server:
        import app as demo

        from predict import Classifier

        demo.THRESHOLD = 0.40
        demo.CLASSIFIER = Classifier(checkpoint)
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
