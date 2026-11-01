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
SKIPPED: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def skip(name: str, detail: str = "") -> None:
    """A check that cannot run here but is not a defect.

    A checkout that serves an exported model deliberately has no dataset - that
    is the point of the ONNX path - so a missing dataset must not fail the suite
    that is supposed to prove the deployment works.
    """
    SKIPPED.append(name)
    print(f"[SKIP] {name}{(' - ' + detail) if detail else ''}")


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
    if has_splits:
        report("dataset splits present",
               True, str({k: v for k, v in summary.items() if k in ("train", "val", "test")}))
    else:
        skip("dataset splits present", "not downloaded (fine for serving an exported model)")

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

            # The page hands the server a sample URL instead of shipping the
            # bytes. That path used to 400 (the URL went to base64.b64decode),
            # which made every gallery click fail - so it is a check now.
            if samples:
                request = urllib.request.Request(
                    f"{base}/api/predict",
                    data=json.dumps({"image": samples[0]["url"], "threshold": 0.4}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    sample_result = json.load(urllib.request.urlopen(request, timeout=30))
                    report("POST /api/predict with a sample URL", "decision" in sample_result,
                           sample_result["decision"]["compartment"])
                except urllib.error.HTTPError as exc:
                    report("POST /api/predict with a sample URL", False, f"HTTP {exc.code}")

                request = urllib.request.Request(
                    f"{base}/api/predict",
                    data=json.dumps({"image": "api/sample/../../../etc/passwd"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    urllib.request.urlopen(request, timeout=30)
                    report("sample path traversal refused", False, "request was accepted")
                except urllib.error.HTTPError as exc:
                    report("sample path traversal refused", exc.code == 400, f"HTTP {exc.code}")

            page = urllib.request.urlopen(f"{base}/", timeout=20).read().decode()
            report("demo page served", "BinGenie" in page and 'id="result"' in page, f"{len(page)} bytes")
        except Exception as exc:  # noqa: BLE001
            report("HTTP demo", False, f"{exc.__class__.__name__}: {exc}")
        finally:
            server.shutdown()
            server.server_close()

    # -- frame-quality guard -------------------------------------------------
    try:
        from PIL import Image as _Image

        from quality import assess_quality

        normal = _Image.new("RGB", (96, 96), (210, 210, 210))
        for x in range(0, 96, 6):
            for y in range(96):
                normal.putpixel((x, y), (30, 30, 30))
        report("quality guard passes a well-lit frame", assess_quality(normal).ok,
               "luminance 0.7")

        dark = _Image.new("RGB", (96, 96), (10, 10, 10))
        verdict = assess_quality(dark)
        report("quality guard refuses a black frame", not verdict.ok and bool(verdict.reasons),
               verdict.reasons[0] if verdict.reasons else "no reason given")
    except Exception as exc:  # noqa: BLE001
        report("quality guard", False, f"{exc.__class__.__name__}: {exc}")

    try:
        from PIL import ImageEnhance

        from predict import Classifier

        classifier = Classifier(args.checkpoint, backend=args.backend, onnx=args.onnx)
        frames = sorted((REPO / "smart_bin" / "datasets" / "trashnet" / "test").glob("*/*.jpg"))
        if not frames:
            raise SystemExit("no test images")
        with _Image.open(frames[0]) as handle:
            frame = handle.convert("RGB")
        clear_decision, _ = classifier.classify(frame, threshold=0.40, guard=True)
        dark_decision, _ = classifier.classify(ImageEnhance.Brightness(frame).enhance(0.25),
                                               threshold=0.40, guard=True)
        report("guard converts a dark frame into a refusal",
               dark_decision.to_residual and clear_decision.quality is not None,
               f"clean refused={clear_decision.to_residual}, dark refused={dark_decision.to_residual}")
    except Exception as exc:  # noqa: BLE001
        report("guard end-to-end", False, f"{exc.__class__.__name__}: {exc}")

    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed"
          + (f", {len(SKIPPED)} skipped" if SKIPPED else ""))
    if FAILED:
        print("failed checks: " + ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
