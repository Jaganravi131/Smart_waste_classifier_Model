"""
BinGenie - live demo server for the trained classifier.

A dependency-light web app: Python standard library for the server, ONNX Runtime
or PyTorch for inference, and one static page (`ui.py`) with no build step.

    python app.py                 # http://0.0.0.0:8000
    python app.py --port 8080 --threshold 0.45

Endpoints
    GET  /                     demo page (from ui.py)
    GET  /favicon.svg          icon
    GET  /api/info             model + measured metrics
    GET  /api/samples          sample images from the held-out test split
    GET  /api/sample/<class>/<file>  image bytes
    POST /api/predict          {"image": "data:image/jpeg;base64,..."}
    GET  /api/decisions        the decision log as JSON (one row per item)

Nothing is uploaded anywhere: inference happens in this process, and the
decision log stays on disk.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

from dataset import TRASHNET_DIR
from fusion import SensorReading, decide
from model import STREAMS
from predict import DecisionLog
from ui import PAGE

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
LOG_PATH = BASE_DIR.parent / "logs" / "bingenie_decisions.csv"

CLASSIFIER = None  # set in main()
DECISION_LOG = None  # set in main(); the same CSV the CLI writes
LOG_LOCK = threading.Lock()
THRESHOLD = 0.40
SAMPLES: list[dict] = []

FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<rect width="24" height="24" rx="6" fill="#0f2a1d"/>'
    '<path d="M5 8h14l-1.2 11.2a2 2 0 0 1-2 1.8H8.2a2 2 0 0 1-2-1.8Z" fill="none" '
    'stroke="#34d399" stroke-width="1.6" stroke-linejoin="round"/>'
    '<path d="M9 8V5.8A1.8 1.8 0 0 1 10.8 4h2.4A1.8 1.8 0 0 1 15 5.8V8" fill="none" '
    'stroke="#34d399" stroke-width="1.6" stroke-linecap="round"/>'
    "</svg>"
)


class Handler(BaseHTTPRequestHandler):
    server_version = "BinGenie/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/predict" in (args[0] if args else ""):
            return

    # -- helpers ---------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj).encode("utf-8"))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/favicon.svg":
            self._send(200, FAVICON.encode("utf-8"), "image/svg+xml")
        elif path == "/api/info":
            self._json(self._info())
        elif path == "/api/samples":
            self._json({"samples": SAMPLES[:36]})
        elif path.startswith("/api/sample/"):
            self._serve_sample(unquote(path[len("/api/sample/"):]))
        elif path == "/api/decisions":
            self._json(self._decisions())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/predict":
            return self._json({"error": "not found"}, 404)
        try:
            payload = self._read_json()
            image = self._decode(payload.get("image", ""))
            threshold = float(payload.get("threshold", THRESHOLD))
            decision, ranking = CLASSIFIER.classify(
                image, threshold=threshold, margin=0.0, sensor=SensorReading()
            )
            # One row per classified item, in the same CSV the CLI appends to -
            # the log on the page is the real log, not a browser-side list.
            if DECISION_LOG is not None:
                with LOG_LOCK:
                    DECISION_LOG.write(decision, source="web", image=str(payload.get("name", ""))[:120])
            self._json(
                {
                    "decision": decision.as_dict(),
                    "ranking": [[name, round(prob, 4)] for name, prob in ranking[:6]],
                    "latency_ms": round(getattr(decision, "latency_ms", 0.0), 1),
                    "runtime": CLASSIFIER.runtime_label,
                    "checkpoint": CLASSIFIER.checkpoint.name,
                }
            )
        except Exception as exc:  # noqa: BLE001 - demo server
            self._json({"error": f"{exc.__class__.__name__}: {exc}"}, 400)

    # -- implementations --------------------------------------------------
    @staticmethod
    def _decode(data_url: str) -> Image.Image:
        if not data_url:
            raise ValueError("no image supplied")
        if data_url.startswith("data:"):
            payload = data_url.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        # The page may point at one of its own held-out samples instead of
        # shipping the bytes. Resolve it through the same guard as /api/sample,
        # so a path can never escape the dataset directory. (Clicking a sample
        # used to fail with a 400: the URL was handed to base64.b64decode.)
        relative = data_url.split("api/sample/", 1)[-1].lstrip("/")
        root = TRASHNET_DIR.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("image path outside the dataset") from exc
        if not target.is_file():
            raise ValueError(f"no such sample image: {relative}")
        return Image.open(target).convert("RGB")

    def _serve_sample(self, rel: str):
        target = (TRASHNET_DIR / rel).resolve()
        try:
            target.relative_to(TRASHNET_DIR.resolve())
        except ValueError:
            return self._json({"error": "invalid path"}, 400)
        if not target.is_file():
            return self._json({"error": "missing sample"}, 404)
        self._send(200, target.read_bytes(), "image/jpeg")

    @staticmethod
    def _decisions():
        if not LOG_PATH.exists():
            return {"rows": []}
        import csv

        with open(LOG_PATH, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return {"rows": rows[-50:]}

    @staticmethod
    def _info():
        artifact = CLASSIFIER.checkpoint
        # When the shipped artefact has its own measurement, show *that* one: the
        # demo runs the ONNX export, so quoting the PyTorch numbers here would be
        # the same "which model are you describing?" mistake as the report's five
        # different accuracies.
        deployed = artifact.parent / "metrics_deployed_test.json"
        metrics_path = deployed if (deployed.exists() and artifact.suffix == ".onnx") else artifact.parent / "metrics_test.json"
        metrics = {}
        measured_on = "PyTorch checkpoint"
        if metrics_path.exists():
            raw = json.loads(metrics_path.read_text())
            if metrics_path == deployed:
                measured_on = f"the shipped artefact ({raw.get('backend', 'onnx')})"
            per_class = [
                {
                    "label": name,
                    "recall": values["recall"],
                    "precision": values["precision"],
                    "f1": values["f1"],
                    "support": values["support"],
                }
                for name, values in (raw.get("per_class") or {}).items()
            ]
            metrics = {
                "top1_accuracy_pct": f"{raw['top1_accuracy'] * 100:.2f} %",
                "top1_ci_text": f"95 % CI {raw['top1_ci95'][0] * 100:.2f}-{raw['top1_ci95'][1] * 100:.2f}",
                "top3_accuracy_pct": f"{raw['top3_accuracy'] * 100:.2f} %",
                "macro_f1_pct": f"{raw['macro_f1'] * 100:.2f} %",
                "routing_accuracy_pct": f"{raw['routing_accuracy_streams'] * 100:.2f} %",
                "residual_rate_pct": f"{raw['residual_rate'] * 100:.2f} %",
                "images": raw["images"],
                "measured_on": measured_on,
                "per_class": per_class,
            }
        try:
            artifact_label = str(artifact.relative_to(BASE_DIR.parent))
        except ValueError:
            artifact_label = str(artifact)
        return {
            "checkpoint": artifact_label,
            "backend": CLASSIFIER.describe(),
            "backend_kind": getattr(CLASSIFIER.backend, "kind", "torch"),
            "model": "MobileNetV3-Small (trained here, no cloud)",
            "imgsz": CLASSIFIER.imgsz,
            "labels": CLASSIFIER.labels,
            "metrics": metrics,
            "note": (
                "Trained on TrashNet (2,527 images, 6 classes) with the official split. "
                "Only camera evidence is used in this demo; the moisture/inductive rules "
                "in fusion.py are exercised by predict.py on the device."
            ),
        }


def collect_samples(limit_per_class: int = 6) -> list[dict]:
    test_dir = TRASHNET_DIR / "test"
    samples: list[dict] = []
    if not test_dir.exists():
        return samples
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        images = sorted(class_dir.glob("*.jpg"))
        random.Random(7).shuffle(images)
        for path in images[:limit_per_class]:
            samples.append(
                {
                    "label": class_dir.name,
                    "url": f"api/sample/test/{class_dir.name}/{path.name}",
                }
            )
    return samples


def main() -> int:
    global CLASSIFIER, THRESHOLD, SAMPLES, DECISION_LOG
    parser = argparse.ArgumentParser(description="BinGenie demo server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint", type=Path, default=None, help="best.pt (PyTorch) to serve")
    parser.add_argument("--onnx", type=Path, default=None, help="explicit ONNX model + .json sidecar")
    parser.add_argument("--backend", choices=["auto", "onnx", "torch"], default="auto",
                        help="auto prefers ONNX (small dependency footprint), then PyTorch")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    from predict import Classifier  # local import keeps --help fast

    THRESHOLD = args.threshold
    CLASSIFIER = Classifier(args.checkpoint, threads=args.threads, tta=args.tta,
                            backend=args.backend, onnx=args.onnx)
    SAMPLES = collect_samples()
    DECISION_LOG = DecisionLog(LOG_PATH)

    print("=" * 62)
    print("  BinGenie demo")
    print("=" * 62)
    print(f"  backend    : {CLASSIFIER.describe()}")
    print(f"  classes    : {CLASSIFIER.labels}")
    print(f"  samples    : {len(SAMPLES)} test images")
    print(f"  listening  : http://{args.host}:{args.port}")
    print("=" * 62)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
