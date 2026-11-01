"""
BinGenie - live demo server for the trained classifier.

A single-file, dependency-light web app (Python standard library + PyTorch):

    python app.py                 # http://0.0.0.0:8000
    python app.py --port 8080 --threshold 0.45

Endpoints
    GET  /                     demo page
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

from dataset import TRASHNET_DIR
from fusion import SensorReading, decide
from model import STREAMS

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
LOG_PATH = BASE_DIR.parent / "logs" / "bingenie_decisions.csv"

CLASSIFIER = None  # set in main()
THRESHOLD = 0.40
SAMPLES: list[dict] = []

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BinGenie - waste segregation demo</title>
<style>
  :root {
    --green:#22c55e; --yellow:#eab308; --orange:#f97316; --red:#ef4444;
    --bg:#0b1210; --panel:#131c19; --line:#22312c; --text:#e6f2ec; --muted:#8fa8a0;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:radial-gradient(1200px 700px at 20% -10%, #123 0%, var(--bg) 55%);
         color:var(--text); font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif; }
  header { padding:26px 28px 10px; display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { margin:0; font-size:26px; letter-spacing:.4px; }
  h1 span { color:var(--green); }
  .tag { color:var(--muted); font-size:13px; }
  main { padding:8px 28px 48px; display:grid; grid-template-columns: minmax(320px, 1.1fr) minmax(320px, 1fr);
         gap:22px; align-items:start; max-width:1180px; }
  @media (max-width: 880px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; }
  .card h2 { margin:0 0 12px; font-size:15px; text-transform:uppercase; letter-spacing:1.2px; color:var(--muted); }
  .drop { border:2px dashed #2c443b; border-radius:14px; padding:26px; text-align:center; cursor:pointer;
          transition:.15s; background:#0f1715; }
  .drop.hover { border-color:var(--green); background:#10201a; }
  .drop img { max-width:100%; max-height:320px; border-radius:10px; display:none; margin-top:6px; }
  .drop p { margin:0; color:var(--muted); font-size:14px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
  button, .btn { background:#17342a; color:var(--text); border:1px solid #2b4a3e; border-radius:10px;
          padding:9px 14px; font-size:13px; cursor:pointer; }
  button:hover, .btn:hover { border-color:var(--green); }
  button:disabled { opacity:.45; cursor:default; }
  .verdict { margin-top:16px; border-radius:14px; padding:16px; border:1px solid var(--line); background:#0f1715; }
  .verdict .cls { font-size:30px; font-weight:700; letter-spacing:.5px; }
  .verdict .bin { font-size:16px; margin-top:4px; }
  .pill { display:inline-block; padding:3px 9px; border-radius:999px; font-size:12px; margin-right:6px;
          background:#1b2a24; color:var(--muted); }
  .bars { margin-top:16px; display:grid; gap:7px; }
  .bar { display:grid; grid-template-columns:88px 1fr 52px; gap:9px; align-items:center; font-size:12.5px; }
  .bar .track { height:9px; background:#1a2a24; border-radius:99px; overflow:hidden; }
  .bar .fill { height:100%; background:linear-gradient(90deg,#16a34a,#4ade80); border-radius:99px; }
  .bar.dim .fill { background:#2f4a40; }
  .muted { color:var(--muted); font-size:12.5px; }
  .gallery { display:grid; grid-template-columns:repeat(auto-fill, minmax(74px,1fr)); gap:8px; margin-top:10px; }
  .gallery img { width:100%; height:70px; object-fit:cover; border-radius:9px; cursor:pointer;
                 border:2px solid transparent; }
  .gallery img:hover { border-color:var(--green); }
  .gallery figure { margin:0; position:relative; }
  .gallery figcaption { font-size:10px; color:var(--muted); text-align:center; margin-top:2px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }
  th, td { text-align:left; padding:5px 6px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; }
  code { background:#0f1715; padding:1px 5px; border-radius:5px; font-size:12px; }
  a { color:#7ee2a8; }
</style>
</head>
<body>
<header>
  <h1>Bin<span>Genie</span></h1>
  <div class="tag">on-device waste classification &rarr; four-stream routing</div>
</header>
<main>
  <section class="card">
    <h2>1 &middot; Present an item</h2>
    <div class="drop" id="drop">
      <p><strong>Drop an image here</strong> or click to upload</p>
      <p class="muted">jpeg / png &middot; processed locally, nothing leaves this machine</p>
      <img id="preview" alt="preview">
    </div>
    <input type="file" id="file" accept="image/*" hidden>
    <div class="row">
      <button id="run" disabled>Classify &amp; route</button>
      <button id="randomSample">Use a test-set image</button>
      <label class="pill" style="padding:8px 10px;">threshold
        <input id="thr" type="range" min="20" max="90" value="40" style="vertical-align:middle;width:90px">
        <span id="thrVal">0.40</span>
      </label>
    </div>
    <div class="muted" id="sampleHint" style="margin-top:8px;"></div>
  </section>

  <section class="card">
    <h2>2 &middot; Decision</h2>
    <div class="verdict" id="verdict">
      <div class="cls" style="color:var(--muted)">-</div>
      <div class="bin muted">waiting for an item</div>
    </div>
    <div class="bars" id="bars"></div>
    <div class="muted" id="meta" style="margin-top:12px;"></div>
  </section>

  <section class="card" style="grid-column:1/-1">
    <h2>3 &middot; Held-out test samples</h2>
    <div class="muted">Images from <code>datasets/trashnet/test</code> - never seen during training.</div>
    <div class="gallery" id="gallery"></div>
  </section>

  <section class="card" style="grid-column:1/-1">
    <h2>4 &middot; Model card</h2>
    <div id="info" class="muted">loading...</div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let current = null;

function streamColor(stream) {
  return {biodegradable:'var(--green)', recyclable:'var(--yellow)',
          electronic:'var(--orange)', residual:'var(--red)'}[stream] || 'var(--muted)';
}

$('thr').addEventListener('input', (e) => {
  const v = (e.target.value / 100).toFixed(2);
  $('thrVal').textContent = v;
});

$('drop').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => loadFile(e.target.files[0]));
['dragover','dragenter'].forEach(ev => $('drop').addEventListener(ev, (e) => {
  e.preventDefault(); $('drop').classList.add('hover'); }));
['dragleave','drop'].forEach(ev => $('drop').addEventListener(ev, (e) => {
  e.preventDefault(); $('drop').classList.remove('hover'); }));
$('drop').addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0]; if (f) loadFile(f);
});

function loadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    current = reader.result;
    $('preview').src = current;
    $('preview').style.display = 'block';
    $('run').disabled = false;
    $('sampleHint').textContent = '';
  };
  reader.readAsDataURL(file);
}

$('run').addEventListener('click', async () => {
  if (!current) return;
  $('run').disabled = true;
  $('meta').textContent = 'classifying...';
  try {
    const response = await fetch('api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image: current, threshold: Number($('thr').value) / 100})
    });
    if (!response.ok) throw new Error(await response.text());
    render(await response.json());
  } catch (err) {
    $('meta').textContent = 'error: ' + err.message;
  } finally {
    $('run').disabled = false;
  }
});

function render(data) {
  const d = data.decision;
  const colour = streamColor(d.stream);
  $('verdict').style.borderColor = colour;
  $('verdict').innerHTML = `
    <div class="cls" style="color:${colour}">${d.label.toUpperCase()}</div>
    <div class="bin" style="color:${colour}">&#8594; ${d.compartment} compartment &middot; ${d.stream}</div>
    <div style="margin-top:8px">
      <span class="pill">${(d.confidence * 100).toFixed(1)}% confident</span>
      <span class="pill">margin ${(d.margin * 100).toFixed(1)}%</span>
      <span class="pill">${d.used_fusion ? 'sensor fusion used' : 'vision only'}</span>
      <span class="pill">${data.latency_ms.toFixed(0)} ms</span>
    </div>
    <div class="muted" style="margin-top:8px">${d.reasons.join(' &middot; ')}</div>`;
  $('bars').innerHTML = data.ranking.map(([name, p], i) => `
    <div class="bar ${i ? 'dim' : ''}">
      <span>${name}</span>
      <span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="muted">${(p * 100).toFixed(1)}%</span>
    </div>`).join('');
  $('meta').textContent = `runtime ${data.runtime} · checkpoint ${data.checkpoint}`;
}

$('randomSample').addEventListener('click', () => {
  if (!SAMPLES.length) return;
  const pick = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  current = pick.url;
  $('preview').src = pick.url;
  $('preview').style.display = 'block';
  $('run').disabled = false;
  $('sampleHint').innerHTML = `loaded a <strong>${pick.label}</strong> image from the test split`;
});

async function boot() {
  const info = await (await fetch('api/info')).json();
  $('info').innerHTML = `
    <div>checkpoint <code>${info.checkpoint}</code> · ${info.model} · ${info.imgsz}px · classes ${info.labels.join(', ')}</div>
    <table>
      <tr><th>metric (held-out test split)</th><th>value</th></tr>
      <tr><td>top-1 accuracy</td><td><strong>${info.metrics.top1_accuracy_pct ?? 'n/a'}</strong></td></tr>
      <tr><td>top-3 accuracy</td><td>${info.metrics.top3_accuracy_pct ?? 'n/a'}</td></tr>
      <tr><td>macro F1</td><td>${info.metrics.macro_f1_pct ?? 'n/a'}</td></tr>
      <tr><td>routing accuracy (correct compartment)</td><td>${info.metrics.routing_accuracy_pct ?? 'n/a'}</td></tr>
      <tr><td>items sent to residual</td><td>${info.metrics.residual_rate_pct ?? 'n/a'}</td></tr>
      <tr><td>images evaluated</td><td>${info.metrics.images ?? 'n/a'}</td></tr>
    </table>
    <div style="margin-top:8px">${info.note}</div>`;
  SAMPLES = (await (await fetch('api/samples')).json()).samples;
  const byClass = {};
  SAMPLES.forEach(s => (byClass[s.label] = byClass[s.label] || []).push(s));
  $('gallery').innerHTML = SAMPLES.map(s => `
    <figure title="${s.label}">
      <img src="${s.url}" data-url="${s.url}" data-label="${s.label}" alt="${s.label}">
      <figcaption>${s.label}</figcaption>
    </figure>`).join('');
  document.querySelectorAll('.gallery img').forEach(img => img.addEventListener('click', () => {
    current = img.dataset.url;
    $('preview').src = current;
    $('preview').style.display = 'block';
    $('run').disabled = false;
    $('sampleHint').innerHTML = `loaded a <strong>${img.dataset.label}</strong> image from the test split`;
  }));
}
boot();
</script>
</body>
</html>
"""


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
            self._json(
                {
                    "decision": decision.as_dict(),
                    "ranking": [[name, round(prob, 4)] for name, prob in ranking[:6]],
                    "latency_ms": round(getattr(decision, "latency_ms", 0.0), 1),
                    "runtime": "CPU, PyTorch",
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
        payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

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
        checkpoint = CLASSIFIER.checkpoint
        metrics_path = checkpoint.parent / "metrics_test.json"
        metrics = {}
        if metrics_path.exists():
            raw = json.loads(metrics_path.read_text())
            metrics = {
                "top1_accuracy_pct": f"{raw['top1_accuracy'] * 100:.2f} % "
                                     f"(95 % CI {raw['top1_ci95'][0] * 100:.2f}-{raw['top1_ci95'][1] * 100:.2f})",
                "top3_accuracy_pct": f"{raw['top3_accuracy'] * 100:.2f} %",
                "macro_f1_pct": f"{raw['macro_f1'] * 100:.2f} %",
                "routing_accuracy_pct": f"{raw['routing_accuracy_streams'] * 100:.2f} %",
                "residual_rate_pct": f"{raw['residual_rate'] * 100:.2f} %",
                "images": raw["images"],
            }
        return {
            "checkpoint": str(checkpoint.relative_to(BASE_DIR.parent)),
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
    global CLASSIFIER, THRESHOLD, SAMPLES
    parser = argparse.ArgumentParser(description="BinGenie demo server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    from predict import Classifier  # local import keeps --help fast

    THRESHOLD = args.threshold
    CLASSIFIER = Classifier(args.checkpoint, threads=args.threads, tta=args.tta)
    SAMPLES = collect_samples()

    print("=" * 62)
    print("  BinGenie demo")
    print("=" * 62)
    print(f"  checkpoint : {CLASSIFIER.checkpoint}")
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
