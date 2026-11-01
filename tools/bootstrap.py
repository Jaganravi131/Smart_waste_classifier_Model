#!/usr/bin/env python3
"""
bootstrap.py - one command to get a working BinGenie demo back.

The sandbox this project is developed in is wiped between sessions: the virtual
environment (excluded from snapshots), the dataset (git-ignored) and the trained
checkpoint (git-ignored, too large for plain git) all disappear, while the source
and the committed metrics survive.  That is exactly what happened once, and the
demo had to be rebuilt by hand.  This script does it in one step.

    python tools/bootstrap.py              # env + dataset + checkpoint (if missing)
    python tools/bootstrap.py --serve      # ... then serve the demo on :8000
    python tools/bootstrap.py --retrain    # force a fresh training run
    python tools/bootstrap.py --quick      # 30-epoch run (fewer minutes, lower accuracy)

Stages, each skipped when already satisfied:
  1. virtualenv in <repo>/.venv with the packages from requirements.txt
  2. TrashNet downloaded and split into data/{train,val,test}
  3. a checkpoint in smart_bin/runs/*/best.pt (trained if absent)
  4. optionally, the demo server
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SMART_BIN = REPO / "smart_bin"
VENV = REPO / ".venv"
VENV_PY = VENV / "bin" / "python"
RUNS = SMART_BIN / "runs"

DOCUMENTED_RUN = [
    "--epochs", "110", "--imgsz", "128", "--batch", "32", "--threads", "2",
    "--lr", "2e-3", "--weight-decay", "0.02", "--scale-min", "0.7",
    "--rotation", "15", "--name", "bingenie_mnv3s_128",
]
QUICK_RUN = [
    "--epochs", "30", "--imgsz", "128", "--batch", "32", "--threads", "2",
    "--lr", "2e-3", "--weight-decay", "0.02", "--scale-min", "0.7",
    "--rotation", "15", "--name", "bingenie_quick",
]


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(part) for part in command))
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


# ---------------------------------------------------------------------------
# 0. re-exec inside the virtualenv
# ---------------------------------------------------------------------------
RUNTIME_PACKAGES = ["numpy", "pillow", "onnxruntime"]
RUNTIME_PROBE = "import numpy, PIL, onnxruntime"
FULL_PROBE = "import torch, torchvision, PIL, numpy, matplotlib"


def shipped_onnx() -> Path | None:
    """The committed ONNX export, if this checkout has one."""
    candidates = sorted(RUNS.glob("*/*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    preferred = [p for p in candidates if p.stem.endswith("_int8")] or candidates
    return preferred[0] if preferred else None


def wants_runtime_only(args: argparse.Namespace) -> bool:
    """Serving an already-exported model needs no training stack at all."""
    if args.full or args.retrain:
        return False
    return bool(args.serve or args.dry_run) and shipped_onnx() is not None


def ensure_venv_and_reexec(args: argparse.Namespace) -> None:
    """Create/refresh .venv, then restart this script with the venv interpreter."""
    if Path(sys.prefix).resolve() == VENV.resolve():
        return  # already inside

    if args.skip_install and VENV_PY.exists():
        log("re-entering the existing virtualenv")
        os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])

    probe_needed = RUNTIME_PROBE if wants_runtime_only(args) else FULL_PROBE
    if args.dry_run and subprocess.run([str(VENV_PY), "-c", probe_needed],
                                       capture_output=True).returncode != 0:
        # A dry run is read-only: never download a gigabyte just to print a plan.
        log(f"would create {VENV} and install "
            f"{'numpy + pillow + onnxruntime' if wants_runtime_only(args) else 'the training stack'}")
        return

    if not VENV_PY.exists():
        log(f"creating virtualenv at {VENV}")
        run([sys.executable, "-m", "venv", str(VENV)])

    runtime_only = wants_runtime_only(args)
    probe = RUNTIME_PROBE if runtime_only else FULL_PROBE
    if subprocess.run([str(VENV_PY), "-c", probe], capture_output=True).returncode == 0:
        log("dependencies already installed")
    else:
        run([str(VENV_PY), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
        if runtime_only:
            log("installing the inference runtime only (numpy + pillow + onnxruntime, 48 MB of wheels)")
            run([str(VENV_PY), "-m", "pip", "install", "--quiet", *RUNTIME_PACKAGES])
        else:
            log("installing the full stack (torch + torchvision, a few minutes and ~1 GB)")
            run([str(VENV_PY), "-m", "pip", "install", "--quiet", "-r", str(REPO / "requirements.txt")])

    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])


# ---------------------------------------------------------------------------
# 1. dataset
# ---------------------------------------------------------------------------
def ensure_dataset() -> Path:
    sys.path.insert(0, str(SMART_BIN))
    from dataset import TRASHNET_DIR, dataset_summary, ensure_dataset as fetch

    summary = dataset_summary(TRASHNET_DIR)
    if all(split in summary for split in ("train", "val", "test")):
        log(f"dataset present: {summary.get('train_total', '?')} training images")
        return TRASHNET_DIR
    log("dataset missing - downloading TrashNet and rebuilding the published split")
    fetch(TRASHNET_DIR)
    return TRASHNET_DIR


# ---------------------------------------------------------------------------
# 2. checkpoint
# ---------------------------------------------------------------------------
def find_checkpoint() -> Path | None:
    candidates = sorted(RUNS.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def train(quick: bool) -> Path:
    arguments = QUICK_RUN if quick else DOCUMENTED_RUN
    log(f"no checkpoint found - training ({'quick 30-epoch' if quick else 'documented 110-epoch'} run)")
    log("expect roughly 11 min (quick) or 40 min (documented) on 2 CPU cores")
    run([sys.executable, str(SMART_BIN / "train_real.py"), *arguments], cwd=str(REPO))
    checkpoint = find_checkpoint()
    if checkpoint is None:
        raise SystemExit("[bootstrap] training finished but no best.pt was produced")
    return checkpoint


# ---------------------------------------------------------------------------
# 3. verification + demo
# ---------------------------------------------------------------------------
def verify(checkpoint: Path) -> None:
    log(f"checkpoint: {checkpoint.relative_to(REPO)}")
    metrics = checkpoint.parent / "metrics_test.json"
    if metrics.exists():
        import json

        data = json.loads(metrics.read_text())
        log(
            f"committed metrics: top-1 {data['top1_accuracy'] * 100:.2f} %, "
            f"routing {data['routing_accuracy_streams'] * 100:.2f} % on {data['images']} test images"
        )
    else:
        log("no metrics_test.json for this run - run evaluate_real.py to create it")

    log("running the smoke test")
    result = subprocess.run([sys.executable, str(REPO / "tools" / "smoke_test.py"), "--skip-server"],
                            capture_output=True, text=True)
    tail = [line for line in result.stdout.splitlines() if line.startswith(("[PASS]", "[FAIL]", "13", "1", "0"))]
    for line in tail[-4:]:
        print("   " + line)
    if result.returncode != 0:
        log("smoke test reported failures - see tools/smoke_test.py output above")


def export(checkpoint: Path) -> Path:
    """Export the checkpoint to ONNX so later runs need no training stack."""
    log("exporting to onnx (fp32 + int8) and verifying float-vs-deployed agreement")
    run([sys.executable, str(SMART_BIN / "onnx_export.py"),
         "--checkpoint", str(checkpoint), "--int8",
         "--report", str(checkpoint.parent / "onnx_verify.json")], cwd=str(REPO))
    exported = shipped_onnx()
    if exported is None:
        raise SystemExit("[bootstrap] export finished but no .onnx was written")
    return exported


def serve(port: int, checkpoint: Path | None, onnx_model: Path | None) -> None:
    """Serve the exported ONNX model when there is one, else the checkpoint."""
    command = [sys.executable, "-u", str(SMART_BIN / "app.py"), "--port", str(port)]
    if onnx_model is not None:
        command += ["--backend", "onnx", "--onnx", str(onnx_model)]
        log(f"serving the demo on http://0.0.0.0:{port}  (onnx {onnx_model.name})")
    else:
        command += ["--checkpoint", str(checkpoint)]
        log(f"serving the demo on http://0.0.0.0:{port}  (checkpoint {checkpoint.parent.name})")
    subprocess.run(command, cwd=str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the BinGenie runtime and demo")
    parser.add_argument("--serve", action="store_true", help="start the demo server when ready")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--retrain", action="store_true", help="train even if a checkpoint exists")
    parser.add_argument("--quick", action="store_true", help="use the shorter 30-epoch recipe")
    parser.add_argument("--skip-install", action="store_true", help="reuse .venv without checking packages")
    parser.add_argument("--no-verify", action="store_true", help="skip the smoke test")
    parser.add_argument("--full", action="store_true",
                        help="install the training stack even when only serving")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is missing and exit without training or serving")
    args = parser.parse_args()

    ensure_venv_and_reexec(args)  # does not return when outside the venv

    if not SMART_BIN.exists():
        raise SystemExit(f"[bootstrap] cannot find {SMART_BIN}")

    checkpoint = find_checkpoint()
    onnx_model = shipped_onnx()
    needs_data = args.retrain or (checkpoint is None and onnx_model is None)
    if needs_data:
        ensure_dataset()
    else:
        log("dataset not required for serving (pass --retrain to rebuild it)")

    if args.dry_run:
        log(f"virtualenv : {VENV} (ready)")
        log(f"dataset    : {'ready' if (SMART_BIN / 'datasets' / 'trashnet' / 'train').exists() else 'not downloaded'}")
        log(f"onnx model : {onnx_model.relative_to(REPO) if onnx_model else 'none (repo has no export)'}")
        log(f"checkpoint : {checkpoint.relative_to(REPO) if checkpoint else 'none (would train)'}")
        log("dry run: nothing trained, nothing served")
        return 0

    if checkpoint is None:
        checkpoint = train(args.quick)
    else:
        log(f"checkpoint present: {checkpoint.relative_to(REPO)}")

    if onnx_model is None:
        onnx_model = export(checkpoint)
    else:
        log(f"onnx present: {onnx_model.relative_to(REPO)}")

    if not args.no_verify:
        verify(checkpoint)

    if args.serve:
        serve(args.port, checkpoint, onnx_model)
    else:
        log("done. start the demo with:  python smart_bin/app.py   (or: --serve)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
