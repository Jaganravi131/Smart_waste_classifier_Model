# BinGenie — Waste Segregation Smart Bin (ML side)

Classification and four-stream routing for the BinGenie smart bin
(20MUPJ701, Phase I, Sri Sairam Engineering College).

**Read [`ERROR_REPORT.md`](ERROR_REPORT.md) first.** It is the full technical review of
the Phase I report and of this repository: what is measured, what is claimed, what
was fabricated, and what has been fixed. The short version:

> The project's architecture is sound. The evidence was missing: no trained model
> shipped, the training figures in the report were generated with `numpy.random`,
> and the code printed in the appendix never ran. This repository now contains a
> pipeline that trains, evaluates and serves a real classifier, with every number
> produced by an actual run.

---

## Quick start

On a fresh checkout — or in any sandbox that has been wiped — one command does
everything (virtualenv, dataset, checkpoint if missing, verification):

```bash
python tools/bootstrap.py --serve        # … then open http://localhost:8000
python tools/bootstrap.py --dry-run      # just say what is missing
python tools/bootstrap.py --quick        # 30-epoch model if you are in a hurry
python tools/bootstrap.py --full         # force the training stack even when only serving
```

When the committed ONNX export is present, `--serve` installs **only**
numpy + pillow + onnxruntime (48 MB of wheels, 190 MB installed). Measured on a
bare clone with no venv, no dataset and no weights except the committed ones:
**9.3 s** from `python3 tools/bootstrap.py --serve` to the first HTTP response,
against 45 min of training plus a 5.7 GB install before this change. The 5.7 GB
training stack is installed only when a model actually has to be produced.

`bootstrap.py` is safe to re-run: every stage is skipped when it is already
satisfied. The dataset (git-ignored), the virtualenv and the trained weights
(`*.pt`, too large for plain git) do not survive a machine reset, so this is the
supported way back.

Or step by step:

```bash
pip install -r requirements.txt

# 1. dataset (TrashNet, 2,527 images, 6 classes, published train/val/test split)
python smart_bin/dataset.py

# 2. train (≈ 25 s/epoch on 2 CPU cores, real curves written to runs/<name>/)
python smart_bin/train_real.py --epochs 110 --imgsz 128

# 3. evaluate on the held-out test split
python smart_bin/evaluate_real.py

# 4. use it
python smart_bin/predict.py --image item.jpg          # one image -> one decision
python smart_bin/predict.py --folder some/folder      # a whole folder
python smart_bin/app.py                               # browser demo on :8000
```

On a machine with internet, add `--pretrained` to step 2 to start from ImageNet
weights: the same dataset then reaches the ~90 % range the literature reports,
instead of the from-scratch baseline.

## What the pipeline guarantees

* **One class list, one routing table.** `smart_bin/routing.py` maps every class to
  one of the four physical compartments (green / yellow / orange / red). Training,
  evaluation, the CLI and the web demo all import it. The previous code carried three
  different, contradictory tables.
* **One pre-processing path.** `smart_bin/model.py` owns the transforms; the
  checkpoint stores the image size, mean and std, so training and deployment cannot
  drift apart.
* **A real held-out split.** `dataset.py` rebuilds TrashNet's *published*
  train/val/test lists (1,768 / 328 / 431), so the test images are never seen
  during training.
* **Metrics with uncertainty.** `evaluate_real.py` reports top-1, top-3, macro-F1,
  per-class precision/recall/F1, Wilson 95 % confidence intervals, expected
  calibration error and a *routing* accuracy (did the item reach the right
  compartment), plus a confusion matrix.
* **Fail-safe fusion you can argue with.** `fusion.py` lets a sensor override the
  camera only when the sensor actually carries evidence, and records the reason for
  every decision. It ships with self-tests (`python smart_bin/fusion.py`).
* **Decisions, not frames, in the log.** One CSV row per routed item.

## Repository layout

| Path | What it is |
|---|---|
| `smart_bin/routing.py` | class → compartment map (stdlib only) |
| `smart_bin/model.py` | model factory, transforms, checkpoint I/O |
| `smart_bin/dataset.py` | dataset download + reproducible split |
| `smart_bin/train_real.py` | training loop, real curves, history JSON |
| `smart_bin/evaluate_real.py` | held-out evaluation, confidence intervals, `--sweep` for the routing threshold |
| `smart_bin/fusion.py` | sensor fusion, fail-safe routing, self-tests |
| `smart_bin/quality.py` | frame-quality guard: refuses frames too dark or too flat to trust |
| `smart_bin/onnx_export.py` | ONNX export + float-versus-deployed verification |
| `smart_bin/backends.py` | ONNX (light) and PyTorch inference backends |
| `smart_bin/predict.py` | image / folder / video / camera inference CLI |
| `smart_bin/app.py` | demo server (standard-library HTTP server, JSON API) |
| `tools/compare_runs.py` | paired run comparison: Wilson CIs, exact McNemar, per-class recall |
| `tools/eval_onnx.py` | scores the deployed int8 ONNX artefact and sweeps its threshold |
| `tools/arch_sweep.py` | tunes each model's routing threshold on val, then scores test |
| `tools/plot_arch_compare.py` | validation curves of several runs, by epoch and by CPU-minute |
| `tools/robustness_sweep.py` | accuracy under simulated dim light, blur, noise, occlusion, tilt |
| `smart_bin/ui.py` | the demo page: markup, CSS and vanilla JS, no build step |
| `smart_bin/train.py`, `detect_smartbin.py`, `eval_per_class.py`, `analyze_runtime_bias.py` | legacy YOLOv8-cls path, kept working, bugs fixed |
| `tools/bootstrap.py` | one-command rebuild: env → dataset → weights → ONNX → demo |
| `tools/verify_deployment.py` | scores the shipped ONNX artefact without PyTorch |
| `tools/make_demo_grid.py` | the real prediction grid in `docs/demo_predictions.jpg` |
| `tools/verify_report_numbers.py` | 42 arithmetic checks (20 pass against the submitted draft) |
| `archive/simulated_plots/` | the scripts that fabricated the report's figures (archived, not deleted) |
| `REPORT_REWRITE.md` | drop-in corrected text for every section the report gets wrong |
| `logs/` | decision logs (CSV), git-ignored |
| `smart_bin/runs/` | metrics/plots per run; the one shipped model's `best.pt` and `.onnx` are committed |

## Data

**TrashNet** — 2,527 images in 6 classes (cardboard 403, glass 501, metal 410,
paper 594, plastic 482, trash 137), photographed on a white background.
It is *not* the dataset the report describes (7 classes including organic, medical
and e-waste) and it is *not* field-representative: it has no kitchen backgrounds, no
partial occlusion and no lighting variation. Everything the report says about
low-light and occlusion performance must be re-measured on a real field set.
`smart_bin/data.yaml` corresponds to this dataset for the YOLO path;
`prepare_dataset.py` targets the 10-class Kaggle garbage set.

## Honest status

| Item | Status |
|---|---|
| Trained classifier | yes — MobileNetV3-Small, 128 px, 110 epochs from scratch (80.49 % validation); **weights committed** under `smart_bin/runs/bingenie_mnv3s_128/` |
| Held-out evaluation | yes — 431 test images: **78.89 % top-1, 94.90 % routing** (`metrics_test.json`); the shipped int8 ONNX export scores **78.65 % / 94.90 %** (`metrics_deployed_test.json`) |
| Browser demo | yes — `python smart_bin/app.py` |
| On-device export | **yes** — ONNX (fp32) + optional dynamic int8, with a float-vs-deployed agreement report (`onnx_export.py`) |
| Moisture / inductive sensor fusion on hardware | logic implemented and tested; no sensor data exists yet |
| Carousel control | **not in this repository** — the report's `carousel_control.py` was never committed |
| 7-class organic / medical / e-waste model | **not yet** — requires a dataset that contains those classes |
