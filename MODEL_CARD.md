# Model card — BinGenie waste classifier

This file is the `model_card.md` that Appendix I of the Phase I report promises
and that the repository did not contain. It follows the standard model-card
structure: intended use, data, metrics, limitations, ethics.

## 1. Model details

| Field | Value |
|---|---|
| Name | `bingenie_mnv3s_128` |
| Architecture | MobileNetV3-Small (torchvision), classifier head replaced for the dataset's class count |
| Parameters | 2.5 M (classifier head replaced) |
| Input | RGB, 128 × 128, ImageNet mean/std normalisation (constants stored in the checkpoint) |
| Output | softmax over the dataset classes; the class is mapped to a physical compartment by `smart_bin/routing.py` |
| Initialisation | **from scratch** (ImageNet weights were not reachable in the build environment; `--pretrained` enables them) |
| Training | AdamW, lr 2e-3 with 3 warm-up epochs + cosine decay, weight decay 0.02, label smoothing 0.1, batch 32, 110 epochs |
| Augmentation | RandomResizedCrop(0.7–1.0), horizontal flip, ±15° rotation, colour jitter, random erasing (p = 0.25) |
| Runtime | CPU (PyTorch); 1.6 ms per image at 128 px (batch 32, 2 threads) |

## 2. Intended use

**In scope**

* Sorting a *single, isolated* item that a user has placed in the bin's imaging cup,
  into one of four compartments: biodegradable, dry recyclable, electronic, residual.
* Operating offline on the device; no image leaves the unit.
* Being overridden: when the model is not confident, the fail-safe policy sends the
  item to the residual compartment rather than guessing (`smart_bin/fusion.py`).

**Out of scope**

* Identifying hazardous waste. A "residual" prediction is not a hazard
  classification, and the model must never be used to decide whether an item is
  safe to handle — this contradicts several classes in the report's 7-class plan
  (medical waste, e-waste) which the shipped model does not contain.
* Multi-item scenes, hand-held items, or objects in motion.
* Any clinical, regulatory or compliance determination.

## 3. Training data

**TrashNet** (`garythung/trashnet`), 2,527 images on a white background:
cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137.
Split used: the **published** file lists — 1,768 train / 328 val / 431 test.

Known properties that matter:

* single-object, studio-like images; no kitchen scene, no clutter, no lighting variation;
* `trash` is under-represented (137 images ~ 5 %);
* the images are not the report's 7 classes; there is no organic, medical or
  electronic-waste class in this dataset;
* TrashNet has documented near-duplicate images between splits; any accuracy on it
  is an upper bound for field performance.

## 4. Metrics

Filled in by `smart_bin/evaluate_real.py` on the held-out test split:

| Metric (held-out test split, 431 images) | Value |
|---|---|
| Top-1 accuracy | 78.89 % (95 % CI 74.79 – 82.48 %) |
| Top-3 accuracy | 94.66 % |
| Macro F1 | 76.35 % |
| Routing accuracy (correct compartment) | 94.90 % (CI 92.39 – 96.61 %) |
| Items refused to residual | 0.93 % |
| Expected calibration error (10 bins) | 0.055 |
| Latency (CPU, 2 threads, batch 32) | 1.8 ms/image |
| Validation accuracy (best epoch 104) | 80.49 % |

Per-class recall: paper 86.1 %, cardboard 81.4 %, plastic 79.7 %, glass 79.3 %,
metal 75.0 %, **trash 51.7 %** (CI 34.4 – 68.6 %). Full detail in
`smart_bin/runs/bingenie_mnv3s_128/metrics_test.json`.

### 4.1 The artefact that ships

| Metric (same 431 images, int8 ONNX, no PyTorch) | Value |
|---|---|
| Top-1 accuracy | 78.65 % (95 % CI 74.54 – 82.26 %) |
| Routing accuracy | 94.90 % |
| Macro F1 | 76.20 % |
| Model size | 1.70 MB |
| Latency, 2 cores, pre-processing included | 30.6 ms/image |
| Agreement with the PyTorch pipeline | 427/431 predictions (99.07 %) |

Produced by `tools/verify_deployment.py`, which imports no PyTorch; the file
`metrics_deployed_test.json` is the source. **Quote these numbers when describing
the demo**, and the §4 numbers when describing the training result — never mix the
two, which is exactly the error the original report made.

Report accuracy **with its confidence interval**. With 431 test images the
interval is roughly ±2 pp, so "95.42 %" is not a meaningful level of precision.

## 5. Limitations

1. From-scratch training on 1,768 images — a transfer-learning run (`--pretrained`)
   is the configuration to quote for comparison with the literature.
2. Domain: white-background studio images. Performance on a kitchen bin with a
   grey plastic cup background, partial occlusion and dim light is unmeasured.
3. Class imbalance: `trash` (91 training images) has the weakest recall, and only
   29 test images support it, so its 51.7 % has a ±17 pp interval.
4. No calibration on device data. The 0.40 threshold in `fusion.py` is a *deliberate*
   operating point, not an unexamined default: sweeping it on the validation split
   (`evaluate_real.py --sweep`, or `tools/eval_onnx.py --sweep` for the int8 file)
   shows routing accuracy is flat from 0.20 to 0.35 and falls as the threshold rises,
   because the model is rarely less than 0.40 confident on this data. Lowering it to
   0.30 recovers ~0.2 pp of routing accuracy on test but refuses almost nothing, so
   0.40 is kept as the fail-safe trade-off. On unknown backgrounds the confidence
   distribution will shift and this threshold must be re-measured.
5. 128 px input: fast, but unsuitable for fine material distinctions (foil vs paper).
6. Single-label output: an item that is genuinely two materials (a laminated cup) is
   forced into one class; the design answer is the confidence threshold + residual
   fallback, not a better label.

## 6. Ethical and privacy notes

* Inference is local; the demo server is local; no image or feature is transmitted.
* The decision log stores class, confidence and compartment only — no images, no
  timestamps tied to a person.
* Failure is asymmetric by design: an unknown item goes to residual, which is
  recoverable, rather than into the recyclable stream, which contaminates a batch.
* Waste composition data ("this household throws away 90 % paper") is commercially
  and personally sensitive; the fleet telemetry described in Chapter 9 aggregates
  per building, never per household.

## 7. How to reproduce

```bash
python smart_bin/dataset.py
python smart_bin/train_real.py --epochs 110 --imgsz 128 --lr 2e-3 --weight-decay 0.02 \
       --scale-min 0.7 --rotation 15 --name bingenie_mnv3s_128
python smart_bin/evaluate_real.py --run smart_bin/runs/bingenie_mnv3s_128
```

The run folder holds `summary.json` (configuration), `history.json` (per-epoch
loss/accuracy), `training_curves.png`, `metrics_test.json` and
`confusion_matrix_test.png`.
