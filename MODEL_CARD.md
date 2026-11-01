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
| Top-1 accuracy | 81.67 % (95 % CI 77.74 – 85.04 %) |
| Top-3 accuracy | 94.90 % |
| Macro F1 | 80.01 % |
| Routing accuracy (correct compartment) | 95.82 % (CI 93.50 – 97.34 %) |
| Items refused to residual | 0.93 % |
| Expected calibration error (10 bins) | 0.040 |
| Latency (CPU, 2 threads, batch 32) | 1.6 ms/image |
| Validation accuracy (best epoch 101) | 80.79 % |

Per-class recall: cardboard 92.9 %, paper 89.8 %, plastic 79.7 %, metal 77.9 %,
glass 74.4 %, **trash 58.6 %**. Full detail in
`smart_bin/runs/bingenie_mnv3s_128/metrics_test.json`.

Report accuracy **with its confidence interval**. With 431 test images the
interval is roughly ±2 pp, so "95.42 %" is not a meaningful level of precision.

## 5. Limitations

1. From-scratch training on 1,768 images — a transfer-learning run (`--pretrained`)
   is the configuration to quote for comparison with the literature.
2. Domain: white-background studio images. Performance on a kitchen bin with a
   grey plastic cup background, partial occlusion and dim light is unmeasured.
3. Class imbalance: `trash` (137 images) has the weakest recall.
4. No calibration on device data; the 0.40 threshold in `fusion.py` is a starting
   value, not a measured optimum. `evaluate_real.py` reports ECE so it can be tuned.
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
