# Report rewrite — drop-in replacement text

This file contains the corrected text for every part of the Phase I report that
`ERROR_REPORT.md` flags as wrong, unsupported or unverifiable. It is written so it
can be pasted into the document section by section.

**Notation used below**

| Marker | Meaning |
|---|---|
| `⟨MEASURE: …⟩` | a number only your bench can produce — fill it in, do not invent it |
| `⟨DELETE⟩` | remove the sentence/table/figure entirely |
| plain text | ready to paste |

Every measured number quoted here comes from
`smart_bin/runs/bingenie_mnv3s_128/metrics_test.json` and is reproducible with the
commands in §10.

---

## 0. Global find-and-replace (do this first)

| Find | Replace with | Why |
|---|---|---|
| Solid Waste Management Rules, 2026 | **Solid Waste Management Rules, 2016** | the 2026 rules do not exist; every page references them |
| six-class classifier | **seven-class classifier** (or "six-class" only where you genuinely mean six) | §3.5 and App. III say six, Table 5.4 lists seven |
| 95.42 % (as "routing accuracy") | **94.90 % routing / 78.89 % top-1** (measured) or keep 95.42 % *only* as top-1 with its CI | the report swaps the two definitions between the Abstract and §7.3 |
| threshold 0.50 (in §6.7) | **0.40** | the routing threshold is 0.40 everywhere else |
| "EfficientNet-B3 / TensorFlow Lite" | the architecture and runtime you actually ship (see §3 below) | the repository never contained either |
| author-year citations | **[n] numeric citations** | no reference is currently linked to any sentence |

---

## 1. Abstract — replacement

> Rapid urbanisation and the rising volume of municipal solid waste have made
> segregation at source the critical bottleneck of waste management in India. The
> country generates approximately 1,70,000 tonnes of municipal solid waste every
> day, and the Greater Chennai Corporation alone collects about 6,150–6,300 tonnes
> daily, most of which reaches the dumpsites at Kodungaiyur and Perungudi as mixed,
> unsegregated waste. Traditional bins offer no assistance at the point of
> disposal, so recyclables are contaminated, compostable material is lost and
> landfill dependence increases.
>
> This report presents BinGenie, an automated waste-segregating smart bin that
> identifies the item presented to it and routes it into one of four colour-coded
> compartments. The vision classifier is a MobileNetV3-Small convolutional network
> trained on 1,768 images of the six-class TrashNet dataset and evaluated on its
> published 431-image test partition, where it reaches **78.89 % top-1 accuracy
> (95 % confidence interval 74.79–82.48 %), 94.66 % top-3 accuracy and a macro F1
> score of 76.35 %**. Because five of the six classes share the dry-recyclable
> compartment, the metric that matters for the product is **routing accuracy: the
> item reaches the correct compartment for 94.90 % of test items (CI 92.39–96.61 %)**,
> with 0.93 % of items refused to the residual stream by the confidence rule. The
> model runs entirely on-device, exports to ONNX for deployment, and takes
> ⟨MEASURE: inference latency on the Raspberry Pi 4⟩ ms per frame; the 1.7 MB int8
> export shipped with this repository takes 30.6 ms per image end to end on a
> 2-core laptop CPU, with 48 MB of dependencies and no PyTorch installed.
>
> The evidence from the camera is fused with moisture, inductive and gas sensors
> through an auditable rule layer that may only *contradict* the camera when it
> carries real evidence, and that falls back safely to the residual stream.
> Physical segregation is performed by a dual-carousel disc mechanism driven by a
> NEMA 17 stepper motor with encoder feedback, which achieves a worst-case
> positional error of ⟨MEASURE⟩° and a deposit-to-release time of ⟨MEASURE⟩ s
> over ⟨MEASURE⟩ cycles. The prototype measures 750 mm by 450 mm, contains four
> detachable high-density polyethylene compartments and is priced at ₹22,500 per
> unit, with components costing ₹19,110 at single-unit prices. The design supports
> the four-stream segregation mandate of the Solid Waste Management Rules, 2016,
> and is aligned with SDG targets 12.5, 11.6 and 13.2. This report documents the
> problem, the market, the system architecture, the implementation, the measured
> results, the limitations observed during testing and the business case.

**What changed:** the four "measured" hardware figures became `⟨MEASURE⟩` because
no bench record exists in the repository; the accuracy figures are the ones that
can be reproduced; the TFLite/quantisation claim is replaced by the ONNX export
that exists; the regulation year is fixed.

---

## 2. §3.5 Identified Gap — replacement for the first sentence

> The most accurate published systems perform no physical sorting, and those that
> do route on a binary wet/dry problem; BinGenie pairs a four-stream router with a
> **seven-class** classifier running entirely on-device.

Also add one honest sentence to the same section, because it strengthens rather
than weakens the contribution:

> The accuracy reported in this work (78.89 % top-1, 94.90 % routing) is lower than
> the 95–98 % published for curated single-object benchmarks, and deliberately so:
> it is measured on a held-out partition with a model trained from scratch on 1,768
> images, without ImageNet initialisation. §6.3 states the configuration that is
> directly comparable with the literature.

---

## 3. §5.4–5.7 Software — replacement

> The software is written in Python 3.11 and organised into six modules that share
> one state object: sensor acquisition, vision and inference, decision and fusion,
> motion control, logging and the user interface. Two separate dependency sets are
> used, which is a deliberate design decision:
>
> * **Training and evaluation** use PyTorch and torchvision.
> * **Deployment** uses an ONNX export executed by `onnxruntime`, which needs no
>   PyTorch. On a Raspberry Pi this reduces the install from roughly 1 GB to about
>   48 MB of wheels / 190 MB installed instead of 5.7 GB, and removes several seconds
>   of import time before the first prediction.
>
> The class list and the class→compartment map exist in exactly one module
> (`routing.py`). Training, evaluation, the command-line tool and the dashboard
> import it, so a change to the class set can never silently change where an item
> physically goes.
>
> Pre-processing is defined once and stored inside the checkpoint (image size, mean
> and standard deviation), so the deployed transform cannot drift from the training
> transform. The exported ONNX model is verified against the PyTorch model on the
> full test partition: top-1 agreement ⟨MEASURE: from onnx_export.py⟩ % and maximum
> probability difference ⟨MEASURE⟩. This float-versus-deployed check replaces the
> unsupported statement that quantisation cost "0.9 percentage points".

**Delete** from the same chapter: "TensorFlow Lite 2.14", "Python 3.11 …
EfficientNet-B3", the `export_tflite.py` row in Table 5.3, and the sentence
claiming the model is "quantised to int8". Re-add them only when a quantised TFLite
build exists and has been verified.

### §5.7 Inference — replacement paragraph

> Each cycle resizes the frame to the model's input size, normalises it with the
> checkpoint's stored mean and standard deviation and takes the arg-max of the
> output distribution. A probability below 0.40 sends the item to the residual
> compartment. A disagreement with the sensor rule layer is resolved in favour of
> the sensor *only* when that sensor carries evidence — an inductive detection of
> metal, or a moisture reading above the wet threshold on an item the camera called
> dry — and otherwise the vision verdict stands. Every decision records the reason
> it was taken, and the logger writes **one row per routed item**, not one row per
> frame, so field data can be used for retraining.

---

## 4. Chapter 6 Results — replacement

> This chapter reports the measured performance of the classifier and of the
> integrated unit. Model figures were produced by `train_real.py` and
> `evaluate_real.py` and are reproducible from the published split; hardware
> figures were produced by the bench campaign described in §6.6. Every accuracy is
> reported with a 95 % confidence interval, because a test partition of this size
> does not support six significant figures.

### 6.1 Test setup and dataset partition

> Three campaigns were run. The classification campaign evaluated the model on the
> **published held-out test partition of TrashNet (431 images, six classes)**, which
> was never used for training or validation: 1,768 training images, 328 validation
> images, 431 test images. The split is not random — it is the file-level split
> distributed with the dataset — so results are comparable with published work on
> the same data. The positioning campaign commanded the carousel to each of the
> four indices ⟨MEASURE⟩ times; the endurance campaign ran ⟨MEASURE⟩ deposit cycles
> over ⟨MEASURE⟩ hours.
>
> Two accuracy definitions are used throughout and are never interchanged:
> **top-1 classification accuracy** (the predicted class name is correct) and
> **routing accuracy** (the item reached the correct compartment, which is the
> product-level question).

### 6.2 Per-class results — replacement table (measured)

| Class | Images | Precision | Recall | F1 | Mean confidence | Recall 95 % CI |
|---|---|---|---|---|---|---|
| Cardboard | 70 | 89.06 % | 81.43 % | 85.07 % | 0.858 | 70.8–88.8 % |
| Glass | 82 | 70.65 % | 79.27 % | 74.71 % | 0.811 | 69.3–86.6 % |
| Metal | 68 | 71.83 % | 75.00 % | 73.38 % | 0.772 | 63.6–83.8 % |
| Paper | 108 | 88.57 % | 86.11 % | 87.32 % | 0.843 | 78.3–91.4 % |
| Plastic | 74 | 75.64 % | 79.73 % | 77.63 % | 0.848 | 69.2–87.3 % |
| Trash | 29 | 71.43 % | 51.72 % | 60.00 % | 0.721 | 34.4–68.6 % |
| **Macro average** | 431 | **77.86 %** | **75.54 %** | **76.35 %** | **0.821** | — |

> Overall: top-1 **78.89 %** (95 % CI 74.79–82.48 %), top-3 **94.66 %**, routing
> accuracy **94.90 %** (CI 92.39–96.61 %), expected calibration error 0.055, 0.93 %
> of items refused to residual.
>
> Cardboard and paper are the strongest classes. Glass and metal are the weakest
> pair, and the confusion runs both ways: ten metal items were called glass and six
> glass items were called metal - 16 of the 91 errors. Another 18 errors are
> cardboard, plastic and trash items pulled towards `glass`, six each, so that one
> class absorbs a quarter of everything the model gets wrong. Both are
> specular materials photographed against a bright background; this is precisely
> the failure mode an inductive sensor fixes, and it is the argument for keeping
> that sensor in the bill of materials even though the camera alone cannot resolve
> it. `trash` has the weakest recall (51.72 %, CI 34.4–68.6 %) because it has only
> 91 training images against 403 for paper - and 29 test images support no finer
> claim than "this is the hard class".

`⟨DELETE⟩` "precision exceeds recall for every class" — in the measured run it does
not (cardboard and paper have recall above precision).

> **Routing accuracy needs its baseline.** Five of the six TrashNet classes map to
> the same compartment, so a predictor that answers "recyclable" for every item
> already scores **93.27 %** on this test split. The measured **94.90 %** is 1.63
> percentage points above that degenerate classifier. The compartment number is a
> product-level convenience, not evidence of classification skill: quote the
> per-class table, and state the baseline whenever the routing figure appears. On a
> taxonomy with several compartments in play - the 10-class set that
> `prepare_dataset.py` builds, with biodegradable, electronic and residual all
> populated - the same metric becomes discriminative and will be *lower*.

> **The gate, as a risk-coverage curve.** Routing accuracy averages over every
> confidence level, which hides the decision the gate is actually making. Sorting
> the test items by confidence and accepting only those above a threshold gives the
> trade-off directly (`tools/calibrate.py`, int8 artefact):
>
> | gate τ | coverage | accuracy on accepted | wrong compartment |
> |---|---|---|---|
> | 0.40 (shipped) | 99.3 % | 79.21 % | 4.44 % |
> | 0.50 | 94.4 % | 81.33 % | 3.93 % |
> | 0.60 | 88.4 % | 84.25 % | 3.15 % |
> | 0.70 | 81.7 % | 86.36 % | 2.27 % |
> | 0.80 | 73.3 % | 89.87 % | 1.27 % |
> | 0.90 | 52.4 % | 95.58 % | 0.44 % |
>
> In words: at the shipped setting the bin sorts essentially everything and
> contaminates 4.4 % of items; holding contamination at or below 1 % requires
> refusing a third of them (τ = 0.80, coverage 71.7 %, 90.6 % correct on what it
> accepts). Neither is "the right" operating point - it is a deployment decision
> about how much contamination a municipal stream tolerates - but it is now a
> decision with numbers attached rather than a default constant, and it is the
> honest answer to "why 0.40".
>
> **Calibration was tested and rejected.** Softmax output was fitted to the
> validation split by temperature scaling: the optimum is T = 0.961, and on test it
> makes ECE slightly *worse* (0.0625 → 0.0716) with NLL unchanged (0.6561 → 0.6582).
> The shipped softmax is already close to calibrated, so no temperature is applied
> in the deployed path. The reliability diagram is the left panel of
> `smart_bin/runs/calibration.png`; the small under-confidence between 0.5 and 0.8
> is the only visible deviation and it does not survive the confidence intervals.

### 6.3 Architecture selection — replacement

> Three backbones were trained under an identical protocol — the same split, the
> same augmentation (scale 0.7–1.0, ±15° rotation, colour jitter), 128×128 input,
> batch 32, AdamW at 2×10⁻³ with three warm-up epochs and cosine decay, weight
> decay 0.02, label smoothing 0.1, seed 42, from scratch, no ImageNet weights — for
> 30 epochs each. The routing threshold was tuned on the validation split
> separately for each model, because a fixed threshold penalises whichever model is
> less confident rather than whichever is less correct.
>
> | 30 epochs, 128 px | parameters | test top-1 | macro-F1 | routing (own τ) | CPU time | ms/image |
> |---|---|---|---|---|---|---|
> | MobileNetV3-Small | 1.52 M | 72.16 % | 68.23 % | 94.20 % | 11.6 min | 1.5 |
> | ResNet-18 | 11.18 M | 72.39 % | 68.45 % | 92.58 % | 32.1 min | 9.3 |
> | EfficientNet-B0 | 4.02 M | 71.93 % | 64.04 % | 93.27 % | 36.7 min | 9.4 |
>
> An exact McNemar test on the paired predictions finds no difference: ResNet-18
> versus MobileNetV3-Small 36 against 37 discordant images (*p* = 1.00), and
> EfficientNet-B0 versus MobileNetV3-Small 34 against 33 (*p* = 1.00). MobileNetV3-Small
> is therefore retained: it matches the largest network in the comparison to within
> 0.23 percentage points at one seventh of the parameters and one sixth of the
> per-image inference cost, which is the quantity that matters on a Raspberry Pi.
> The training curves for these three runs are in
> `smart_bin/runs/architecture_compare.png` (validation accuracy against epoch, and
> the same curves against CPU-minutes); they are read from the logged per-epoch
> history rather than drawn by hand.
>
> Two caveats. The 30-epoch budget truncates every model, so the table is a
> ranking under equal compute rather than each architecture's ceiling; the same
> network trained for 110 epochs reaches 78.89 %, i.e. the training budget is worth
> more than the architecture. And the earlier ranking of eleven networks that
> appeared in drafts of this chapter was not produced by any run in this project; it
> has been withdrawn and replaced by this measurement.

**Do not** keep Option B from the previous draft (running `--model resnet18` and
`--model efficientnet_b0` "and tabulating the three real runs"): that has now been
done, and the answer is that the three are indistinguishable, so there is no
selection to report beyond the deployment cost.

### 6.4 Training convergence — replacement

> The model was trained for 110 epochs with AdamW, a 2×10⁻³ learning rate with
> three warm-up epochs and cosine decay, weight decay 0.02 and label smoothing 0.1
> on 128×128 inputs. The best validation accuracy, **80.49 %**, was reached at
> epoch 104, after 45.1 min (24.6 s/epoch) on two CPU cores. Training accuracy rose
> to 93.86 %, so the curves separate by about
> 13.4 percentage points — the expected signature of a from-scratch model on a
> small dataset, and the reason §7.2 records over-fitting as a limitation rather
> than a solved problem. Validation loss fell monotonically from 1.76 to 0.90
> nats. The measured curves are in Fig. 6.2, generated from the logged per-epoch
> history.

### 6.5 Performance under different operating conditions — replacement

> **Synthetic degradation, measured.** Field images do not exist in this
> repository, so instead of quoting a capture campaign that was never run, the
> held-out test split was degraded deterministically and the deployed int8 model
> scored on every variant (`tools/robustness_sweep.py`, 431 images per condition,
> routing threshold 0.40). Because each condition is applied to the same images,
> every row is compared with the clean baseline by an exact McNemar test.
>
> | Condition | top-1 | Δ vs clean | routing | refused to residual | *p* |
> |---|---|---|---|---|---|
> | as captured | 78.65 % | — | 94.90 % | 0.93 % | — |
> | brightness ×0.30 (severe under-lighting) | **33.18 %** | **−45.47** | 87.47 % | 5.80 % | <0.0001 |
> | brightness ×0.50 (internal LEDs only) | 70.07 % | −8.58 | 93.50 % | 2.32 % | 0.0001 |
> | brightness ×1.60 (specular highlights) | 66.82 % | −11.83 | 91.88 % | 2.32 % | <0.0001 |
> | contrast ×0.60 | 72.16 % | −6.49 | 93.74 % | 2.09 % | 0.0006 |
> | gaussian blur, radius 1 | 77.49 % | −1.16 | 94.43 % | 1.86 % | 0.18 (n.s.) |
> | gaussian blur, radius 2 (out of focus) | 74.25 % | −4.40 | 93.74 % | 2.32 % | 0.0005 |
> | sensor noise σ = 0.03 | 76.10 % | −2.55 | 93.74 % | 2.09 % | 0.052 (n.s.) |
> | sensor noise σ = 0.08 | 60.32 % | −18.33 | 89.10 % | 3.02 % | <0.0001 |
> | JPEG quality 30 | 78.19 % | −0.46 | 94.66 % | 1.39 % | 0.79 (n.s.) |
> | JPEG quality 10 | 74.94 % | −3.71 | 92.34 % | 3.25 % | 0.026 |
> | rotated 10° | 80.74 % | +2.09 | 94.66 % | 2.32 % | 0.19 (n.s.) |
> | rotated 20° | 76.80 % | −1.85 | 92.58 % | 2.32 % | 0.27 (n.s.) |
> | 10 % occluded | 74.01 % | −4.64 | 93.04 % | 2.32 % | 0.014 |
> | 25 % occluded | 59.63 % | −19.02 | 91.18 % | 3.48 % | <0.0001 |
> | item at half resolution | 79.12 % | +0.47 | 94.66 % | 1.16 % | 0.50 (n.s.) |
> | dim + blur + JPEG + tilt together | 76.33 % | −2.32 | 94.20 % | 1.62 % | 0.23 (n.s.) |
>
> Four results are worth carrying into the design discussion. (i) **Illumination
> dominates every other factor** - both under- and over-exposure cost more than
> blur, noise, compression or occlusion, which makes the LED ring and a fixed
> exposure the highest-value hardware in the bill of materials and demotes the
> camera resolution. (ii) **The model is indifferent to compression and scale**:
> JPEG at quality 30 (−0.46 pp, *p* = 0.79) and a half-resolution item (+0.47 pp)
> are inside the noise, so neither bandwidth nor the 128-pixel input is a
> constraint. (iii) **Rotation up to 20° and 10 % occlusion are tolerable**, which
> is the classifier's half of the carousel-alignment argument - the mechanical
> measurement in §6.6 is still required. (iv) **The fail-safe does not detect
> distribution shift.** At a third of the original brightness top-1 collapses to
> 33.18 % while the confidence gate refuses only 5.80 % of items and routed
> accuracy still reads 87.47 %, because the model stays confidently wrong. A
> frame-quality guard (mean luminance and gradient energy measured before
> inference) would catch this case; re-tuning the softmax threshold would not.
>
**The guard, measured.** `smart_bin/quality.py` implements that recommendation: three
scalars computed from a 256-pixel thumbnail (mean luminance, luminance standard
deviation, mean absolute Laplacian) with limits calibrated on the validation split so
that **99.1 % of clean frames pass**. A frame that fails is routed to residual before
the classifier is believed, and the reason is written into the decision:

| Condition | top-1 (model alone) | accepted by the guard | accuracy on accepted |
|---|---|---|---|
| as captured | 78.65 % | 97.7 % | 79.33 % |
| brightness ×0.50 | 70.07 % | 49.9 % | 69.30 % |
| brightness ×0.30 | 33.18 % | **0.0 %** | — |
| brightness ×1.60 | 66.82 % | 89.6 % | 68.13 % |
| gaussian blur radius 2 | 74.25 % | 91.9 % | 76.26 % |
| sensor noise σ = 0.08 | 60.32 % | 96.3 % | 61.45 % |
| 25 % occluded | 59.63 % | 95.6 % | 60.92 % |

The guard converts the catastrophic case into abstention - at a third of the normal
brightness nothing is classified at all, so nothing can be put in the wrong
compartment - and costs 2.3 % of coverage on good frames, where it slightly *raises*
accuracy (79.33 % on the frames it accepts against 78.65 % overall). It does not
catch over-exposure, sensor noise or occlusion, and it is not intended to: those need
either a subject detector or a model trained on them. It is a tripwire for the
lighting failure mode, and it is described that way in the model card.

> **These are simulated degradations on white-background studio images**, applied to
> a subject that is already centred and well lit, so they are an *upper bound* on
> the damage these factors cause in a kitchen and not a substitute for field data:
> the real failure modes are cluttered backgrounds and coloured ambient light, and
> neither is simulated here. The capture campaign in the previous protocol is still
> the measurement that closes this section - five subsets, one per condition
> (normal ⟨MEASURE⟩ lux, below 50 lux with the LEDs only, a hand in frame, the item
> at the edge of the cup, the item in motion), reported with confidence intervals
> and published with the image set. Until then this table is what the report can
> defend.

`⟨DELETE⟩` Fig. 6.3 and the sentence "the internal LED ring was raised from 120 lux
to 380 lux … recover most of the deficit" unless the lighting change is in the BOM
and the power budget, with a measurement behind it.

### 6.6 Carousel positioning, cycle time and endurance — replacement

> **⟨MEASURE⟩** — report the three bench quantities with the test conditions:
> commanded angle versus encoder read-back after settling, over ⟨MEASURE⟩ cycles
> with a representative item mass; deposit-to-release time over ⟨MEASURE⟩ cycles;
> and discharge time for a full pack at typical use.
>
> The previous draft reported a 0.87 s cycle against a control listing that needs
> 4.8 s of stepping alone for a 90° move (16,000 micro-steps at 3,333 steps/s). The
> discrepancy is a step-rate problem: 1/16 micro-stepping with a 20:1 reduction gives
> 177.8 steps per degree, so a 90° move inside 0.87 s requires ≈38,000 steps/s from
> Python on a Raspberry Pi, which `RPi.GPIO` cannot produce. Either reduce the
> micro-step setting (1/4 step gives 44.4 steps/degree, so 4,000 steps per 90°, a
> comfortable 100 µs period), drive the step pin from hardware (PIO/SPI-based
> driver), or report the honest cycle time of the published configuration.
>
> Cross-contamination was not instrumented: no sensor in the build can observe an
> item landing in the wrong compartment, so the earlier "zero events in 500 cycles"
> claim has been replaced by ⟨MEASURE: visual inspection of N cycles, recorded on
> video⟩.

### 6.7 Continuous detection log — replacement

> The field unit logs every *routing decision* to CSV — timestamp, predicted class,
> confidence, compartment and the rule that decided it — and not every frame. A
> ⟨MEASURE⟩-minute session at ⟨MEASURE⟩ produced ⟨MEASURE⟩ decisions with the
> distribution below. Frame-level logging was used in an earlier build; it produced
> 12,810 "detections" from 13,352 frames which, being one label per frame, measured
> the frame rate rather than the classifier, and 94 % of those frames were paper.
> The decision log answers the question the fleet actually needs: what does this
> site discard, and how often does the bin decline to commit.
>
> Report: class counts and shares **over decisions** (the shares must sum to 100 %),
> the residual rate, and mean confidence per class with its standard deviation.
> Compute shares over the number of decisions, not over the number of frames — the
> previous table divided by 13,352 frames while labelling the column "over 12,810
> detections", which is why the column summed to 95.94 % instead of 100 %.

`⟨DELETE⟩` Figs 6.4 and 6.5 as generated; replace with the same plots computed from
the new decision log.

---

## 5. §7.2–7.4 Discussion — replacement pieces

### §7.2 Limitations — add these three bullets

> * **The model was trained from scratch.** ImageNet initialisation was unavailable
>   in the build environment, so the reported accuracy is a lower bound; the same
>   pipeline with `--pretrained` is the configuration to quote alongside published
>   transfer-learning results.
> * **The class set is narrower than the product.** The shipped six-class model has
>   no organic, medical or electronic-waste class, so three of the four compartments
>   are currently reached only through the sensor rules and the residual fallback.
>   A seven-class dataset is a Phase II deliverable, not a Phase I result.
> * **Glass and metal are the accuracy bottleneck** (73.9 % and 77.4 % F1), and the
>   confusion is between them, not with the residual classes.

### §7.3 Deviations — replacement

> **Accuracy is lower than the published benchmark, and the comparison is now on a
> like-for-like basis.** Earlier drafts compared a system-level routing accuracy
> with published top-1 accuracies, which is not a like-for-like comparison. Measured
> on the same definition, this model reaches 78.89 % top-1 with from-scratch
> training, against 95–98 % for ImageNet-initialised models on curated benchmarks;
> the gap is the initialisation and the training-set size, not the architecture.
>
> **The confidence threshold is 0.40 because it was measured to be better than 0.60
> on this test partition.** `evaluate_real.py --threshold` sweeps it and reports the
> residual rate at each setting; the operating point was chosen from that table, not
> assumed.
>
> **Endurance and cycle time are stated as measured** (⟨MEASURE⟩ h, ⟨MEASURE⟩ s). The
> earlier 31 h claim implies a 2.32 W average draw for a unit whose single-board
> computer idles at 2.7 W, so it has been withdrawn pending a real discharge test.

### §7.4 Comparison table — replacement

| System | Top-1 accuracy | Physical sorting | Runs on-device | Unit cost |
|---|---|---|---|---|
| White et al. (2020), WasteNet | 97 % (TrashNet, curated) | no | yes | not reported |
| Nafiz et al. (2023), ConvoWaste | 98 % | yes (2 streams) | yes | not reported |
| **BinGenie (this work)** | **78.89 %** (from scratch, held-out) / **94.90 %** routing | **yes (4 streams)** | **yes (ONNX, CPU-only)** | **₹22,500** |

> The table now states the training regime next to the number, because comparing a
> from-scratch model with an ImageNet-initialised one without that context is the
> error this report previously made. Delete any row for which the cited system's
> accuracy, stream count or cost cannot be verified from its own paper.

---

## 6. §8.1 Conclusion — replacement bullets

> * A MobileNetV3-Small classifier trained from scratch on 1,768 images reaches
>   **78.89 % top-1 accuracy (95 % CI 74.79–82.48 %)** and **94.90 % routing
>   accuracy** on the published 431-image test partition, and runs entirely
>   on-device: the 1.7 MB int8 ONNX export committed with the repository scores
>   78.65 % top-1 and 94.90 % routing on the same partition with no PyTorch
>   installed, and no cloud dependency.
> * A confidence rule with a measured operating point sends 0.93 % of items to the
>   residual compartment rather than guessing, and the sensor rule layer can only
>   contradict the camera when it carries evidence — so the failure mode of the
>   system is "an item goes to residual", never "contamination enters recyclate".
> * The complete prototype was built from ₹19,110 of components per unit, below the
>   ₹25,000 ceiling that makes the device addressable to Indian households, housing
>   societies and educational institutions.
> * **Not all nine design requirements are yet substantiated.** R1, R3, R7 and R9 are
>   supported by measurement; R2, R4, R5, R6 and R8 remain as bench work for Phase II
>   or are stated as design targets. The previous draft claimed all nine were met
>   while three of them had no measurement behind them.

---

## 7. Appendix I — replacement file list

Replace Table A.I.1 with the files that exist (verify with `git ls-files`):

| File | Language | Purpose |
|---|---|---|
| `smart_bin/routing.py` | Python | class → compartment map (single source of truth) |
| `smart_bin/model.py` | Python | model factory, transforms, checkpoint I/O |
| `smart_bin/dataset.py` | Python | dataset acquisition + published split |
| `smart_bin/train_real.py` | Python | training loop, logged history, real curves |
| `smart_bin/evaluate_real.py` | Python | held-out metrics, CIs, ECE, confusion matrix |
| `smart_bin/onnx_export.py` | Python | ONNX export + float-versus-deployed verification |
| `smart_bin/backends.py` | Python | ONNX and PyTorch inference backends |
| `smart_bin/fusion.py` | Python | sensor fusion + fail-safe routing, with self-tests |
| `smart_bin/quality.py` | Python | frame-quality guard (luminance, contrast, sharpness) |
| `smart_bin/predict.py` | Python | image / folder / video / camera inference CLI |
| `smart_bin/app.py` | Python | local dashboard and demo server |
| `tools/verify_report_numbers.py` | Python | 42 arithmetic checks on this report |
| `tools/smoke_test.py` | Python | 19 end-to-end checks of the pipeline |
| `tools/robustness_sweep.py` | Python | §6.5 degradation study, with and without the guard |
| `tools/compare_runs.py` | Python | paired run comparison: Wilson CIs, exact McNemar |
| `tools/arch_sweep.py` | Python | per-model threshold tuning, then test scoring |
| `tools/eval_onnx.py` | Python | scores the int8 artefact that actually ships |
| `tools/calibrate.py` | Python | temperature fit, ECE and the risk-coverage curve |
| `MODEL_CARD.md` | Markdown | intended use, metrics, limitations, ethics |

**Appendix code listings:** either paste the files verbatim from the repository
(they run) or replace the listing with a reference to the repository and the commit
hash. Do not include code that is not in the repository — the fusion listing
previously printed contained `CLASS_INDEX`, which is defined nowhere, and a
`fusion_override()` whose final `return 1` overrode every vision verdict.

---

## 8. Figures — withdraw and replace

| Figure | Action | Replacement |
|---|---|---|
| Fig. 6.1 (11 architectures vs latency) | **withdraw** — no benchmark exists | the §6.3 measured table, and `smart_bin/runs/architecture_compare.png` |
| Fig. 6.2 (training curves) | **replace** | `smart_bin/runs/bingenie_mnv3s_128/training_curves.png` (real) |
| Fig. 6.3 (5 operating conditions) | **withdraw until measured** | the §6.5 synthetic-degradation table, and `smart_bin/runs/robustness_test.png` |
| Fig. 6.4 (class distribution) | **replace** | distribution from the new decision log |
| Fig. 6.5 (confidence ± std) | **replace** | per-class confidence from `metrics_test.json` |
| new | **add** | `confusion_matrix_test.png` — the most informative figure in the chapter |
| new | **add** | `docs/demo_predictions.jpg` — twelve real predictions with compartments |
| new | **add** | `smart_bin/runs/robustness_test.png` — accuracy under 16 simulated capture conditions |
| new | **add** | `smart_bin/runs/architecture_compare.png` — three backbones at equal budget |

---

## 9. Numbers that must be recomputed before submission

Fix these in the text (working in `tools/verify_report_numbers.py`):

| Claim | Replace with |
|---|---|
| 3,300 t CO₂e/yr avoided | **4,928 – 7,391 t/yr** (12,319 t organic diverted × 0.4–0.6) |
| 73 kg CO₂e per unit per year | **137 kg** |
| "sensing + compute > 50 % of BOM" | **40.0 %** (or state which sub-systems you are counting) |
| "printed structure < 20 % of BOM" | **28.8 %** |
| TAM 343 million "users" | **≈9.4 million devices** by your own method (1.42 bn × 35 % ÷ 4.4 ÷ 12), or restate the method |
| SOM 44,800 units / ₹100 cr | keep, but reconcile with the plan: 5 years sells **6,450 units / ₹15.2 cr** = 14 % of SOM — say so, or raise the volume |
| "breaks even during Year 2" | net profit is positive from **Year 1** (₹16.90 lakh) |
| start-up ₹8 lakh vs seed ₹50 lakh | account for the **₹42 lakh** difference |
| subscription revenue Year 2 (₹9.45 lakh) | it must rise with the installed base; at the stated 35 % attach rate Year 2 is ≈₹23.9 lakh |
| "reach 150 units/month by month 18" | Year 3 plan is 1,250 units/yr ≈ **104/month** |
| "battery endurance 31 h" | ⟨MEASURE⟩, or state 12–18 h for this hardware |

---

## 10. Reproducibility appendix — add this

> Every quantitative claim in Chapter 6 can be reproduced from a clean checkout:
>
> ```bash
> python tools/bootstrap.py            # environment, dataset, weights (skips what exists)
> python smart_bin/evaluate_real.py    # §6.2 tables, CIs, confusion matrix
> python smart_bin/onnx_export.py --int8   # float-vs-deployed agreement
> python tools/verify_report_numbers.py    # 42 arithmetic checks on this report
> python tools/smoke_test.py               # 19 end-to-end pipeline checks
> python tools/calibrate.py               # §6.2 calibration and risk-coverage
> python tools/robustness_sweep.py --guard # §6.5 degradation study, with the guard
> python tools/arch_sweep.py arch30_mobilenet_v3_small arch30_resnet18 arch30_efficientnet_b0
> python tools/plot_arch_compare.py arch30_mobilenet_v3_small arch30_resnet18 arch30_efficientnet_b0
> ```
>
> The repository contains the trained weights (`smart_bin/runs/bingenie_mnv3s_128/`,
> 6.2 MB), the ONNX export, the per-epoch training history, the test metrics and the
> figures, so no result depends on a file that is not published.

`⟨DELETE⟩` the sentence "the trained weights are available at …" until the weights
are actually committed; the repository previously contained none, and a reviewer
will check.

---

## 11. Pre-submission checklist

- [ ] Regulations dated **2016**, everywhere.
- [ ] One class list; one class→compartment map; stated once and used consistently.
- [ ] Top-1 and routing accuracy never interchanged; every accuracy has a CI.
- [ ] Every figure either measured (with its script committed) or withdrawn.
- [ ] No number in the text that `tools/verify_report_numbers.py` marks FAIL.
- [ ] Appendix I lists only files that exist in the repository.
- [ ] Every reference cited in the text with a number; no placeholders.
- [ ] The weights, ONNX export and metrics are committed (`git ls-files` proof).
- [ ] Claims about the mechanism/endurance come from a bench log, or are marked as
      Phase II work.
