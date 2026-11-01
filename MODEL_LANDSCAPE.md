# Waste-detection models that already exist — landscape and shortlist

*Compiled 1 November 2026 for the BinGenie project. Every number below is quoted
from the source named next to it; nothing here was measured in this repository
except where it says "this repo".*

---

## 0. Read this before comparing any two numbers

Waste-model papers report accuracies on **different splits of different datasets**,
and several of the highest published figures are inflated by near-duplicates and
augment-before-split leakage:

* TrashNet has documented near-duplicate photographs; grouping by perceptual hash
  before splitting cost one independent reimplementation several points, and its
  authors state plainly that "accuracy on TrashNet may be somewhat optimistic"
  ([leakage-aware benchmark](https://github.com/kang-prayit/n1-trustworthy-waste-classification),
  [EcoVision](https://github.com/kashifabrar24/EcoVision)).
* RealWaste's authors showed the same models score **89.19 %** on real landfill
  photographs but only **49.69 %** when trained on pristine studio material and
  tested in the real environment ([RealWaste, *Informatics* 14(12):633](https://www.mdpi.com/2078-2489/14/12/633)).
* In industrial cluttered scenes the absolute numbers collapse: Mask R-CNN gets
  **22.8 AP** on ZeroWaste versus 87.2 mAP on a curated set
  ([ZeroWaste](https://ar5iv.labs.arxiv.org/html/2106.02740),
  [review](https://www.researchgate.net/publication/395049831_A_Review_of_Computer_Vision_for_Industrial-Grade_Waste_Classification)).

**Therefore: the only comparison that matters for this project is a candidate model
scored on our own 431-image held-out split with `tools/verify_deployment.py`.**
The list below is for shortlisting, not for quoting.

---

## 1. Shortlist (ranked for *this* project)

| # | Model | Task | Reported score (source) | Params | Weights | Why it is on the list |
|---|---|---|---|---|---|---|
| 1 | **ImageNet-pretrained MobileNetV3-Small / MobileNetV2**, fine-tuned | 6-class classifier | 88.7 – 91.0 % test on TrashNet, leakage-aware splits ([1](https://github.com/Zaidranjha095/waste-image-classification), [2](https://github.com/samarth1234-dev/trashnet-ai), [3](https://github.com/kashifabrar24/EcoVision)) | 1.5 – 3.5 M | torchvision (already installed) | Same architecture we ship, same pre-processing, **one flag away** (`train_real.py --pretrained`). The cheapest ~10 pp in the whole list. |
| 2 | **EfficientNet-B0**, fine-tuned | 6-class classifier | 92 % test, weighted F1 0.92 on TrashNet ([JTRSS comparison](https://journal.umk.edu.my/index.php/jtrss/article/download/2492/1364/11526)) | 5.3 M | torchvision | Beats ResNet-18 (90 %) under an identical protocol; still CPU-friendly, exports to ONNX cleanly. |
| 3 | **Trash-Net** — SigLIP2-base fine-tune | 6-class classifier | 96.26 % accuracy / 0.957 macro-F1 on 5,054 eval images ([model card](https://huggingface.co/prithivMLmods/Trash-Net)) | 92.9 M | 🤗 `prithivMLmods/Trash-Net` | Same six class names as ours, so it drops straight into our routing table — the fastest way to test whether a vision-language backbone beats a from-scratch CNN. |
| 4 | **Augmented-Waste-Classifier-SigLIP2** | 10-class classifier | 99.26 % on 38,686 images ([model card](https://huggingface.co/prithivMLmods/Augmented-Waste-Classifier-SigLIP2)) | ~93 M | 🤗 Apache-2.0 | Adds biological / battery / clothes / shoes — closer to the four *physical* streams than TrashNet is. Treat 99.26 % as an augmented-set upper bound. |
| 5 | **YOLO26n + our classifier on the crop** | detection then classify | COCO mAP50-95 40.9, 38.9 ms CPU ONNX ([Ultralytics](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)) | 2.4 M | Ultralytics | Needed only if the bin must handle *several* items or heavy clutter per frame. Newer and ~30 % faster on CPU than YOLO11n. |
| 6 | **YOLOv8m, 11 waste classes**, released weights | detection | val mAP50 **0.789**, test mAP50 **0.761** ([release](https://github.com/Santanu12-paria/AI-Waste-Detection/releases/tag/v1.0)) | 25.9 M | GitHub release | One of the few *waste* detectors with weights actually published and a test-split number. |
| 7 | **YOLOv8s on TACO** | detection, 18–60 classes | mAP50 0.67, mAP50-95 0.45 ([project](https://github.com/AssadUllahKhan/Object-Detection-For-Waste-Management-Project)) | 11.2 M | repo weights / retrain | Real in-the-wild litter, many classes; useful as a *detector* upstream of a 4-stream router, not as a classifier. |
| 8 | **OpenCLIP ViT-L/14-2B, zero-shot + prompt tuning** | 6-class classifier | 82.71 % zero-shot → **90.48 %** after prompt engineering ([Novelis / Waste Management](https://novelis.io/research-lab/a-comparative-analysis-of-vision-language-models-for-scalable-waste-recognition/)) | ~430 M | OpenCLIP | No training data needed. 430 M params is a laptop/edge-GPU proposition, not a Pi one. |
| 9 | **ConvNeXt-Tiny / ConvNeXt-Large**, fine-tuned | 6-class classifier | 85 % / 93 % on TrashNet ([MDPI ensemble study](https://www.mdpi.com/2071-1050/18/1/24)) | 28 M / 198 M | torchvision | Largest single-model gains in that study; a weighted ensemble reached 96 %. |
| 10 | **Mask R-CNN / YOLO on ZeroWaste** | instance segmentation | 22.8–30 AP (Mask R-CNN), **51.6 mAP** current SOTA ([ZeroWaste](https://ar5iv.labs.arxiv.org/html/2106.02740), [2025 baseline](https://arxiv.org/pdf/2508.18799)) | 44 M+ | detectron2 / Ultralytics | Only relevant if the arm must grab a specific object off a belt. Whole different project. |

Everything above is *downloadable*. Section 6 covers the commercial systems that
are better than all of them but cannot be installed.

---

## 2. Ready-to-use classifiers on the Hugging Face Hub

All of these are `transformers`-loadable in five lines, which matters because it
means no training run at all to get a baseline.

| Model | Backbone | Classes | Reported accuracy | Eval size | Licence | Notes |
|---|---|---|---|---|---|---|
| [`prithivMLmods/Trash-Net`](https://huggingface.co/prithivMLmods/Trash-Net) | SigLIP2-base-p16-224 | 6 (TrashNet names) | **96.26 %**, macro-F1 0.957 | 5,054 | check card | Per-class F1 0.91–0.98; `trash` weakest (0.914) — same failure mode as ours |
| [`prithivMLmods/Augmented-Waste-Classifier-SigLIP2`](https://huggingface.co/prithivMLmods/Augmented-Waste-Classifier-SigLIP2) | SigLIP2-base-p16-224 | 10 | 99.26 % | 38,686 | Apache-2.0 | Heavily augmented set — upper bound, not field accuracy |
| [`ysfad/openclip-finetune-waste`](https://huggingface.co/ysfad/openclip-finetune-waste) | CLIP ViT-B/16 (laion2b) | **30** fine-grained (aerosol cans, glass jars, …) | 91.33 % val | 15,000 (70/10/20) | check card | Most useful *vocabulary*: maps to recycling streams far better than 6 coarse classes |
| [`yangy50/garbage-classification`](https://huggingface.co/yangy50/garbage-classification) | ViT-base-p16-224 | 6 | 95 % (self-reported, 2,467 imgs) | small | AFL-3.0 | The original open ViT on the Kaggle garbage set; treat 95 % with suspicion |
| [`watersplash/waste-classification`](https://huggingface.co/watersplash/waste-classification) | ViT (85.8 M) | 6 | not stated | — | check card | Fine-tune of the above |
| [`hmrizal/recycled_waste_classification`](https://huggingface.co/hmrizal/recycled_waste_classification) | ViT-base-p16-224 | recycled set | 80.23 % | ~3 k | check card | Realistic number for a plain 5-epoch fine-tune |
| [`Giecom/…-vit-model-clasification-waste`](https://huggingface.co/Giecom/giecom-vit-model-clasification-waste) | ViT-base | recycled (3.1 k) | 99.74 % | 3.1 k | check card | Textbook leakage: 99.74 % on a 3 k set with no held-out protocol — cited only as a caution |
| [`reyhanwiira/…-vgg-realwaste`](https://huggingface.co/reyhanwiira/eunoiawiira-vgg-realwaste-classification) | VGG16 | 9 (RealWaste) | 83.65 % | RealWaste | check card | Real-life photos, not studio |

**Practical caveat:** none of these was trained on *our* split, and SigLIP2/CLIP
models pre-process at 224 px with their own normalisation — the routing accuracy
they produce in our app has to be measured, not assumed.

---

## 3. Object detectors (when one image holds several items)

| Model | Dataset | Score | Weights |
|---|---|---|---|
| YOLOv8m, 11 waste classes | custom waste set | mAP50 0.789 val / **0.761 test** | [released `best.pt`](https://github.com/Santanu12-paria/AI-Waste-Detection/releases/tag/v1.0) |
| YOLOv8s | TACO (60 litter categories, often collapsed to ~18) | mAP50 0.67, mAP50-95 0.45 | [repo](https://github.com/AssadUllahKhan/Object-Detection-For-Waste-Management-Project) |
| YOLOv8 + TACO | TACO | train/infer scripts + run weights | [jeremy-rico/litter-detection](https://github.com/jeremy-rico/litter-detection) |
| YOLO26n / YOLO11n | COCO (backbone only) | mAP50-95 40.9 / 39.5; 38.9 / 56.1 ms CPU | Ultralytics, `yolo26n.pt` |
| Faster R-CNN / YOLOv5 / YOLOv8 comparison | Roboflow "Garbage Classification 3" (7 classes) | side-by-side mAP, FPS, latency | [repo](https://github.com/GhadaElfeki77/Garbage-Object-Detection) |

Detection is the right answer only if the frame can contain more than one object.
For a single item presented to a bin, detection adds a failure mode (missed box)
without adding routing accuracy.

---

## 4. Segmentation (robot picking off a conveyor)

| Model | Dataset | Score |
|---|---|---|
| SPDA-SAM (ViT-B, 2026) | ZeroWaste test | **34.4 mAP** |
| USIS-SAM (ViT-H) | ZeroWaste test | 33.8 mAP |
| Mask R-CNN (R-50) | ZeroWaste test | 29.0 mAP |
| Mask R-CNN / TridentNet / RetinaNet | ZeroWaste-f (authors' baselines) | 22.8 / 21.0 AP |
| Semi-supervised YOLO11 + RT-DETR ensemble | ZeroWaste-f | **51.6 mAP** (2025 state of the art) |

Sources: [SOTA2 leaderboard](https://www.sota2.com/research/sota/instance-segmentation-on-zerowaste-test),
[ZeroWaste paper](https://ar5iv.labs.arxiv.org/html/2106.02740),
[Robust and Label-Efficient Deep Waste Detection](https://arxiv.org/pdf/2508.18799).

The lesson here is not "use ZeroWaste", it is "cluttered real scenes are 20–50 mAP,
not 90 % accuracy" — which is why the report should never compare a TrashNet
accuracy with a field deployment figure.

---

## 5. Zero-shot / vision-language models

| Approach | Score on waste | Comment |
|---|---|---|
| CLIP RN50, zero-shot | 72.93 % | [Waste Management 2025](https://www.sciencedirect.com/science/article/pii/S0956053X25003502) |
| OpenCLIP ViT-L/14-2B, zero-shot | 82.71 % → **90.48 %** with engineered prompts | Prompt text matters more than the model here |
| MetaCLIP ViT-H/14 | 84.96 % zero-shot, 97.74 % fully supervised | Best zero-shot of the compared family |
| Open-vocabulary *detectors* (Grounding DINO, OWLv2, YOLO-World) on ZeroWaste | **mAP ≤ 7.3** zero-shot | [ZeroWaste OVOD benchmark](https://arxiv.org/pdf/2508.18799) — do not expect open-vocabulary detection to work on a bin |

Vision-language models are attractive for one specific reason: they need no
training data, so the ~1,768-image ceiling stops applying. They are unattractive
for a Raspberry-Pi-class bin because of the compute.

---

## 6. Commercial systems (better, not installable)

Worth one paragraph in the report for context, not more:

* **Greyparrot** (UK) — waste-monitoring AI for conveyor belts.
* **AMP Robotics** (US) — vision-guided robotic sorters used in material recovery
  facilities.
* **CleanRobotics TrashBot** (US) — a sorting bin with camera plus sensors, the
  closest commercial form factor to BinGenie.

None of them publishes weights or evaluation data; the published numbers are
vendor claims, which is exactly why the industrial ZeroWaste benchmark in §4
exists. Cite them as context for "this is a solved problem commercially", never as
a comparison baseline.

---

## 7. What is different about *our* numbers

Measured in this repository (2 CPU cores, held-out 431-image TrashNet test split,
`metrics_test.json`): **78.89 % top-1 / 94.90 % routing, trained from scratch**.

| Same dataset, different protocol | Test accuracy |
|---|---|
| MobileNetV3-Small, **from scratch** (this repo) | 78.89 % |
| MobileNetV2, ImageNet init, pHash-grouped split | 89.71 % [86.54, 92.61] |
| MobileNetV2, ImageNet init, 1768/379/380 split | 88.68 % |
| MobileNetV3-Small + augmentation, 70/15/15 | 91.03 % |
| EfficientNet-B0, ImageNet init, same protocol as ResNet-18 | 92.00 % |

That table is the single most useful thing in this document: **about ten points of
our gap is the missing ImageNet initialisation, not the architecture.** It also
means switching to a bigger model without changing the initialisation policy would
buy much less than the papers imply.

### 7.1 The one upgrade experiment we did run

Instead of trusting that argument, `bingenie_mnv3s_160` was trained on the same
split with the same recipe plus the regularisation that usually pays for extra
resolution: 160 px input, MixUp 0.2, EMA 0.999, balanced sampling. 90 epochs,
45.8 min on the same 2 cores, best validation 0.835 and still rising when the
budget ran out.

The first comparison looked like a wash, but it was not a fair one: it held the
routing threshold at 0.40, a value tuned on the *previous* model. The 160 px model
is systematically less confident (mean top-1 0.733 vs 0.821), so a fixed 0.40 sent
6.03 % of items to `residual` instead of 0.93 %, and routing accuracy fell. Re-tuning
the threshold on validation for each model separately (`evaluate_real.py --sweep`,
`tools/eval_onnx.py --sweep`) gives the like-for-like number:

| Same 431 test images, threshold tuned per model on val, hflip TTA on | 128 px (shipped) | 160 px + MixUp + EMA |
|---|---|---|
| top-1 accuracy | 79.81 % (75.8–83.3) | **81.21 %** (77.2–84.6) |
| routing accuracy | 96.06 % (93.8–97.5) | **97.22 %** (95.2–98.4) |
| macro-F1 | 77.44 % | **80.06 %** |
| residual rate | 0.00 % | 0.70 % |
| trash recall | 51.7 % | **69.0 %** |
| glass recall | **80.5 %** | 76.8 % |
| inference latency | 3.4 ms | 5.1 ms |

McNemar, exact, on the paired predictions: top-1 23 vs 29 discordant (**p = 0.49**),
routing 6 vs 11 (**p = 0.33**). So the honest verdict is **+1.4 pp, inside the noise**:
the model is better on the class we care about most (`trash`, +17 pp) and worse on
`glass`, and 431 images cannot resolve a difference that small. It is not worth
doubling the artefact to ship it, but it is a genuinely well-behaved run and it
belongs in the report as the second data point.

The same sweep found free headroom in the model we already ship. Scoring the *int8
ONNX artefact itself* on the test split:

| shipped `bingenie_mnv3s_128_int8.onnx` | top-1 | routing | residual rate | latency |
|---|---|---|---|---|
| threshold 0.40, no TTA (as shipped) | 78.65 % | 94.90 % | 0.93 % | 30.6 ms |
| threshold 0.30, no TTA | 78.65 % | 95.13 % | 0.23 % | 26.2 ms |
| threshold 0.30 + hflip TTA | 79.35 % | **96.06 %** | 0.00 % | 58.6 ms |

The trade-off is real and should be stated as such: on validation the model is never
less than 0.30 confident, so lowering the threshold to 0.30 recovers routing accuracy
but **effectively disables the fail-safe** — nothing is ever refused. 0.40 is kept as
the shipped default because a bin that abstains on a genuinely ambiguous item is a
feature, not a bug; the numbers above exist so the choice is documented rather than
implicit.

### 7.2 Three backbones, one budget

The report used to quote a ranking of eleven architectures that no run in this
repository supported. Section 6.3 of `REPORT_REWRITE.md` promised a controlled
comparison instead; this is it. Identical protocol for all three — same split, same
augmentation, 128 px, batch 32, AdamW at 2×10⁻³ with 3 warm-up epochs and cosine
decay, weight decay 0.02, label smoothing 0.1, seed 42, **from scratch, no ImageNet
weights** — and the same 30-epoch budget, because on two CPU cores that is already
80 minutes of compute in total. The routing threshold is tuned per model on
validation (`tools/arch_sweep.py`) rather than borrowed from the shipped run, for
the reason established in §7.1.

| 30 epochs, 128 px | params | test top-1 | macro-F1 | routing (own τ) | CPU time | ms/image |
|---|---|---|---|---|---|---|
| MobileNetV3-Small | 1.52 M | 72.16 % | 68.23 % | 94.20 % | **11.6 min** | **1.5** |
| ResNet-18 | 11.18 M | **72.39 %** | **68.45 %** | 92.58 % | 32.1 min | 9.3 |
| EfficientNet-B0 | 4.02 M | 71.93 % | 64.04 % | 93.27 % | 36.7 min | 9.4 |

Exact McNemar on the paired test predictions: ResNet-18 vs MobileNetV3-Small, 36 vs
37 discordant images, **p = 1.00**; EfficientNet-B0 vs MobileNetV3-Small, 34 vs 33,
**p = 1.00**. Nothing separates them.

Two things follow, and both matter more than the ranking:

* **Architecture is not the lever at this data scale.** Seven times the parameters
  and six times the inference cost buy 0.23 pp, which is not distinguishable from
  zero on 431 images. The levers that *were* measured to move the number are the
  training budget (same network, 110 epochs instead of 30: **78.89 %**, +6.7 pp for
  33 extra minutes) and, per §7, the ImageNet initialisation.
* **Validation does not rank them either.** ResNet-18 was the best on validation
  (71.95 % vs 69.82 % for MobileNetV3-Small, +2.1 pp) and the two are inseparable on
  test (+0.23 pp). With 328 validation images, a 2-point gap is a coin flip.

The curves are in `smart_bin/runs/architecture_compare.png`. The right-hand panel is
the one to look at for an embedded target: MobileNetV3-Small crosses 70 % validation
accuracy after about 10 CPU-minutes, ResNet-18 needs 22, and EfficientNet-B0 does not
get there inside its 37-minute budget — it is a long-schedule architecture and this
experiment never gave it one.

Caveat, stated plainly: 30 epochs is a truncated budget for every model, so the
table is a *relative* ranking under equal compute, not each architecture's ceiling.
The shipped model was trained for 110 epochs and lands 6.7 pp higher than its own
30-epoch entry above.

### 7.3 The obvious fix for `trash`, tried and rejected

`trash` is the failure class: 91 training images against 403 for paper, and recall
that has sat near 50 % in every run. The textbook answer is class-balanced sampling,
so it was run with everything else held fixed — same 128 px input, same 110-epoch
schedule, same seed, `--balanced` (inverse-frequency sampling, `trash` drawn 4.4×
as often):

| same protocol, `--balanced` the only difference | shipped | balanced |
|---|---|---|
| test top-1 | **78.89 %** | 77.26 % |
| macro-F1 | **76.35 %** | 74.95 % |
| routing accuracy | 94.90 % | **95.36 %** |
| `trash` recall | **51.72 %** | 48.28 % |
| `trash` precision | 71.43 % | **82.35 %** |
| `plastic` recall | **79.73 %** | 70.27 % |
| `cardboard` recall | 81.43 % | **85.71 %** |

Exact McNemar on top-1: 34 vs 27 discordant images, **p = 0.44**; on routing, 8 vs
10, **p = 0.81**. Neither difference is significant, so the honest statement is that
re-weighting did not help: it moved `trash` from a recall-heavy to a
precision-heavy operating point without making the class more separable, and it cost
`plastic` nine points. The class is limited by the *variety* of its 91 images, not by
how often they are drawn — which is why the recommendation in §8.3 is more data, and
specifically more kinds of trash, rather than a different loss.

---

## 8. Recommendation

**Ship what we have.** For this project — a single item in front of a fixed camera,
onnxruntime on a Pi-class CPU, 2,527 training images, no GPU — the best deployable
model is the one that is already in the repository: **MobileNetV3-Small, 128 px,
trained from scratch, exported to 1.7 MB int8 ONNX**, at 78.7 % top-1 and 94.9 %
routing, 30.6 ms per image. Nothing in the shortlist above beats it *at equal
deployment cost*: the 96 % SigLIP2 checkpoint is 92.9 M parameters, and the honest
RealWaste ceiling for any conventional model is ~89 %.

Ranked by value per hour spent, on our own evidence:

1. **Do nothing to the architecture.** §7.2 shows ResNet-18 at seven times the
   parameters scoring +0.23 pp (p = 1.00), and §7.1 shows the 160 px run buying
   +1.4 pp (p = 0.49) for twice the artefact. Report both as ablations; ship
   neither.
2. **The only lever with a known large payoff is ImageNet initialisation**
   (≈ +10 pp by §7). It is one command, it needs a machine with internet, and it is
   already wired up:
   `python smart_bin/train_real.py --pretrained --name bingenie_pretrained`.
   Re-export, re-evaluate, re-run `tools/eval_onnx.py --sweep` to re-pick the
   threshold, and swap the artefacts only if `tools/compare_runs.py` clears p < 0.05.
3. **If the vocabulary is the real complaint** (our `trash` class has 91 training
   images), the move is not a bigger model and, per §7.3, not class re-weighting
   either: the 30-class CLIP fine-tune or a RealWaste-trained 9-class head, plus more
   *kinds* of trash. That is a data-collection project as much as a modelling one.
4. **Do not** start with detection or segmentation. Our frame holds one object and
   the confidence threshold already handles ambiguity; detectors would add an
   annotation problem to a labelling problem we have not solved.

Every candidate must be scored with the same script and the same 431 images before
any of it goes in the report:

```bash
python smart_bin/evaluate_real.py --run smart_bin/runs/<name> --split test --tta --sweep
python tools/compare_runs.py --baseline bingenie_mnv3s_128 --candidate <name> --tta --threshold <tuned>
python smart_bin/onnx_export.py --checkpoint smart_bin/runs/<name>/best.pt --int8
python tools/eval_onnx.py --run <name> --split test --threshold <tuned> --tta
```

---

## 9. Reference list

**Datasets** — TrashNet (2,527 imgs, 6 classes, MIT) · RealWaste (4,752 imgs,
9 classes, UCI 908; licence stated as CC BY 4.0 by UCI and CC BY-NC-SA 4.0 by the
authors — resolve before use) · TrashBox (17,785 imgs, 7 classes incl. medical and
e-waste) · TACO (in-the-wild litter, 60 fine classes) · ZeroWaste (1,752 imgs,
4 classes, industrial MRF) · Kaggle *Waste Classification* (organic/recyclable).

**Papers** — RealWaste, *Informatics* 2023 · ZeroWaste, CVPR 2021 ·
Robust and Label-Efficient Deep Waste Detection, BMVC 2025 ·
A Systematic Review of AI-Based Techniques for Automated Waste Classification,
*Sensors* 25(10):3181 · A Review of Computer Vision for Industrial-Grade Waste
Classification, 2025 · TrashBox, FRUCT 2022.
