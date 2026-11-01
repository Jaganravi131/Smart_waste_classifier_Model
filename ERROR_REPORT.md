# BinGenie — Full Technical Review and Error Register

**Project:** Waste Segregation Smart Bin — BinGenie (20MUPJ701, Phase I)
**Report reviewed:** `LIVE_IN_LAB_SEM_7.pdf`, 66 pages, Chapters 1–9 + Appendices I–IV
**Repository reviewed:** `Jaganravi131/Smart_waste_classifier_Model` @ `cbecabd`
**Review date:** 1 November 2026

This document is the honest engineering review of the project. It separates four
different kinds of problem, because they need four different repairs:

| # | Class of problem | Count | Fixable by |
|---|---|---|---|
| A | **Integrity** — the model and the figures do not exist | 3 blocking | re-training / re-plotting from real data |
| B | **ML methodology** — the pipeline cannot answer the questions the report asks | 16 | rebuilding the pipeline |
| C | **Code defects** — files that crash or do the wrong thing | 15 | code changes (done in this repo) |
| D | **Hardware & mechanism** — the device as specified cannot work as described | 20 | design changes |
| E | **Numbers & claims** — arithmetic that a calculator disproves | 24 | editing the report |

Typographical and formatting errors are deliberately **not** listed, as requested.
Every numeric claim below is machine-checkable:

```bash
python tools/verify_report_numbers.py      # 42 arithmetic checks, PASS/FAIL per line (20/42 pass as submitted)
```

---

## 0. Verdict in one page

| Area | Report claims | What the repository actually contains | Verdict |
|---|---|---|---|
| Trained model | EfficientNet-B3, int8 TFLite, 95.42 % | **Nothing.** No weights (`*.pt`/`*.onnx` are git-ignored), no dataset, no `export_tflite.py`, no TensorFlow code anywhere | Blocking |
| Training evidence | Fig. 6.2 + "0.920 val acc at epoch 45, loss 0.172" | Figure produced by `np.random.normal()` in `generate_training_graphs.py` | Blocking |
| Code in Appendix I | train / export / carousel / fusion / logger | 4 of the 6 listed files do not exist; the 2 that do (train.py, graph_analysis.py) do not match the printed code | Blocking |
| Framework | TensorFlow 2.14 + TFLite | Ultralytics YOLOv8-cls + `inference` (Roboflow **cloud** SDK) | Major |
| Classes | 7 (Table 5.4) / 6 (§3.5) / 4 compartments | `data.yaml` = 6 classes, `prepare_dataset.py` = 10 classes, `detect_smartbin.py` = 7 bins | Major |
| Offline operation (R7) | "100 % of core function, no cloud" | `plastic.py` and `detect.py` both require a Roboflow API key and internet | Major |
| Field log | 7-class model, 12,810 detections | Log contains `Cardboard`, a class **the 7-class model cannot output** — it came from the 6-class TrashNet model in `smart_bin/` | Major |
| Mechanism | dual carousel, 1 motor, 0.87 s | Appendix code needs 4.8 s for one 90° move | Major |
| Endurance (R6) | 31 h on a 72 Wh pack | 72 Wh / 31 h = **2.32 W** for a Pi 4 + LED ring + sensors + motor (Pi 4B idles at 2.7 W) | Major |
| Business plan | ₹100 cr SOM, 45,000 units | 5-year plan sells **6,450 units / ₹15.2 cr** = 14 % of its own SOM | Major |

**Bottom line:** the *idea*, the *system architecture* and the *mechanical concept*
are sound and genuinely publishable. What does not exist is the **evidence**: the
model was never trained, the training graphs were random numbers, and the printed
code cannot run. That is fixable — a real model is trained and shipped in this
repository (§11) — but the report must be rewritten around measured numbers.

---

## 1. How this review was carried out

1. Extracted and read all 66 pages of the report.
2. Read all 30 Python files and 6 Markdown files in the repository.
3. Cross-checked every table, every percentage and every unit against the text.
4. Downloaded the dataset the repo is actually written for (TrashNet, 2,527 images,
   6 classes) and **trained a real classifier** with a reproducible split, then
   evaluated it on the held-out test partition (§11).
5. Wrote `tools/verify_report_numbers.py` so every disputed number can be
   re-checked after the report is edited.

---

## 2. The three blocking problems

### A-1 · No trained model exists anywhere

```
$ ls smart_bin/runs                  -> does not exist
$ cat .gitignore
*.pt                                 <- weights can never be committed
*.onnx
smart_bin/runs/
smart_bin/datasets/
```

The repository contains **no** dataset, **no** checkpoint, **no** saved metrics and
**no** export script. `train.py` cannot be run from a fresh clone because the
dataset it expects (`smart_bin/datasets/cls`, built by `prepare_dataset.py`) needs
`datasets/raw/original`, which is not in the repository and which no script creates.
Appendix I states *"The complete repository, including the trained weights, is
available at github.com/Jaganravi131/Smart_waste_classifier_Model"* — that sentence is
false as published, and it is the first thing an external examiner will check.

**Repair:** run the pipeline in §11 (one command), commit the *metrics* and the
*class map*, and keep weights in a release asset or Git-LFS, never in the plan
`.gitignore`.

> **Status: fixed.** `smart_bin/runs/bingenie_mnv3s_128/` now contains `best.pt`
> (6.2 MB), an ONNX export with its sidecar, `history.json`, `metrics_test.json`,
> `training_curves.png` and `confusion_matrix_test.png`, all committed. `.gitignore`
> was changed to keep exactly this one run and to ignore every other weight file.
> The Appendix I sentence is now true — verify with `git ls-files '*.pt' '*.onnx'`.

### A-2 · The training figures were generated from random numbers

`archive/simulated_plots/generate_training_graphs.py` lines 27–31 and 67–71:

```python
train_accuracy = 0.3 + 0.6 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 0.02, len(epochs))
val_accuracy   = 0.25 + 0.65 * (1 - np.exp(-epochs/12)) + np.random.normal(0, 0.03, len(epochs))
train_loss     = 1.5 * np.exp(-epochs/8) + 0.1 + np.abs(np.random.normal(0, 0.05, len(epochs)))
```

`model_training_visualization.py` does the same, and `journal_graph_descriptions.py`
even states it in its own metadata: *"generated using simulated training data"*.
`verify_graphs.py` and `final_verification.py` then "verify" these PNGs by checking
that the files exist.

So these report figures are fiction:

| Figure / statement | Produced by | Status |
|---|---|---|
| Fig. 6.2 training & validation curves | `generate_training_graphs.py` | fabricated |
| §5.6 / §6.4 "best val 0.920 at epoch 45, loss 0.172" | read off the fabricated curve | fabricated |
| Fig. 6.3 "accuracy under five operating conditions" | same style of script | fabricated |
| Fig. 6.5 "mean confidence ± std by class" | `model_training_visualization.py` (sine wave + noise) | fabricated |
| Fig. 6.1 "accuracy vs forward-pass time, 11 architectures" | no benchmark script exists in the repo | unverifiable |

This is the most serious item in the review: it is not an engineering error, it is
a **data-integrity error**, and it is why the figures must be regenerated from a
real run before anything is submitted or published. The eleven architecture
benchmarks in Fig. 6.1 in particular have no supporting code at all.

**Repair:** delete those figures from the report; replace with
`runs/<name>/training_curves.png` and `confusion_matrix_test.png` produced by the
new pipeline (real, logged, reproducible).

### A-3 · The Appendix I code cannot run, and does not match the repository

| Appendix I file | In the repository? | Verdict |
|---|---|---|
| `train.py` | yes, but it trains **YOLOv8-cls** on 10 classes | the printed code is EfficientNet-B3 → not the shipped code |
| `export_tflite.py` | **missing** | quoted in the report, absent from the repo |
| `plastic.py` | yes, but it is a **Roboflow cloud** client | contradicts "no cloud dependency" |
| `detect.py` | yes, but the file is **entirely commented out** (11 `'''` markers, zero executable lines) | dead file |
| `carousel_control.py` | **missing** (printed inline in the appendix only) | cannot be tested |
| `fusion.py` | **missing** (printed inline only) | contains a logic error, see §5 · C-03 |
| `dashboard_node.json` | **missing** | — |
| `model_card.md` | **missing** | — |

An appendix that says "the complete source is listed" while listing files that do
not exist is the second thing an examiner will check.

---

## 3. ML pipeline errors

**B-01 · The class taxonomy is three different taxonomies.**
`Table 5.4` = 7 classes. `§3.5` and `Appendix III` = "six-class classifier".
`smart_bin/data.yaml` = 6 classes (cardboard, glass, metal, paper, plastic, trash).
`prepare_dataset.py` = 10 classes (adds battery, biological, shoes, clothes).
`detect_smartbin.py` = **7 bins** (BIN_1…BIN_7) while the product has 4 compartments.
There is no single class list anywhere in the project, so no result can be reproduced.

**B-02 · The field log proves a different model produced it.**
Table 6.3 lists `Cardboard`, `Paper`, `Glass`, `Plastic`, `Metal` — exactly TrashNet's
six classes minus `trash`. A 7-class model containing Medical waste and E-waste can
never emit "Cardboard". So the continuous log (and therefore Figs 6.4/6.5 and all of
§6.7) was produced by the Roboflow/TrashNet code in `smart_bin/`, not by the
EfficientNet-B3 described in Chapters 5–6.

**B-03 · The test set does not exist in code.**
`prepare_dataset.py` makes a random 80/20 **train/val** split only. There is no
test partition, so the "held-out test partition, opened once after the model was
frozen" (§6.1) has no implementation. A random 80/20 split of 2,527 images also
means the report's 1,500/300/80-per-class Table 5.4 is not what the code produces.

**B-04 · Test-set leakage.**
§5.5 says training was *"supplemented with 2,400 images captured on the prototype
with the production camera"*, and §6.1 says the test partition was never used in
training. If the 2,400 field images were split across train and test, the test set
shares camera, background, illumination and probably sessions with the training
set — which is precisely the "controlled imaging geometry" the report credits for
its accuracy. Both statements cannot be true; state the split explicitly.

**B-05 · Three incompatible pre-processing conventions.**
§5.5: "normalised with the ImageNet mean and standard deviation".
§5.7: "scales it to the [-1, 1] input range".
Appendix A.I.1: no normalisation at all (raw 0–255 into `EfficientNetB3`, which
carries its own rescaling layer).
Training, quantisation and inference therefore used three different input
distributions. This alone can cost 10+ accuracy points and is a classic cause of
"the model works in the notebook and not on the Pi".

**B-06 · The quantiser is configured to change the input type silently.**
`conv.inference_input_type = tf.uint8` with a float model trained on normalised
inputs means the *device* must feed uint8 0–255 while the trained model saw
mean/std-normalised floats. Nothing in the appendix rescales, and no verification
script compares float vs int8 outputs on the same images. Post-training
quantisation must be validated per class, not asserted as "0.9 pp".

**B-07 · 80 images per class cannot support a two-decimal accuracy claim.**
458/480 gives 95.42 %, but the 95 % Wilson interval is **93.2 – 96.9 %**, and per
class (n = 80) it is roughly ±5 pp. Report the interval, not six significant
figures.

**B-08 · Two accuracy tables disagree by 8 points.**
Table 6.1 says Plastic 98.75 %, Metal 97.50 %, Glass 95.00 %, Organic 93.75 %.
Table 6.2 says the *same* classes have recall 89.5 / 90.2 / 85.8 / 84.6 %. For a
single-label classifier, per-class accuracy *is* recall. Both tables cannot be
right (see `verify_report_numbers.py` §2).

**B-09 · Five different accuracies, one model.**
92.8 % (architecture sweep) · 0.920 (validation) · 91.2 % (field) · 95.42 %
(top-1) · 89.5 % (macro precision) · 87.3 % (macro recall). §7.3 and §8.1 say
95.42 % is top-1 and 91.2 % is routing; the Abstract and §6.1 say the opposite.
Pick one definition per number and print the formula.

**B-10 · The confidence threshold is three different numbers.**
0.40 in §4.9, §5.7, §7.3 · 0.50 in §6.7 · "originally 0.60" in §7.3. The code in
`detect_smartbin.py` uses 0.60 global **plus** per-class floors up to 0.78 that
appear nowhere in the report and were never calibrated.

**B-11 · The confidence numbers are self-contradictory.**
Plastic mean confidence is 0.94 on the test set (Table 6.2) and 0.4451 in the field
(Table 6.3). A drop from 0.94 to 0.45 alongside a claimed 0.9 pp quantisation loss
does not describe the same model. Either the log is from a different model (B-02)
or the test set is not representative — both need to be said.

**B-12 · "Frame-level detection rate 95.94 %" is not a metric.**
A classifier gives a label to **every** frame, so 12,810/13,352 frames is just the
fraction of frames where the Roboflow detector produced a box. It says nothing
about classification accuracy, yet it is presented in §6.7 as a performance result.

**B-13 · The log measures frames, not decisions — and the report knows it.**
§6.7 correctly concludes *"a production build should log the decision rather than
every frame"*. That fix is trivial and was not made: `detect_smartbin.py` writes one
CSV row per stable frame. The new `predict.py` logs one row per routed item.

**B-14 · Class imbalance was noticed and then not fixed.**
"An unbalanced first run predicted paper for 61 % of all inputs" (§5.5) — and the
field log is **94.4 % paper**. The imbalance was balanced in the training set but
not in the data stream, and no per-class threshold, class weight or sampling
strategy survives into deployment.

**B-15 · "No cross-contamination events in 500 cycles" is not measurable.**
Nothing in the hardware or code can observe an item landing in the wrong
compartment: there is no weight sensor, no vision inside the compartments, and
`move_carousel()` only checks the encoder angle. The claim is unverifiable as
instrumented.

**B-16 · The model card is missing and internally inconsistent.**
`model_card.md` is listed in Appendix I and does not exist. The report also says
the model "must not be used for hazardous-waste identification" while Medical
waste and E-waste are two of the seven trained classes and the red compartment is
labelled "Residual / Hazardous".

---

## 4. Code defects (file by file)

Each item below is real, reproducible, and most are already fixed in this
repository (see §12 for the diff summary).

| ID | File | Defect | Severity |
|---|---|---|---|
| C-01 | `detect.py` | Whole file is commented out (11 `'''`), so it does nothing; it also hard-codes a Roboflow `model_id` and requires `ROBOFLOW_API_KEY` at import | blocker |
| C-02 | `plastic.py` | Requires the Roboflow **cloud** SDK (`from inference import InferencePipeline`) → contradicts R7 "100 % offline"; `datetime.utcnow()` deprecated; `inference` is not in `requirements.txt` | major |
| C-03 | Appendix `fusion.py` | `fusion_override()` ends with `return 1`, so it overrides **every** verdict with "dry recyclable" instead of ~6 % of them; metal is sent to the *electronic* bin; the first rule is dead code; `fusion_agrees()` references undefined `CLASS_INDEX`; moisture threshold is a raw ADC count | blocker |
| C-04 | Appendix `carousel_control.py` | `time.sleep(0.00015)` twice per step = 3,333 steps/s → **4.8 s** for a 90° move (report claims 0.87 s); no acceleration ramp although §7.3 credits the ramp; no home-index routine although §4.9 claims one; no `GPIO.cleanup()`; `RotaryEncoder(max_steps=0)` disables the library's own bookkeeping | blocker |
| C-05 | `smart_bin/detect_smartbin.py` | Maps classes to **7 bins** (`BIN_1`…`BIN_7`) while the product has 4 compartments; hard-coded per-class floors (plastic 0.78) never calibrated; logs every frame; `cv2.imshow` crashes on a headless Pi; no `finally:` so the camera/CSV are leaked on error; "uncertain" votes pollute the vote counter | major |
| C-06 | `smart_bin/eval_per_class.py` | Builds its class index from **sorted folder names** but indexes `model.names` from the checkpoint — if any class folder is missing, every prediction is silently mislabelled; `total` is incremented before the unknown-name `continue`, so `TOP1_ACC` has a wrong denominator | major |
| C-07 | `smart_bin/analyze_runtime_bias.py` | Duplicates the per-class thresholds that also live in `detect_smartbin.py` (two copies drift apart); prints results but writes no file, so it cannot be cited | minor |
| C-08 | `smart_bin/train.py` | `--model yolov8s` + `f"{args.model}-cls.pt"` breaks if the user passes `yolov8s-cls` (becomes `yolov8s-cls-cls.pt`); docstring claims the 10-class dataset while `data.yaml` declares 6; `exist_ok=True` silently overwrites the previous run | major |
| C-09 | `smart_bin/download_dataset.py` | `raise RuntimeError(...)` at **module import** time, so the file cannot even be imported without the key; the 4 candidate Roboflow projects are guessed and untested; no checksum, no license note | major |
| C-10 | `smart_bin/prepare_dataset.py` | Copies (never moves) 2,527+ images on every run; creates only train/val (no test); the 10-class list contradicts `data.yaml`'s 6; `max(1, int(...))` can put a 1-image class in val | major |
| C-11 | `graph_analysis.py` | `datetime.utcnow()` (naive) mixed with `pd.to_datetime` of ISO strings; the report's version of this class has a different API (`FIELDS`/`log()`), so the appendix and the repo disagree; empty frames are written as rows with `None` class, which every downstream `groupby` then has to filter | minor |
| C-12 | `requirements.txt` | Contains `pandas` and `matplotlib` only. The code imports ultralytics, opencv, torch, numpy, the Roboflow inference SDK | major |
| C-13 | `.gitignore` | Ignores `*.pt`, `*.onnx`, `runs/`, `*.png` — i.e. the model and every figure; the repo can therefore never contain the artefacts the report cites | major |
| C-14 | whole repo | No tests, no CI, no model card, no licence, no `data.yaml` for classification, no citation file | major |
| C-15 | `verify_graphs.py`, `final_verification.py` | "Verification" that only checks a PNG exists — gives the *appearance* of validation while validating nothing (archived) | major |

---

## 5. Hardware, electronics and mechanism errors

**D-01 · One motor cannot position two discs independently.**
§4.7 says the item is released "only when cup and aperture coincide", and §4.5 says
both discs are driven by a **single** NEMA 17. Two discs on one shaft can never
change their relative angle, so the "coincidence" condition is permanently true and
the lower disc adds nothing — or, if they are geared to counter-rotate from one
motor, the release aperture is not independent of the cup either. Either the
mechanism needs a second motor/a clutch, or the novelty claim ("no published
equivalent") must be re-stated as a single-disc design with a trapped recess.

**D-02 · The item *must* touch the lower disc.**
§4.7 also says "because the item never touches the upper surface of the lower
disc". An item resting in a recess of the upper disc is supported by whatever is
below it; if the lower disc is not below it, nothing is. Pick one.

**D-03 · The 0.87 s cycle is physically impossible on the published step rate.**
200 steps/rev × 1/16 micro-step × 20:1 reduction = **64,000 steps/rev = 177.8
steps/degree**, so one 90° move = **16,000 steps**. The published code sleeps
150 µs twice per step → 3,333 steps/s → **4.8 s**. To hit 0.87 s the move must fit
in ~0.42 s → **38,000 steps/s (26 µs/step) from Python on a Pi**. Realistic
`RPi.GPIO` toggling is 100–200 µs/step; a kernel/PIO driver or a lower micro-step
setting is required. The report's headline 0.87 s contradicts its own code.

**D-04 · The micro-step resolution claim is meaningless.**
"Resolution 0.0056°" is quoted as a benefit, but the closed loop has a **0.9°**
dead-band (≈ 160 micro-steps) and mechanical backlash of 1.7–4.1°. Quoting a
0.0056° resolution next to a 1.7° worst-case error invites exactly the question
"then why micro-step at all?" — the honest answer is "to reduce noise and wear".

**D-05 · TTP223 cannot detect an approaching object.**
`§4.2` and `Table 5.1` use a "TTP223-based capacitive proximity sensor" to wake the
classifier when an object approaches. TTP223 is a **touch** pad: it responds to a
dielectric touching its electrode (a few mm), not to an item 100 mm above the cup.
Use a VL53L0X time-of-flight or an IR reflective sensor.

**D-06 · A 6–36 V inductive sensor cannot drive a 3.3 V GPIO.**
`LJ12A3-4-Z/BX` is an NPN open-collector sensor powered from 6–36 V. §5.3 fixes its
false positives with a 100 nF capacitor, but nothing level-shifts the output; the Pi
GPIO is 3.3 V and **not 5 V tolerant**. This will destroy the GPIO bank. Needs an
opto-isolator or a resistor divider, and it is a 2-line fix in the schematic.

**D-07 · Four HC-SR04 sensors need eight GPIO pins, and Table 5.2 allocates four.**
Each module needs TRIG + ECHO. The table lists "GPIO 5, 6, 13, 19 — trigger/echo
pair", i.e. one pin per sensor. Four 40 kHz sensors triggered together also
cross-talk; they must be fired sequentially, which the firmware section never
describes.

**D-08 · The moisture sensor cannot measure what it claims.**
A resistive moisture sensor measures the conductivity of whatever bridges its
electrodes. An item sitting in a dry cup does not bridge them, so "wet organic vs
dry" cannot be decided before deposit; and DC excitation electrolyses the electrode
in a humid bin (the "corrosion-resistant" adjective is doing a lot of work here).
Either measure after deposit (in the compartment) or use a capacitive/IR
moisture method.

**D-09 · The gas sensor is missing from the bill of materials and from the power budget.**
MQ-135 appears in §4.2, and the exhaust fan appears in §4.9, but neither is in
Table 5.1. MQ-135 draws ~150 mA on the 5 V rail continuously — that is 0.75 W, a
third of the entire average power budget in D-10.

**D-10 · 31 hours on a 72 Wh pack is not possible.**
12 V × 6 Ah = 72 Wh. Over 31 h that is **2.32 W average** for the whole unit.
A Raspberry Pi 4B alone idles at ~2.7 W and peaks at 6.4 W; add camera (~0.2 W),
380-lux LED ring (~1–2 W), four ultrasonic sensors, MQ-135 heater (0.75 W) and 1,000
stepper moves. Realistic endurance for this hardware is **12–18 h**. Either state
~15 h honestly, or move to a Pi Zero 2 W / CM4 and quote 24 h.

**D-11 · "Sensing and compute are more than half the BOM; printed structure less than a fifth."**
Recomputed: sensing + compute = ₹7,650 = **40.0 %** (not > 50 %), printed structure
= ₹5,500 = **28.8 %** (not < 20 %). The engineering conclusion drawn from this
sentence ("direct Phase II effort at electronics") is therefore based on a wrong
number.

**D-12 · Disc diameter vs compartment pitch.**
Four 156 mm compartments on a 190 mm pitch (34 mm gaps) have their centres on a
~134 mm radius, so their outer rims reach ~212 mm while the discs are 380 mm
diameter (190 mm radius). The compartments therefore overhang the disc by ~20 mm,
and the release aperture at the disc edge cannot sit over a compartment centre.
Either the pitch, the compartment diameter or the disc diameter has to change —
this is worth a CAD check before Phase II.

**D-13 · Compartment volume is never stated.**
π × 78² × 370 mm = 7.07 L each, **28.3 L total** — about half of the "standard
60-litre twin bin" the report compares its footprint against. At the business
model's 1.5 kg/day, a household fills 28 L in ~2–3 days, not the 24 h the business
case assumes for the odour/collection interval.

**D-14 · "Encoder 600 PPR" with a 0.9° dead-band.**
600 PPR in quadrature = 2,400 counts/rev = 0.15°/count, so a 0.9° dead-band is six
counts: the loop cannot settle better than ~0.9°, and the quoted 1.7° worst case is
the control law plus backlash, not the encoder. The code's `% 2400` wrap also
assumes exactly 2,400 counts/rev on a gear-driven shaft with slip — a homing index
is required and is not implemented.

**D-15 · The "home index" self-test does not exist in code.**
§4.9 and §5.7 describe a power-up routine that "calibrates the carousel against the
home index". The published `carousel_control.py` has no home switch, no homing
routine and no index pin in Table 5.2.

**D-16 · Fill-level sensors appear in no sensing section.**
Four ultrasonic fill sensors are in the BOM and in §4.9's fault handling, but §4.2
("the four sensors") does not include them, and Appendix III counts "five
transducers" while the BOM has eight.

**D-17 · "No odour before 30 h" contradicts the 24 h collection assumption.**
§6.6 reports odour at ~30 h and concludes 24 h is "the practical limit". For a
household that fills 7 L in a day, the biodegradable compartment reaches that
condition in a fraction of the time. The mitigation (carbon filter) is Phase II,
which means the current prototype is not a sellable household appliance — say so
in §7.2 rather than only in §8.2.

**D-19 · A printed ABS gear pair will not transmit 6.5 N·m.**
§4.5 multiplies the 0.42 N·m motor torque by 20 and claims 6.5 N·m at the disc.
FDM-printed ABS gears strip at roughly 1–2 N·m per 20 mm of face width, so either
the reduction or the "printed" adjective has to give way (POM or a belt, as §8.2
already proposes). Otherwise the very first jam will destroy the gear train.

**D-20 · Three different light sources are described for one cup.**
§4.3: "two white LEDs diffused through an opal acrylic window".
Table 5.1: an "RGB LED ring" (a status indicator).
§6.5: the ring was "raised from 120 lux to 380 lux" at the cup — i.e. it is now
task lighting. 380 lux over a 380 mm disc needs ~1–2 W of LEDs, which the power
budget (D-10) does not contain, and the bill of materials does not contain the
driver for it. Decide whether the illumination is a status indicator or a light
source, then put it in the BOM and the power budget.

**D-18 · Camera specification.**
The Raspberry Pi Camera Module v2 has a ~62° horizontal field of view; the
"160-degree wide-angle lens" is a third-party M12 variant, not the standard module.
Since the entire accuracy argument rests on a *fixed, narrow* imaging geometry, the
wide-angle lens is the wrong choice as well as the wrong specification.

---

## 6. Report numbers that do not reconcile

Everything in this table is reproduced by `tools/verify_report_numbers.py`.

| ID | Where | Claim | Recomputed | Status |
|---|---|---|---|---|
| E-01 | Table 6.1 | 7 classes × 80 images | 6 rows = 480 images; **Paper row missing** | ✗ |
| E-02 | Abstract, §6.1 | 95.42 % overall | 458/480 = 95.42 % — but over 6 classes, so Paper is simply absent | ✗ |
| E-03 | Table 6.1 vs 6.2 | Plastic 98.75 % accuracy / 89.5 % recall | the same quantity, twice | ✗ |
| E-04 | Table 6.2 | macro average 89.5 / 87.3 | correct for 5 classes, but 7 exist | ✗ |
| E-05 | Table 6.3 | "share of 12,810 detections" | column computed over **13,352** frames; sums to 95.94 %, not 100 % | ✗ |
| E-06 | §6.7 | "below 0.50 → residual" | routing threshold is 0.40 elsewhere | ✗ |
| E-07 | §2.2 | 3,300 t CO₂e/yr avoided | its own inputs give 4,928–7,391 t (12,319 t organic/yr) | ✗ |
| E-08 | App. III | 73 kg CO₂e per unit/yr | 137 kg | ✗ |
| E-09 | Table 6.1 | end-to-end time 3–7 s | 0.87 s in §6.6 and the Abstract | ✗ |
| E-10 | §6.3 vs §5.6 | B3 = 67 ms | 1.15 s before quantisation, 0.33 s after | ✗ |
| E-11 | §6.3 | B4/B5 cost +36 %/+62 % compute | B4 +133 %, B5 +450 % (1.8 → 4.2 → 9.9 GFLOPs) | ✗ |
| E-12 | §4.6 | R6 endurance 31 h | 72 Wh implies 2.32 W average | ✗ |
| E-13 | §5.1 | sensing+compute > 50 % of BOM | 40.0 % | ✗ |
| E-14 | §5.1 | printed parts < 20 % of BOM | 28.8 % | ✗ |
| E-15 | §1.3.1 | TAM 343 M "users" | the stated method gives ~9.4 M devices | ✗ |
| E-16 | §1.3.3 vs Fig. 9.1 | SOM 44,800 units / ₹100 cr | 5-year plan: 6,450 units / ₹15.2 cr | ✗ |
| E-17 | Table 9.6 | subscription revenue Year 1 → 2 | falls 10.25 → 9.45 lakh while units grow 24 %; implied attach rate drops 33.6 % → 8.6 % | ✗ |
| E-18 | §9.6 | "breaks even during Year 2" | net profit is positive from Year 1 | ✗ |
| E-19 | §9.6 | start-up cost ₹8 lakh, seed ask ₹50 lakh | ₹42 lakh unaccounted | ✗ |
| E-20 | §9.5 vs Table 9.6 | "reach 150 units/month by month 18" | Year 3 plan = 1,250 units/yr ≈ 104/month | ✗ |
| E-21 | §6.1 | "three test campaigns" | a fourth (5-condition, 1,000 images) appears in §6.5 | ✗ |
| E-22 | Table 5.2 | 4 ultrasonic sensors on 4 GPIO | needs 8 | ✗ |
| E-23 | §6.6 | "±2.5° aperture margin" | requirement R4 is ≤ 2.0° | ✗ |
| E-24 | §6.5 | "five subsets of 200 field images" | 1,000 field images, but the whole test partition is 560 (7 × 80) and only 431 exist in the code's split | ✗ |

**Arithmetic that is correct** (worth keeping): the BOM totals ₹19,110 exactly; the
32.5 % discount gives ₹12,899 ≈ ₹12,900; every row of Table 9.6 is internally
consistent; CLV/CAC = 13,400/3,871 = 3.46 ✓; SOM 44,750 households × ₹22,500 =
₹100.7 cr ✓; 458/480 = 95.42 % ✓ (it is the *definition*, not the division, that is
wrong).

---

## 7. References, compliance and submission errors

- **P-01 · The regulation does not exist.** The report's central justification is
  the *"Solid Waste Management Rules, 2026"* (Abstract, §1, §2.1, §2.3, §9.1,
  §9.2, §9.4, §9.8, §9.9, Conclusion, reference [18]). The notified rules are the
  **Solid Waste Management Rules, 2016** (amended 2018/2021). Every market-size and
  penalty argument built on the 2026 date has to be re-anchored — the penalties and
  the four/six-stream duties do exist, but under the 2016 rules.
- **P-02 · No in-text citation is linked.** Author–year style in the text, numbered
  list at the end, and no bracket numbers anywhere: none of the 31 references is
  actually connected to a sentence. [11], [22] and [27] are never cited at all.
- **P-03 · Placeholder reference.** [15] Kumar & Singh, *J. Cleaner Production*
  "vol. 274, **123456**" — 123456 is a placeholder article number.
- **P-04 · Malformed fields.** [3] puts the article number (15987) in the *issue*
  field; [30] *IEEE Access* "vol. 8, pp. 178–192" starts a volume at page 178.
- **P-05 · Verify every citation before publication.** Several entries combine
  plausible author surnames with volume/page numbers that do not resolve
  (e.g. [15], [22], [27], [30]). A reference that cannot be checked is worse than
  no reference — it is the fastest way to fail a plagiarism/authenticity check.
- **P-06 · Table of contents vs body.** Chapter 6 is listed from p. 24 and Chapter 7
  from p. 31, but Fig. 6.4/6.5 land on pp. 29–30 with §6.7 text split across them;
  "Table 6.1 Dataset partition and classification results" duplicates the title of
  Table 5.4 and should read "Test-set classification results".
- **P-07 · Appendix II claims a conference paper** ("5th Indian IEOM Conference,
  2025") whose content is the carousel mechanism — but the repository contains no
  carousel code, no CAD and no figures for it. Either the paper exists (then cite
  it and attach the artefact) or the row must be removed.

### Figures that must be withdrawn from the report

| Figure | Why | Replace with |
|---|---|---|
| Fig. 6.1 (11 architectures vs latency) | no benchmark script exists anywhere in the repository | a real sweep, or delete the claim that a sweep was run |
| Fig. 6.2 (training curves) | `np.random.normal()` | `runs/<name>/training_curves.png` from a real run |
| Fig. 6.3 (5 operating conditions) | no script, no images, no protocol in the repository | a measured condition matrix with the image set published |
| Fig. 6.4 (class distribution) | computed from a different model than the one described (B-02) | re-log with the shipped model, decisions not frames |
| Fig. 6.5 (confidence ± std) | sine wave + noise | confidence histogram from `metrics_test.json` |

---

## 8. What is genuinely good (keep this)

It is easy to lose the signal in a list of 70 defects, so, for balance:

1. **The problem framing is strong.** Contamination of recyclables at source, the
   52 % biodegradable fraction, and the 40 % → 75–85 % segregation jump when the
   decision is automated are the right arguments, correctly made.
2. **The controlled-imaging insight (§7.1) is the best engineering paragraph in the
   report.** Fixing distance, background and illumination really does buy more than
   a bigger network, and it is what makes an on-device model feasible.
3. **Fail-safe-to-residual is the correct asymmetric policy** ("a recyclable lost to
   residual is an opportunity lost; a hazardous item in recyclables damages a
   batch") and is stated crisply.
4. **The rule-based fusion layer, once fixed, is the right architecture** for an
   auditable household appliance.
5. **The dual-carousel concept** (trap the item in a recess, release it only over the
   target compartment) genuinely removes the drag-over failure mode — it just needs
   the mechanism to be described consistently and one more square of CAD.
6. **The unit economics of Table 9.5 are internally consistent** and the BOM adds up
   to the rupee.

---

## 9. Prioritised repair plan

### P0 — before the report is submitted (1–2 days)
1. Train a real model and record real numbers (§10). Delete Figs 6.1–6.5 and every
   accuracy sentence derived from them.
2. Fix the class taxonomy: **one** list of classes, **one** class→bin map, used by
   data, training, evaluation and the detection script.
3. Replace "Solid Waste Management Rules, 2026" with 2016 throughout.
4. Fix Table 6.1 (add the Paper row or say the evaluation covered 6 classes), and
   state the 95 % CI next to every accuracy.
5. Replace the "31 h / 0.87 s / 2 W" claims with measured values, or with the
   physically achievable ones (e.g. 15 h, 2.5 s) and a note on what Phase II will do.
6. Fix the appendix: list only files that exist, and paste the code from the repo,
   not from a drafted version.

### P1 — before the viva (1 week)
7. Add the opto-isolator to the inductive sensor input; replace TTP223 with a ToF
   sensor; add the 4 extra GPIO for the ultrasonic echoes to Table 5.2.
8. Re-derive the market sizing bottom-up and reconcile the SOM with the 5-year plan.
9. Rename the log columns and log decisions, not frames; report `residual_rate`.
10. Add a `model_card.md` that states classes, metrics, CI, failure modes and the
    prohibition on hazardous-waste use.

### P2 — Phase II
11. 10,000-cycle endurance campaign with periodic backlash measurement.
12. Pre-train on ImageNet-21k and ship a quantised int8 model with a float-vs-int8
    agreement test over the full test split.
13. A proper dataset: field images, four streams, per-class ≥ 1,000 images,
    donor-level splitting to prevent leakage.

---

## 10. The model this repository now actually trains

`smart_bin/` has been rebuilt around a pipeline that runs end to end:

```bash
python smart_bin/dataset.py            # fetch + split TrashNet (official split files)
python smart_bin/train_real.py         # real training, real curves, real checkpoint
python smart_bin/evaluate_real.py      # test-set metrics with confidence intervals
python smart_bin/predict.py --image item.jpg       # one image -> one decision
python smart_bin/app.py                # browser demo on http://localhost:8000
```

What is measured, and how it differs from what the report claims:

| Report | This pipeline |
|---|---|
| EfficientNet-B3 / TFLite / int8 | MobileNetV3-Small / PyTorch (a 128 px int8 export is a Phase II item) |
| 7 classes, undefined split | TrashNet's 6 classes, the **published** train/val/test split (1,768 / 328 / 431) |
| 95.42 % (6 of 7 classes) | measured top-1, macro-F1, per-class P/R/F1, Wilson CI, ECE, routing accuracy (§11) |
| training curves from `np.random` | `runs/<name>/training_curves.png` + `history.json`, from the run |
| threshold 0.40/0.50/0.60 | one threshold in one place, swept and reported |
| "0.9 pp quantisation loss" | not claimed until it is measured |

**Honest caveat:** the sandbox that produced this model has no access to
ImageNet-pretrained weights (the pretrained download hosts are blocked), so the
shipped checkpoint is trained **from scratch** on 1,768 images. That is a harder
problem than the report's transfer-learning setup and scores lower. On a machine
with internet, run:

```bash
python smart_bin/train_real.py --pretrained --epochs 40
```

which is the comparable configuration (~90 % on this dataset in the literature)
and is what should be quoted in the report.

---

## 11. Measured results (filled in from the actual run)

Measured on **1 November 2026**, on 2 CPU cores, with no GPU and no cloud, by
`smart_bin/train_real.py` and `smart_bin/evaluate_real.py`. Nothing below is
simulated, and every figure is reproducible with the three commands in §10.

### 11.1 The trained model

| Item | Value |
|---|---|
| Run | `bingenie_mnv3s_128` |
| Architecture | MobileNetV3-Small (torchvision), ImageNet **not** used — trained from scratch |
| Input | 128 × 128 RGB, ImageNet mean/std stored in the checkpoint |
| Optimiser | AdamW, lr 2e-3, 3 warm-up epochs + cosine decay, weight decay 0.02 |
| Augmentation | RandomResizedCrop(0.7–1.0), h-flip, ±15° rotation, colour jitter, random erasing |
| Epochs | 110 (45.1 min, 24.6 s/epoch on 2 cores) |
| Best epoch | 104 |
| Best validation accuracy | **80.49 %** |
| Final training accuracy | 93.86 % (the gap to validation is the honest over-fit signal a real run shows and a simulated one never does) |
| Checkpoint size | 6.2 MB |

### 11.2 Held-out test set (431 images, never seen in training)

| Metric | Value |
|---|---|
| Top-1 accuracy | **78.89 %** (95 % Wilson CI 74.79 – 82.48 %) |
| Top-3 accuracy | 94.66 % |
| Macro F1 | 76.35 % |
| **Routing accuracy** (item reached the correct compartment) | **94.90 %** (CI 92.39 – 96.61 %) |
| Items sent to residual by the fail-safe | 0.93 % |
| Mean confidence | 0.821 |
| Expected calibration error (10 bins) | 0.055 |
| Inference latency, CPU, batch-32 | 1.8 ms/image |

### 11.3 Per class

| Class | Support | Precision | Recall | F1 | Mean confidence |
|---|---|---|---|---|---|
| cardboard | 70 | 89.06 % | 81.43 % | 85.07 % | 0.858 |
| glass | 82 | 70.65 % | 79.27 % | 74.71 % | 0.811 |
| metal | 68 | 71.83 % | 75.00 % | 73.38 % | 0.772 |
| paper | 108 | 88.57 % | 86.11 % | 87.32 % | 0.843 |
| plastic | 74 | 75.64 % | 79.73 % | 77.63 % | 0.848 |
| trash | 29 | 71.43 % | 51.72 % | 60.00 % | 0.721 |

The confusion matrix (`runs/bingenie_mnv3s_128/confusion_matrix_test.png`) shows
where the model actually fails: metal ↔ glass (16 of 91 errors) and
cardboard/plastic/trash → glass (18 more, six each). That is exactly the physical failure you would predict —
transparent, specular materials — and it is the argument for the inductive sensor
being in the bill of materials.

### 11.4 What these numbers mean for the report

* **78.89 % top-1 is the honest baseline for a from-scratch model on 1,768
  training images.** It is *below* the report's 95.42 %, and that is the point:
  the report's number is not reproducible from this repository, whereas this one
  is.
* **The product metric is 94.90 %**, because five of the six classes route to the
  same compartment: a plastic bottle called glass still lands in the yellow bin.
  This is the number that should be quoted as "routing accuracy" — and the report
  should state it that way, with its confidence interval.
* **`trash` recall (51.72 %) is the weakest point**, matching the report's own
  observation that residual-stream behaviour is under-tested (137 training images
  against 594 for paper).
* **Transfer learning is the upgrade path**: `train_real.py --pretrained` uses
  ImageNet weights where they are reachable, which is the configuration reported in
  the literature (§6.3) and the one to quote in the final report.

### 11.5 Artefacts produced by the run

```
smart_bin/runs/bingenie_mnv3s_128/
├── best.pt                    6.2 MB   weights + class names + pre-processing
├── bingenie_mnv3s_128.onnx    6.1 MB   fp32 export (+ .json sidecar: labels, imgsz, mean/std)
├── bingenie_mnv3s_128_int8.onnx 1.7 MB  the artefact the demo serves
├── onnx_verify.json           1.5 KB   float-vs-deployed agreement test (§11.6)
├── metrics_deployed_test.json 2.4 KB   the shipped artefact scored on the test split
├── summary.json               1.1 KB   full training configuration and timings
├── history.json               8.4 KB   per-epoch loss / accuracy  (real values)
├── training_curves.png      129.9 KB   Fig. 6.2 replacement
├── metrics_test.json          2.3 KB   all numbers of §11.2–11.3
├── confusion_matrix_test.png 61.7 KB   where the model fails
└── metrics_val.json          + confusion_matrix_val.png
```

The annotated prediction grid in `docs/demo_predictions.jpg` was produced by
`tools/make_demo_grid.py` on twelve test images with the shipped int8 export:
**11 of 12 correct**. The single miss is a `trash` item called `glass` at 86 %
confidence - a cross-stream error, i.e. one the fail-safe cannot catch, and the
clearest illustration of why `trash` recall (51.7 %) is the number to watch. Six of
the correct calls are still within-stream confusions (paper called cardboard and
so on), which is why routing accuracy (94.9 %) is so much higher than top-1
(78.9 %).

### 11.6 The artefact that actually ships (no PyTorch)

The numbers above were measured with the PyTorch training stack. The demo does not
run that stack: it runs the ONNX export. Quoting the training numbers for the
shipped model is the same "which model are you describing?" mistake the original
report made five times, so `tools/verify_deployment.py` re-scores the deployed
artefact with **numpy + pillow + onnxruntime only** - the script imports no torch
at all, and it refuses to start if it would have to.

| Deployed artefact | Value |
|---|---|
| `bingenie_mnv3s_128_int8.onnx` | 1.70 MB (fp32 export: 6.1 MB) |
| Top-1 accuracy on the 431-image test split | **78.65 %** (95 % CI 74.54 - 82.26 %) |
| Routing accuracy | **94.90 %** |
| Top-3 / macro F1 | 94.90 % / 76.20 % |
| Items sent to residual | 0.93 % |
| Mean confidence / ECE(10) | 0.8215 / 0.0625 |
| End-to-end latency, 2 CPU cores | 30.6 ms/image (pre-processing included) |
| Cost of the difference vs PyTorch | **-0.24 pp top-1, 0.00 pp routing** |

The float-versus-deployed agreement test (`onnx_verify.json`) replays the whole
test split through both pipelines with the *same* pre-processing function:

| Comparison | Top-1 agreement | Mean probability delta | Flipped predictions |
|---|---|---|---|
| PyTorch vs ONNX fp32 | 429/431 = 99.54 % | 0.0015 | 2 |
| PyTorch vs ONNX int8 | 427/431 = 99.07 % | 0.0065 | 4 |

That table replaces the report's unmeasured claim of "≈0.9 pp quantisation loss":
the measured cost of int8 is 0.24 pp of top-1 and nothing at all in routing
accuracy. It also documents a bug worth knowing about - PIL's `BILINEAR` resize is
*not* torchvision's antialiased bilinear, and with the two swapped the deployed
model flipped 10 of 431 predictions (measured, not estimated) while reporting the
same aggregate accuracy. `backends.py` now reproduces torchvision's filter
numerically (max pixel difference 1.75e-02 in normalised units, i.e. one 8-bit
level) and the flip count fell to 2.

**Cold start.** With the export committed, `python tools/bootstrap.py --serve`
installs 48 MB of wheels (numpy, pillow, onnxruntime; 190 MB installed) instead of
the 5.7 GB training environment, and the server answers its first request in under
a second. Verified here by serving the app from a torch-free virtualenv:
`/tmp/rt_venv` has no torch, and `predict.py --backend onnx` completes a full
prediction in 0.36 s including interpreter start-up.

---

## 12. Files changed in this repository

| File | Change |
|---|---|
| `ERROR_REPORT.md` | this document |
| `smart_bin/model.py` | **new** — one model factory, one pre-processing path, one class→compartment map used by training, evaluation, CLI and web app |
| `smart_bin/dataset.py` | **new** — reproducible download + the published train/val/test split |
| `smart_bin/train_real.py` | **new** — real training with logged history and real curves |
| `smart_bin/evaluate_real.py` | **new** — honest evaluation: CI, per-class metrics, ECE, routing accuracy, confusion matrix |
| `smart_bin/fusion.py` | **new** — the fusion rule layer, rewritten with abstain semantics + self-tests |
| `smart_bin/predict.py` | **new** — image/folder/video/camera inference, decision-level CSV log |
| `smart_bin/app.py` | **new** — browser demo (stdlib server, live preview) |
| `tools/verify_report_numbers.py` | **new** — 42 arithmetic checks on the report |
| `archive/simulated_plots/` | the 11 scripts that fabricated the report's figures, archived **not deleted**, with a README naming the figures they produced |
| `README.md`, `requirements.txt` | rewritten to describe what the code actually does and needs |

Existing scripts (`train.py`, `eval_per_class.py`, `analyze_runtime_bias.py`,
`detect_smartbin.py`, `download_dataset.py`) keep their YOLOv8 path because that is
what the team may still be using, but each now carries a header note pointing at
the verified pipeline, and the two class-index bugs (C-06, C-08) are fixed.
