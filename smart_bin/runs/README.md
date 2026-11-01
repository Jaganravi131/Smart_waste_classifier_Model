# Runs

Every folder here is a real training run written by `train_real.py`. Nothing in
this directory is hand-edited; the numbers in the report come from these files.

| Folder | What it is | Weights in git? |
|---|---|---|
| `bingenie_mnv3s_128/` | the shipped model (MobileNetV3-Small, 128 px, 110 epochs) plus its fp32 and int8 ONNX exports | **yes** — this is the deployable artefact |
| `bingenie_mnv3s_160/` | ablation: 160 px + MixUp + EMA, 90 epochs (`MODEL_LANDSCAPE.md` §7.1) | no (`*.pt` is ignored) |
| `arch30_mobilenet_v3_small/`, `arch30_resnet18/`, `arch30_efficientnet_b0/` | equal-budget architecture comparison, 30 epochs each (`MODEL_LANDSCAPE.md` §7.2) | no |
| `bingenie_mnv3s_128_balanced/` | ablation: class-balanced sampling, 110 epochs — did not fix `trash` (`MODEL_LANDSCAPE.md` §7.3) | no |
| `arch_sweep.json` | per-model validation-tuned threshold and test scores for all runs above | n/a |
| `calibration.json`, `calibration.png` | reliability diagram and risk-coverage curve for the shipped int8 model | n/a |
| `robustness_test.json`, `robustness_guard_test.json` | the §6.5 degradation study, with and without the guard | n/a |
| `architecture_compare.png` | validation curves for the three backbones, by epoch and by CPU-minute | n/a |

Checkpoints are excluded except for the shipped model: ResNet-18 alone is 43 MB,
which is not something to version. The evidence that matters — `history.json`,
`summary.json`, `metrics_*.json` and the plots — is committed, and every number can
be regenerated from the commands below.

```bash
# the shipped model
python smart_bin/train_real.py --imgsz 128 --epochs 110 --lr 2e-3 --weight-decay 0.02 \
    --scale-min 0.7 --rotation 15 --name bingenie_mnv3s_128

# the 160 px ablation
python smart_bin/train_real.py --imgsz 160 --epochs 90 --lr 2e-3 --weight-decay 0.02 \
    --scale-min 0.7 --rotation 15 --mixup 0.2 --ema --name bingenie_mnv3s_160

# the equal-budget architecture comparison (about 80 minutes on 2 cores)
for m in mobilenet_v3_small resnet18 efficientnet_b0; do
  python smart_bin/train_real.py --model "$m" --epochs 30 --imgsz 128 --lr 2e-3 \
      --weight-decay 0.02 --scale-min 0.7 --rotation 15 --seed 42 --name "arch30_$m"
done

# the class-balance ablation
python smart_bin/train_real.py --imgsz 128 --epochs 110 --lr 2e-3 --weight-decay 0.02 \
    --scale-min 0.7 --rotation 15 --seed 42 --balanced --name bingenie_mnv3s_128_balanced

# score them, tuning each model's routing threshold on validation
python tools/arch_sweep.py arch30_mobilenet_v3_small arch30_resnet18 arch30_efficientnet_b0
python tools/compare_runs.py --baseline arch30_mobilenet_v3_small --candidate arch30_resnet18
python tools/plot_arch_compare.py arch30_mobilenet_v3_small arch30_resnet18 arch30_efficientnet_b0
```
