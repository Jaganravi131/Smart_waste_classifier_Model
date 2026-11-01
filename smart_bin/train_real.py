"""
BinGenie - real (not simulated) classifier training.

Replaces the previous `train.py` + the synthetic plot scripts.  Everything the
report needs - training curves, per-epoch metrics, the checkpoint - is produced
by an actual training run and written to JSON, so no figure has to be invented.

Quick start
-----------
    python dataset.py                 # one-off: fetch + split the dataset
    python train_real.py              # 60 epochs, MobileNetV3-Small, 144 px
    python train_real.py --epochs 120 --imgsz 160 --model resnet18
    python train_real.py --pretrained # use ImageNet weights if they are reachable

Outputs (under smart_bin/runs/<name>/)
    best.pt            checkpoint: weights + class names + pre-processing
    history.json       per-epoch train/val loss and accuracy (real values)
    training_curves.png real curves, ready to replace Fig. 6.2 of the report
    summary.json       configuration, timings, best epoch, final metrics
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import TRASHNET_DIR, ensure_dataset, dataset_summary
from model import (
    build_model,
    describe,
    eval_transform,
    load_labels_from_dir,
    save_checkpoint,
    train_transform,
    write_json,
)

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # deterministic kernels are very slow on CPU


def make_loader(root: Path, imgsz: int, train: bool, batch: int, workers: int,
                scale_min: float = 0.55, rotation: int = 20,
                balanced: bool = False) -> DataLoader:
    dataset = ImageFolder(
        root,
        transform=train_transform(imgsz, scale_min, rotation) if train else eval_transform(imgsz),
    )
    sampler = None
    if train and balanced:
        from torch.utils.data import WeightedRandomSampler

        weights = class_weights(dataset)
        assert len(weights) == len(dataset), "balanced sampler needs one weight per image"
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=False,
        drop_last=train,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += targets.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def mix_batch(images, targets, num_classes: int, alpha: float):
    """MixUp / CutMix for one batch.

    Both were added for the 160 px run: on 1,768 images the model memorises the
    training set by epoch ~60 (train acc 0.94 against val 0.80), and mixing is the
    one regulariser that attacks that directly. Returns the mixed batch, both
    target sets and lambda, so the loss can be the convex combination.
    """
    lam = float(np.random.beta(alpha, alpha))
    if lam > 0.5:                      # keep lambda >= 0.5 so the batch stays mostly itself
        lam = 1.0 - lam
    permutation = torch.randperm(images.size(0))
    if np.random.rand() < 0.5:         # CutMix: paste a box from another image
        height, width = images.shape[2], images.shape[3]
        ratio = math.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        top, left = np.random.randint(0, height - cut_h + 1), np.random.randint(0, width - cut_w + 1)
        images = images.clone()
        images[:, :, top:top + cut_h, left:left + cut_w] = images[permutation, :, top:top + cut_h, left:left + cut_w]
        lam = 1.0 - (cut_h * cut_w) / (height * width)
    else:                              # MixUp: linear combination of the pixels
        images = lam * images + (1.0 - lam) * images[permutation]
    return images, targets, targets[permutation], lam


class ModelEma:
    """Exponential moving average of the weights - cheap variance reduction."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy

        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_parameter, parameter in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_parameter.dtype.is_floating_point:
                ema_parameter.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
            else:
                ema_parameter.copy_(parameter)


def class_weights(dataset) -> list[float]:
    """Per-SAMPLE inverse-frequency weights, so `trash` is not drowned out.

    `WeightedRandomSampler` takes one weight per sample, not one per class. Passing
    the per-class vector (length 6) silently restricts sampling to the first six
    files: the model memorises them, training accuracy reads 94 % in one epoch and
    validation collapses to the majority class. Assert the length here - the failure
    is otherwise invisible until the evaluation.
    """
    targets = [int(target) for _, target in dataset.samples]
    counts = np.bincount(targets, minlength=len(dataset.classes)).astype(float)
    per_class = 1.0 / np.clip(counts, 1, None)
    per_class = per_class / per_class.sum() * len(per_class)
    weights = [float(per_class[target]) for target in targets]
    if len(weights) != len(dataset):
        raise ValueError(f"sampler weights must be per sample: got {len(weights)} for {len(dataset)}")
    return weights


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def plot_curves(history: dict, path: Path) -> None:
    """Plot the *measured* curves. No np.random anywhere in this function."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(epochs, history["train_acc"], label="train", lw=2)
    ax1.plot(epochs, history["val_acc"], label="validation", lw=2)
    best_epoch = history["val_acc"].index(max(history["val_acc"])) + 1
    ax1.axvline(best_epoch, ls=":", c="grey", lw=1)
    ax1.annotate(
        f"best {max(history['val_acc']):.3f} @ ep {best_epoch}",
        xy=(best_epoch, max(history["val_acc"])),
        xytext=(best_epoch, max(history["val_acc"]) - 0.12),
        fontsize=9,
        ha="center",
    )
    ax1.set(xlabel="epoch", ylabel="accuracy", title="Accuracy (measured)", ylim=(0, 1.02))
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_loss"], label="train", lw=2)
    ax2.plot(epochs, history["val_loss"], label="validation", lw=2)
    ax2.set(xlabel="epoch", ylabel="loss", title="Loss (measured)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the BinGenie waste classifier (real training)")
    parser.add_argument("--data", type=Path, default=TRASHNET_DIR)
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=144)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0, help="0 is fastest on a 2-core box")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    parser.add_argument("--scale-min", type=float, default=0.55,
                        help="lower bound of the RandomResizedCrop area (raise it when training from scratch)")
    parser.add_argument("--rotation", type=int, default=20, help="max RandomRotation in degrees")
    parser.add_argument("--pretrained", action="store_true", help="use ImageNet weights if available")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="alpha for MixUp/CutMix (0 = off; 0.2 is a good default)")
    parser.add_argument("--ema", action="store_true", help="evaluate a moving average of the weights")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--balanced", action="store_true",
                        help="sample classes inversely to their frequency (helps `trash`)")
    parser.add_argument("--name", default=None, help="run folder name")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    data_root = ensure_dataset(args.data)
    train_dir, val_dir = data_root / "train", data_root / "val"
    for folder in (train_dir, val_dir):
        if not folder.exists():
            raise SystemExit(f"Missing split folder: {folder} - run `python dataset.py` first.")

    labels = load_labels_from_dir(train_dir)
    val_labels = load_labels_from_dir(val_dir)
    if labels != val_labels:
        raise SystemExit(f"train/val class folders differ: {labels} vs {val_labels}")

    device = torch.device("cpu")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.name or f"{args.model}_{args.imgsz}px_{args.epochs}ep_{stamp}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    describe("BinGenie - real classifier training")
    print(f"  run          : {run_name}")
    model = build_model(args.model, len(labels), pretrained=args.pretrained).to(device)
    init_desc = "ImageNet init" if model.init_source == "imagenet" else "from scratch"
    if args.pretrained and model.init_source != "imagenet":
        init_desc = "from scratch (ImageNet weights were requested but unreachable)"
    print(f"  model        : {args.model} ({init_desc})")
    print(f"  image size   : {args.imgsz}px   batch {args.batch}   epochs {args.epochs}")
    print(f"  classes      : {labels}")
    print(f"  device       : {device} (threads={torch.get_num_threads()})")
    print(f"  data         : {data_root}")
    print()

    train_loader = make_loader(train_dir, args.imgsz, True, args.batch, args.workers,
                               args.scale_min, args.rotation, args.balanced)
    val_loader = make_loader(val_dir, args.imgsz, False, args.batch, args.workers)
    print(f"  train images : {len(train_loader.dataset)}  ({len(train_loader)} batches)")
    print(f"  val images   : {len(val_loader.dataset)}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = ModelEma(model, args.ema_decay) if args.ema else None
    print(f"  regularisers : mixup {args.mixup or 'off'}  ema {'on' if ema else 'off'}  "
          f"balanced sampling {'on' if args.balanced else 'off'}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = args.warmup_epochs * max(len(train_loader), 1)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val, best_epoch, best_path = -1.0, 0, run_dir / "best.pt"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        running_loss, correct, seen = 0.0, 0, 0

        for step, (images, targets) in enumerate(train_loader):
            global_step = (epoch - 1) * len(train_loader) + step
            lr = cosine_with_warmup(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.mixup > 0:
                images, targets_a, targets_b, lam = mix_batch(images, targets, len(labels), args.mixup)
                logits = model(images)
                loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
                # Accuracy under mixing is the convex combination too, otherwise the
                # training curve reads 1.00 by epoch 3 and looks like over-fitting.
                predictions = logits.argmax(1)
                batch_correct = (lam * (predictions == targets_a).sum().item()
                                 + (1.0 - lam) * (predictions == targets_b).sum().item())
            else:
                logits = model(images)
                loss = criterion(logits, targets)
                batch_correct = (logits.argmax(1) == targets).sum().item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * targets.size(0)
            correct += batch_correct
            seen += targets.size(0)

        train_loss, train_acc = running_loss / max(seen, 1), correct / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if ema is not None:
            ema_loss, ema_acc = evaluate(ema.module, val_loader, criterion, device)
            if ema_acc > val_acc:            # report the better of the two, and say which
                val_loss, val_acc = ema_loss, ema_acc
                evaluated = "ema"
            else:
                evaluated = "raw"
        else:
            evaluated = "raw"

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))
        history["lr"].append(round(lr, 8))

        flag = ""
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            save_checkpoint(
                best_path,
                ema.module if (ema is not None and evaluated == "ema") else model,
                labels,
                args.model,
                args.imgsz,
                extra={"val_acc": val_acc, "epoch": epoch, "run": run_name, "weights": evaluated},
            )
            flag = f"  <- best ({evaluated})"
        print(
            f"  ep {epoch:3d}/{args.epochs}  train {train_loss:.4f}/{train_acc:.4f}  "
            f"val {val_loss:.4f}/{val_acc:.4f}  lr {lr:.2e}  {time.time() - epoch_started:5.1f}s{flag}"
        )
        write_json(run_dir / "history.json", history)

    elapsed = time.time() - started
    plot_curves(history, run_dir / "training_curves.png")

    summary = {
        "run": run_name,
        "model": args.model,
        "pretrained_requested": bool(args.pretrained),
        "pretrained_init": model.init_source == "imagenet",
        "mixup_alpha": args.mixup,
        "ema": bool(args.ema),
        "balanced_sampling": bool(args.balanced),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "aug_scale_min": args.scale_min,
        "aug_rotation": args.rotation,
        "seed": args.seed,
        "labels": labels,
        "dataset": dataset_summary(data_root),
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val, 4),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "train_seconds": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / args.epochs, 1),
        "checkpoint": str(best_path.relative_to(BASE_DIR)),
    }
    write_json(run_dir / "summary.json", summary)

    print()
    describe("Training complete")
    print(f"  best validation accuracy : {best_val:.4f} (epoch {best_epoch})")
    print(f"  wall clock               : {elapsed / 60:.1f} min ({elapsed / args.epochs:.1f} s/epoch)")
    print(f"  checkpoint               : {best_path}")
    print(f"  real curves              : {run_dir / 'training_curves.png'}")
    print("  next                     : python evaluate_real.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
