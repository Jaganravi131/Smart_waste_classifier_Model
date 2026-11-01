"""
BinGenie - model factory and shared inference helpers.

This module is the single source of truth for:
  * which architectures the project supports,
  * how an image is pre-processed for inference,
  * how a class name is mapped to one of the four physical compartments.

It is imported by train_real.py, evaluate_real.py, predict.py and app.py so
that training and deployment cannot drift apart (a real bug in the previous
code base, where train.py used YOLOv8-cls while the report documented
EfficientNet-B3).

Nothing in this module downloads anything.  Pretrained ImageNet weights are
used only if they are already reachable; if not, the model is trained from
scratch and the caller is told so explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

def _torch_stack():
    """Import the training stack only when a function actually needs it.

    This module also owns `MEAN`/`STD` and re-exports the routing tables, and the
    ONNX inference path imports it.  A module-level `import torch` therefore made
    "inference without PyTorch" impossible: the app died with `ModuleNotFoundError:
    torch` even though it never touches a tensor.  Keeping the import local is what
    lets a deployment install 48 MB of wheels instead of 5.7 GB.
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    return torch, nn, models, transforms

# ---------------------------------------------------------------------------
# Reproducible image statistics (ImageNet).  Using fixed constants instead of
# downloading them keeps inference identical on the Raspberry Pi, where the
# device is usually offline.
# ---------------------------------------------------------------------------
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SUPPORTED_MODELS = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "shufflenet_v2_x1_0",
    "resnet18",
    "efficientnet_b0",
)


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Create a classifier with `num_classes` outputs.

    `pretrained=True` asks torchvision to load ImageNet weights.  If the
    weights are not available (offline machine, blocked download) we fall back
    to random initialisation and tell the user, rather than crashing or - worse
    - silently training a randomly-initialised "pretrained" model.
    """
    torch, nn, models, transforms = _torch_stack()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}")

    weights = "DEFAULT" if pretrained else None
    try:
        net = getattr(models, name)(weights=weights)
    except Exception as exc:  # pragma: no cover - depends on network
        print(f"[model] WARNING: could not load ImageNet weights ({exc.__class__.__name__}).")
        print("[model] WARNING: falling back to training from scratch.")
        net = getattr(models, name)(weights=None)

    # Replace the classifier head of each family with a fresh one.
    if name.startswith("mobilenet_v3"):
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name.startswith("efficientnet"):
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "shufflenet_v2_x1_0":
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet18":
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover
        raise ValueError(f"No head-replacement rule for {name}")
    return net


def train_transform(imgsz: int, scale_min: float = 0.55, rotation: int = 20) -> transforms.Compose:
    """Augmentation used during training.

    TrashNet is small (1,768 training images), so the augmentation is doing
    most of the regularisation work.  Every transform here is justified by a
    condition the bin really meets: the item can land rotated, at a different
    scale, at a different distance, under a different white balance.
    """
    _, _, _, transforms = _torch_stack()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(imgsz, scale=(scale_min, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(rotation),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def eval_transform(imgsz: int) -> transforms.Compose:
    """Deterministic pre-processing used for validation, test and deployment."""
    _, _, _, transforms = _torch_stack()
    resize = int(round(imgsz / 0.875))  # same ratio torchvision uses (224 <- 256)
    return transforms.Compose(
        [
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


# ---------------------------------------------------------------------------
# Four-stream routing lives in routing.py (standard library only) and is
# re-exported here so that call sites can keep importing from one module.
# ---------------------------------------------------------------------------
from routing import CLASS_TO_STREAM, STREAMS, stream_for  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    labels: Sequence[str],
    model_name: str,
    imgsz: int,
    extra: Dict[str, object] | None = None,
) -> None:
    torch, _, _, _ = _torch_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "labels": list(labels),
        "model_name": model_name,
        "imgsz": imgsz,
        "normalize": {"mean": MEAN, "std": STD},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu") -> Tuple[nn.Module, List[str], int]:
    """Rebuild a trained model from a checkpoint written by save_checkpoint()."""
    torch, _, _, _ = _torch_stack()
    payload = torch.load(path, map_location=device, weights_only=False)
    labels = payload["labels"]
    model = build_model(payload["model_name"], len(labels), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(device)
    return model, labels, int(payload.get("imgsz", 144))


def describe(text: str) -> None:
    """Small helper used by the CLIs for consistent banners."""
    print("=" * 70)
    print(text)
    print("=" * 70)


def load_labels_from_dir(root: Path) -> List[str]:
    """Sorted class-folder names - the ImageFolder convention."""
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
