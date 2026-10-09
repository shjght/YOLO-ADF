# -*- coding: utf-8 -*-
"""Backbones for the weather classifier benchmark.

All ImageNet-pretrained (torchvision weights API, torchvision >= 0.13).
MobileNetV3-Small is the deployment candidate; the others bracket the
accuracy/latency trade-off for the paper's Table.
"""

import torch.nn as nn
from torchvision import models

from common import CLASSES

# name -> (builder, feature_dim)
BACKBONES = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, 576),
    "mobilenet_v3_large": (models.mobilenet_v3_large, 960),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, 1024),
    "resnet18":           (models.resnet18, 512),
    "efficientnet_b0":    (models.efficientnet_b0, 1280),
}


def build_model(name: str, num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    if name not in BACKBONES:
        raise KeyError(f"unknown model '{name}', choose from {sorted(BACKBONES)}")
    builder, feat = BACKBONES[name]
    try:
        weights = "IMAGENET1K_V1" if pretrained else None
        m = builder(weights=weights)
    except TypeError:  # very old torchvision fallback
        m = builder(pretrained=pretrained)

    if name.startswith("mobilenet_v3"):
        m.classifier[3] = nn.Linear(feat, num_classes)
    elif name.startswith("efficientnet"):
        m.classifier[1] = nn.Linear(feat, num_classes)
    elif name.startswith("shufflenet"):
        m.fc = nn.Linear(feat, num_classes)
    elif name.startswith("resnet"):
        m.fc = nn.Linear(feat, num_classes)
    return m


def count_params(model: nn.Module) -> float:
    """Trainable params in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
