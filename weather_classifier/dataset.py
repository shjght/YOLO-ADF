# -*- coding: utf-8 -*-
"""Datasets and transforms for the weather classifier benchmark."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common import CLASSES, CLS2IDX

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),   # squash (routing input is fixed-size, no center-crop truncation)
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BddWeatherDataset(Dataset):
    """Reads labels_bdd.csv produced by build_labels.py."""

    def __init__(self, csv_path, split="val", size=224, train=False):
        self.samples = []
        with Path(csv_path).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split and row["label"] in CLS2IDX:
                    self.samples.append((row["path"], CLS2IDX[row["label"]]))
        self.tf = train_transform(size) if train else eval_transform(size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), y


class AcdcValDataset(Dataset):
    """ACDC val split, labels from folder condition (fog/rain/snow/night).

    Used only for evaluation (never training) - keeps the paper's cross-domain
    protocol intact: classifier trains on BDD100K only.
    """

    def __init__(self, acdc_root, size=224, files=None):
        if files is None:
            from common import acdc_val_files
            files = list(acdc_val_files(Path(acdc_root)))
        self.samples = [(str(p), CLS2IDX[c]) for p, c in files]
        self.tf = eval_transform(size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), y


def class_weights(labels, damp: float = 0.5):
    """sqrt-damped inverse-frequency weights (mean=1).

    Raw 1/freq would give fog ~390x weight over clear (130 vs 51k frames) and
    destabilize training; damping by **0.5 keeps the rare classes visible
    without letting 130 fog frames dominate the gradient.
    """
    counts = torch.zeros(len(CLASSES))
    for y in labels:
        counts[y] += 1
    counts = counts.clamp(min=1)
    w = counts.max() / counts
    w = w ** damp
    return w / w.mean(), counts
