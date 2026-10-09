# -*- coding: utf-8 -*-
"""Shared config for the lightweight weather classifier benchmark (R1-b).

Classes follow the YOLO-ADF routing table exactly:
  clear -> no ADF branch (bypass)
  rain  -> ADF-R   snow -> ADF-S   fog -> ADF-F   night -> ADF-N

Label sources (no manual annotation needed):
  BDD100K  attributes.weather + attributes.timeofday  (original det label JSONs)
  ACDC     folder name (fog / rain / snow / night), val split only
"""

import os
import random
from pathlib import Path

import numpy as np

CLASSES = ["clear", "rain", "snow", "fog", "night"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Candidate dataset roots, probed in order (local mirror first, then training machine).
CANDIDATE_ROOTS = [
    Path(__file__).resolve().parent.parent,      # F:/xlw  or  C:/Users/KUST/xl
    Path("F:/xlw"),
    Path("C:/Users/KUST/xl"),
    Path("C:/Users/KUST/xlw"),
]

# Clear-ish weather values in BDD100K folded into the "clear" class.
CLEAR_WEATHER = {"clear", "partly cloudy", "cloudy", "overcast"}
WEATHER2CLS = {
    "clear": "clear", "partly cloudy": "clear", "cloudy": "clear", "overcast": "clear",
    "rainy": "rain", "snowy": "snow", "foggy": "fog",
}
VALID_TIMEOFDAY = {"daytime", "night"}  # dawn/dusk and undefined are excluded


def bdd_to_label(weather: str, timeofday: str):
    """Map raw BDD100K attributes to one of CLASSES; None = exclude frame.

    Night takes precedence over weather (routing picks a single branch),
    consistent with the ADF-N design.
    """
    if timeofday == "night":
        return "night"
    if timeofday != "daytime":
        return None
    return WEATHER2CLS.get(weather)


def find_data_root() -> Path:
    """Return first existing root that contains bdd100k_final/labels."""
    for root in CANDIDATE_ROOTS:
        cand = root / "bdd100k_final" / "labels"
        if (cand / "bdd100k_labels_images_train.json").exists() or \
           (cand / "bdd100k_labels_images_val.json").exists():
            return root
    raise FileNotFoundError(
        "Could not locate dataset root containing bdd100k_final/labels/*.json. "
        "Pass --data-root explicitly.")


def find_acdc_root(data_root: Path = None) -> Path:
    """Locate the ACDC folder (rgb_anon/<cond>/<split>/<seq>/images/*.png)."""
    candidates = []
    if data_root is not None:
        candidates += [data_root / "ACDC", data_root / "acdc", data_root.parent / "ACDC"]
    candidates += [Path("F:/xlw/ACDC"), Path("C:/Users/KUST/xl/ACDC"),
                   Path("C:/Users/KUST/xlw/acdc")]
    for c in candidates:
        if (c / "rgb_anon" / "fog" / "val").exists():
            return c
    raise FileNotFoundError(
        "Could not locate ACDC (rgb_anon/<cond>/val/...). Pass --acdc-root explicitly.")


def set_seed(seed: int = 0):
    """Global seed (paper convention: seed 0)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def acdc_val_files(acdc_root: Path):
    """Yield (image_path, class_name) for the ACDC val split.

    Supports both layouts:
      flattened:  rgb_anon/<cond>/val/images/*.png   (this project's converted copy)
      standard:   rgb_anon/<cond>/val/<seq>/images/*.png
    ACDC is all-adverse, so 'clear' never appears here.
    """
    pats = [("images/*.png",), ("images/*.jpg",),
            ("*/*/images/*.png",), ("*/*/images/*.jpg",)]
    for cond, cls in [("fog", "fog"), ("rain", "rain"), ("snow", "snow"), ("night", "night")]:
        split_dir = acdc_root / "rgb_anon" / cond / "val"
        if not split_dir.exists():
            continue
        seen = set()
        for pat in [p for ps in pats for p in ps]:
            for img in sorted(split_dir.glob(pat)):
                if img.resolve() not in seen:
                    seen.add(img.resolve())
                    yield img, cls
