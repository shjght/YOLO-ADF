# -*- coding: utf-8 -*-
"""Step 1: parse original BDD100K label JSONs -> labels_bdd.csv.

Run this BEFORE training, on whichever machine holds the label JSONs:
    python build_labels.py                       # auto-detect F:/xlw or C:/Users/KUST/xl
    python build_labels.py --data-root D:/somewhere

Output: weather_clf/labels_bdd.csv with columns
    name, split, weather_raw, timeofday_raw, label
Only frames whose image file actually exists in one of the image dirs are kept
(the local mirror holds a train subset; the full 70k lives on the training machine).
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import CLASSES, bdd_to_label, find_data_root


def collect_image_dirs(images_root: Path, split: str, extra: str = ""):
    """Candidate folders that may hold the split's jpgs (probed in order)."""
    dirs = [images_root / "100k" / split]
    if split == "train":
        dirs += [images_root / "train_merged", images_root / "train"]
    if extra:
        dirs += [Path(p) for p in extra.split(",") if p.strip()]
    return [d for d in dirs if d.exists()]


def build_existing_index(dirs):
    idx = {}
    for d in dirs:
        for p in d.iterdir():
            if p.suffix.lower() == ".jpg":
                idx[p.stem] = p
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="folder containing bdd100k_final/")
    ap.add_argument("--extra-train-dirs", default="",
                    help="comma-separated extra folders with training jpgs")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "labels_bdd.csv"))
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else find_data_root()
    labels_dir = root / "bdd100k_final" / "labels"
    images_root = root / "bdd100k_final" / "images"

    rows, excluded = [], Counter()
    stats = {}
    for split, json_name in [("train", "bdd100k_labels_images_train.json"),
                             ("val", "bdd100k_labels_images_val.json")]:
        jp = labels_dir / json_name
        if not jp.exists():
            print(f"[warn] {jp} not found, skipping split '{split}'")
            continue
        print(f"parsing {jp.name} ...")
        frames = json.loads(jp.read_text(encoding="utf-8"))
        img_dirs = collect_image_dirs(images_root, split, args.extra_train_dirs)
        existing = build_existing_index(img_dirs)
        print(f"  {len(frames)} frames in json, image dirs probed: "
              f"{[str(d) for d in img_dirs]}, {len(existing)} jpgs found on disk")

        kept = Counter()
        for fr in frames:
            attr = fr.get("attributes") or {}
            weather = attr.get("weather")
            tod = attr.get("timeofday")
            label = bdd_to_label(weather, tod)
            if label is None:
                excluded[f"{weather or '?'}|{tod or '?'}"] += 1
                continue
            stem = Path(fr["name"]).stem
            if stem not in existing:
                excluded["image-missing-on-disk"] += 1
                continue
            rows.append({"name": existing[stem].name,
                         "path": str(existing[stem]),
                         "split": split,
                         "weather_raw": weather,
                         "timeofday_raw": tod,
                         "label": label})
            kept[label] += 1
        stats[split] = dict(kept)
        print(f"  kept: {dict(kept)}")

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "path", "split", "weather_raw",
                                          "timeofday_raw", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out}  ({len(rows)} rows)")
    print("excluded:", dict(excluded))
    for split in stats:
        total = sum(stats[split].values())
        print(f"\n[{split}] distribution (total {total}):")
        for c in CLASSES:
            n = stats[split].get(c, 0)
            pct = 100.0 * n / total if total else 0.0
            print(f"  {c:6s}: {n:6d}  ({pct:.2f}%)")
        n_fog = stats[split].get("fog", 0)
        if 0 < n_fog < total * 0.01:
            print(f"  [note] fog is scarce in BDD100K ({n_fog} frames) - known dataset "
                  f"property; fog routing is additionally evaluated cross-domain on ACDC.")


if __name__ == "__main__":
    main()
