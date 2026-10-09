# -*- coding: utf-8 -*-
# create_acdc_yamls.py
from pathlib import Path

ACDC_ROOT = Path(r"C:\Users\KUST\xlw\acdc")
SCENARIOS = {
    "acdc_fog":   "fog/val",
    "acdc_night": "night/val",
    "acdc_rain":  "rain/val",
    "acdc_snow":  "snow/val",
}
NAMES = ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "truck"]

for yaml_name, val_dir in SCENARIOS.items():
    val_path = ACDC_ROOT / "rgb_anon" / val_dir
    yaml_path = ACDC_ROOT / f"{yaml_name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {val_path}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write("nc: 9\n")
        f.write("names:\n")
        for i, name in enumerate(NAMES):
            f.write(f"  {i}: {name}\n")
    print(f"Generated {yaml_path}")