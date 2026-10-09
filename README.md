# YOLO-ADF

Code and replication package for the paper *"YOLO-ADF: Real-Time Attention–Frequency Fusion Object Detection for Adverse Weather"*.

YOLO-ADF extends YOLOv11m with (i) a discrete wavelet transform (DWT) frequency branch combined with attention modules, and (ii) weather-specific detection branches — ADF-F / ADF-S / ADF-N for fog / snow / night and CA for rain — switched at inference time by a lightweight MobileNetV3-Small weather classifier.

Key measured results (all numbers below are reproduced by the files in [`results/`](results/)):

| Item | Value |
|---|---|
| Weather classifier accuracy (BDD100K val, 5 classes) | 94.9% (best val acc 0.9487) |
| Weather classifier zero-shot accuracy (ACDC, n = 406) | 65.3% |
| WADI, oracle switching (ideal weather label) | 0.4806 |
| WADI, realized switching (measured classifier routing) | 0.465 |
| Ideal-switching gain retained under measured routing | 96.8% |

## Repository layout

```
YOLO-ADF/
├── detector/                  # YOLO-ADF detector: training + evaluation
│   ├── custom_modules.py      # DWT / window-Transformer / attention / NMS module definitions
│   ├── register_modules.py    # registers custom modules into the Ultralytics framework
│   ├── yolo_compatible_modules.py / _fixed.py   # Ultralytics-compatible wrappers
│   ├── train_config.py        # unified config for the 7 ablation experiments (exp01–exp07)
│   ├── train_unified.py       # unified training entry point
│   ├── run_experiments.py     # batch runner
│   ├── eavl_bdd.py            # BDD100K evaluation
│   ├── eval_exp15.py          # per-experiment evaluation
│   ├── visualize_acdc_results.py  # ACDC qualitative visualizations
│   ├── GFLOPs.py, nms.py, time.py # complexity / NMS / latency probes
│   ├── bdd100k.yaml           # dataset config (machine-specific paths — edit for your layout)
│   ├── yolo11m*.yaml          # model definitions for all variants
│   ├── training_notes_zh.txt  # training commands and notes (Chinese)
│   └── results/
│       ├── final_ablation_result.csv    # final ablation table (all 7 variants)
│       ├── ablation_results_train.csv   # training-time ablation records
│       ├── amar_module_diagram.png      # module schematic
│       └── final_run/                   # final run config (args.yaml) + training curve (results.csv)
├── weather_classifier/        # MobileNetV3-Small weather classifier (see its README.md, Chinese)
│   ├── models.py, dataset.py, common.py
│   ├── train.py, benchmark.py # training + accuracy/latency benchmark
│   ├── build_labels.py, rebuild_labels_full.py  # BDD100K attribute-based label construction
│   ├── download_bdd100k.py, find_full_train.py, find_acdc_full.py
│   ├── compute_wadi_real.py   # WADI computation (oracle + realized, bootstrap CI)
│   ├── routing_breakeven.py   # routing-threshold sensitivity analysis
│   ├── labels_bdd.csv         # scene-level weather labels used for training
│   ├── table3.csv             # per-model, per-scene mAP50 (detector side)
│   └── 实验步骤.md             # step-by-step experiment log (Chinese)
├── results/                   # consolidated benchmark outputs
│   ├── wadi_routing_per_scene.json        # oracle vs. realized WADI, per-scene routing
│   ├── classifier_benchmark_full.json     # full classifier benchmark
│   └── classifier_train_history.json      # classifier training history (10 epochs)
└── release-assets/            # (git-ignored) trained weights, uploaded to GitHub Releases
```

## Environment

- Python ≥ 3.9
- PyTorch ≥ 2.0 with a CUDA build matching your GPU (training used 4× RTX 3090)
- Ultralytics ≥ 8.3 (detector), plus `requirements.txt`

```bash
pip install -r requirements.txt
```

## Data

All experiments use public datasets only:

- **BDD100K** (https://www.bdd100k.com) — training and in-domain evaluation. Weather labels for the classifier are built from the official BDD100K attribute annotations (weather + time-of-day, night takes precedence), after excluding dawn/dusk, unlabeled, and missing-image frames; the construction script is included (`weather_classifier/build_labels.py`, `rebuild_labels_full.py`).
- **ACDC** (https://acdc.vision.ee.ethz.ch) — used **strictly for zero-shot cross-domain evaluation** (validation scenes fog / rain / snow / night, n = 406); no ACDC training images are used anywhere.

`detector/bdd100k.yaml` contains machine-specific dataset paths — edit it to point at your local BDD100K copy.

## Reproducing the paper numbers

1. **Weather classifier** — see `weather_classifier/README.md` and `实验步骤.md` (step-by-step, Chinese): `train.py` (training), `benchmark.py` (accuracy / latency), `routing_breakeven.py` (routing sensitivity). Expected: 94.9% BDD100K val, 65.3% ACDC (n = 406), single-image latency in the low milliseconds on GPU.
2. **WADI** — `weather_classifier/compute_wadi_real.py` computes the oracle (0.4806) and realized (0.465) routing WADI with bootstrap confidence intervals, using `table3.csv` (per-scene mAP50 of all branches) and classifier predictions.
3. **Detector ablation (7 variants)** — `detector/train_config.py` defines exp01–exp07; launch with
   `python train_unified.py --exp exp01` (see `detector/training_notes_zh.txt` for the full command list),
   then evaluate with `eavl_bdd.py` / `eval_exp15.py`. Final numbers: `detector/results/final_ablation_result.csv`.

## Trained weights

Trained detector weights (`yolo_adf_best.pt`) and the weather-classifier checkpoint are attached to this repository's **Releases** (too large for git). `release-assets/` is git-ignored and holds the local copy for upload.

## License

Released under the [MIT License](LICENSE).

## Citation

Paper under review. BibTeX will be added upon acceptance.
