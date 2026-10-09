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
│   ├── yolo_compatible_modules_fixed.py   # Ultralytics-compatible module wrappers
│   ├── train_config.py + train_unified.py # component-ablation experiments exp01–exp07
│   ├── train_exp08_final_fixed.py, train_exp11/12/13.py   # final weather-branch models
│   ├── train_scale_adaptive*.py           # scale-adaptive attention line (AMAR, ADF-F)
│   ├── train_dwt.py, train_comp_ca.py, train_comp_se.py   # DWT-only / CA / SE comparison models
│   ├── multiseed/             # canonical evaluation pipeline
│   │   ├── eval_all_multiseed.py  # evaluates all models on BDD100K + ACDC scenarios
│   │   ├── eval_acdc_multiseed.py, run_all.py, train.py
│   │   ├── aggregate.py, aggregate_final.py
│   │   └── 训练代码与实验对应说明.docx  # code ↔ experiment mapping table (Chinese)
│   ├── acdc/                  # ACDC zero-shot evaluation
│   │   ├── acdc_fog / night / rain / snow.yaml
│   │   ├── create_acdc_yamls.py   # regenerates the scenario configs
│   │   └── eval_acdc_final.py
│   ├── eavl_bdd.py            # BDD100K evaluation
│   ├── visualize_acdc_results.py  # ACDC qualitative visualizations
│   ├── GFLOPs.py, nms.py, time.py # complexity / latency / timing probes
│   ├── bdd100k.yaml           # dataset config (machine-specific paths — edit for your layout)
│   ├── yolo11m*.yaml          # model definitions for all variants
│   ├── training_notes_zh.txt  # training commands and notes (Chinese)
│   └── results/
│       ├── final_ablation_result.csv    # final ablation table (all variants)
│       ├── ablation_results_train.csv   # training-time ablation records
│       ├── amar_module_diagram.png      # module schematic
│       └── final_run/                   # final run config (args.yaml) + training curve (results.csv)
├── weather_classifier/        # MobileNetV3-Small weather classifier (see its README.md, Chinese)
│   ├── models.py, dataset.py, common.py
│   ├── train.py, benchmark.py # training + accuracy/latency benchmark
│   ├── build_labels.py, rebuild_labels_full.py  # BDD100K attribute-based label construction
│   ├── download_bdd100k.py
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

Note: several scripts contain absolute paths from the original training machine (`C:\Users\KUST\xlw\...`). Adjust them for your own dataset layout; `bdd100k.yaml` and the `acdc/*.yaml` files are the central places to edit.

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
- **ACDC** (https://acdc.vision.ee.ethz.ch) — used **strictly for zero-shot cross-domain evaluation** (validation scenes fog / rain / snow / night, n = 406); no ACDC training images are used anywhere. Scenario configs are in `detector/acdc/` (regenerate with `create_acdc_yamls.py`).

## Reproducing the paper numbers

1. **Weather classifier** — see `weather_classifier/README.md` and `实验步骤.md` (step-by-step, Chinese): `train.py` (training), `benchmark.py` (accuracy / latency), `routing_breakeven.py` (routing sensitivity). Expected: 94.9% BDD100K val, 65.3% ACDC (n = 406).
2. **Detector** — the component ablation is defined in `detector/train_config.py` (exp01–exp07, launched with `python train_unified.py --exp exp01` etc., see `training_notes_zh.txt`); the final weather-branch models are trained by `train_exp08_final_fixed.py` / `train_exp11–13.py` / `train_scale_adaptive*.py` (mapping in `multiseed/训练代码与实验对应说明.docx`).
3. **Evaluation** — `detector/multiseed/eval_all_multiseed.py` evaluates every model on BDD100K and the four ACDC scenarios; aggregators (`aggregate.py`, `aggregate_final.py`) produce the per-model, per-scene mAP50 table (cf. `weather_classifier/table3.csv`).
4. **WADI** — `weather_classifier/compute_wadi_real.py` computes the oracle (0.4806) and realized (0.465) routing WADI with bootstrap confidence intervals, using `table3.csv` and classifier predictions.

## Trained weights

The trained detector weights (`yolo_adf_best.pt`) are attached to this repository's **Releases** (too large for git); the weather-classifier checkpoint will be added to the same place. `release-assets/` is git-ignored and holds the local copy for upload.

## License

Released under the [MIT License](LICENSE).

## Citation

Paper under review. BibTeX will be added upon acceptance.
