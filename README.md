# SCLC-Classification

Deep-learning pipeline for **3-class lung-cancer subtype classification** (Adenocarcinoma / Small-Cell / Squamous) from CT volumes. Built for the master's thesis "PneumOrpheus" (NTNU, 2026). Two-phase recipe: **DAPT** on `Lung-PET-CT-Dx`, then **fine-tune** on `BigLunge`.

Three model families share one entrypoint (`main.py`) and one config format (`configs/experiments/*.yaml`):

| Pipeline | Models | Input | Notes |
|---|---|---|---|
| **2D** (single slice) | `efficientnet_b0_2d`, `resnet50_2d`, `densenet121_2d`, `swin_tiny_2d`, `resnet50_2d_rin`, `densenet121_2d_rin` | `(1, H, W)` axial slice cropped around the largest tumor CC | `_rin` variants use RadImageNet-pretrained backbones (ImageNet otherwise) |
| **MIL** | `mil_resnet50` | `(N, 1, H, W)` bag of lung-anchored slices | Attention pooling (MONAI `MILModel`); DAPT trains a 2D ResNet-50, FT rebuilds as MIL and transfers backbone weights |
| **3D** | `swin_unetr` | `(1, X, Y, Z)` volume with optional segmentation aux loss | MONAI SwinUNETR; classification via forward hook on `swinViT` |

## Class taxonomy

| Index | Class | Lung-PET-CT-Dx letter | BigLunge `MorphologicalGroup` |
|---|---|---|---|
| 0 | Adenocarcinoma (ADC) | `A` | `Adenokarsinom` / `Adenocarcinoma` |
| 1 | Small Cell Carcinoma (SCLC) | `B` | `Småcelletkarsinom` / `Small cell carcinoma` |
| 2 | Squamous Cell Carcinoma (SCC) | `G` | `Plateepitelkarsinom` / `Squamous cell carcinoma` |

Large Cell (`E` / `Storcellet`) and other morphological groups are filtered out.

## Repository layout

```
SCLC-Classification/
├── README.md                       # you are here
├── LICENSE
├── pyproject.toml                  # `pip install -e .` makes the `sclc` package importable
├── environment.yaml
├── requirements.txt
├── Dockerfile
│
├── sclc/                           # the source package
│   ├── main.py                     # pipeline entrypoint (DAPT -> fine-tune -> inference)
│   ├── logger.py
│   ├── models/                     # model factory + classifier wrappers
│   │   ├── factory.py              # get_sclc_model + pipeline helpers
│   │   ├── swin_unetr.py
│   │   ├── classifiers_2d.py       # ImageNet 2D wrappers
│   │   ├── classifiers_rin.py      # RadImageNet 2D wrappers
│   │   └── classifiers_mil.py      # MIL bag classifier
│   ├── data/                       # runtime data loading
│   │   ├── loaders.py              # 3D dataset + splits + class maps
│   │   ├── dataset_2d.py
│   │   ├── dataset_mil.py
│   │   ├── transforms.py           # MONAI transforms (3D / 2D / MIL)
│   │   └── exclusions.py           # patient blocklist
│   ├── training/
│   │   ├── train_3d.py             # 3D train/validate
│   │   ├── train_2d.py
│   │   ├── train_mil.py
│   │   └── bootstrap.py            # patient-level bootstrap CIs
│   └── grad_cam/                   # interpretability tooling
│       ├── grad_cam.py             # `python -m sclc.grad_cam.grad_cam ...`
│       ├── colorize.py
│       └── mock.py
│
├── configs/experiments/            # one YAML per model
│
├── scripts/                        # runnable analysis / orchestration tools
│   ├── build_thesis_results.py     # consolidates results/output/ into results/thesis/
│   ├── thesis_plots.py             # learning curves, confusion matrices, ROC
│   ├── report_test_metrics.py      # markdown summary of metrics.jsonl
│   ├── audit_multifocal.py         # multifocal-mask audit (BigLunge)
│   └── runners/                    # bash chains
│       ├── run_all_2d_v3.sh        # sequential runner for the 6 2D configs
│       ├── run_swinunetr_ft_then_infer.sh
│       └── run_swinunetr_then_mil.sh
│
├── data_pipeline/                  # one-shot dataset acquisition / preprocessing
│   ├── README.md                   # reproducibility chain
│   ├── create_masks.py             # produces /home/data/Lung-PET-CT-Dx-Clean
│   ├── annotation_mapping.py
│   ├── recover_annotations.py
│   ├── fetch_tcia.ipynb
│   └── notebooks/                  # exploratory data analysis
│       ├── biglunge_audit.ipynb
│       └── eda_2d.ipynb
│
├── results/                        # all training artifacts under one tree
│   ├── output/<pipeline>/<model>/  # raw per-run logs + metrics.jsonl + inference probs
│   ├── runs/                       # per-run shell-script summaries
│   ├── thesis/                     # consolidated tables + figures
│   └── figures/                    # standalone thesis PDFs
│
└── docs/
    ├── limitations.md              # methodological audit
    └── augmentations_2d.md         # strong-augs experiment notes
```

## Default data paths

Hardcoded in `sclc/main.py` and `sclc/data/loaders.py`:

- DAPT: `/home/data/Lung-PET-CT-Dx-Clean/{patient}/{series_uid}_image.nii.gz` (+ optional `_mask.nii.gz`)
- Fine-tune: `/home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1` + `patients_parameters.csv`
- SwinUNETR pretrain: `/home/data/pre_trained_models/model_swin_unetr_btcv_segmentation_v1.pt`
- RadImageNet weights: `/home/hansstem/RadImageNet_swin/rin_swintf.pth`, `/home/data/pre_trained_models/RadImageNet-ResNet50_notop.pth`, `/home/data/pre_trained_models/RadImageNet-DenseNet121_notop.pth`

`PersistentDataset` caches live under `~/.cache/monai_*/`.

## Running

Always launch long jobs in **tmux** (a typical full run is 1-6 h).

First, install the package in editable mode (one-time setup):
```bash
pip install -e .
```

```bash
# Full pipeline: DAPT -> DAPT-test -> fine-tune -> BL-test
python -m sclc.main --config configs/experiments/2d_efficientnet_b0.yaml

# DAPT only (stops after DAPT-test)
python -m sclc.main --config configs/experiments/3d_swin_unetr.yaml --mode dapt

# Resume fine-tune from a saved DAPT pbest checkpoint
python -m sclc.main --config configs/experiments/3d_swin_unetr.yaml \
    --mode finetune --model-checkpoint /path/to/dapt_pbest_raw.pth

# Inference only (BL-test) from a fine-tune pbest
python -m sclc.main --config configs/experiments/3d_swin_unetr.yaml \
    --mode inference --model-checkpoint /path/to/finetune_pbest_raw.pth

# Sequential runner: all 6 2D models, dapt 30 ep
bash scripts/runners/run_all_2d_v3.sh results/runs/$(date +%Y-%m-%d)_2d
```

Common CLI overrides (otherwise read from the YAML):

- `--mode {full,dapt,finetune,inference}`
- `--model-type` — must match the loaded config
- `--batch-size`, `--accumulation-steps` (default effective batch = 8)
- `--depth-size` (3D pipeline only, default 128)
- `--testing` — tiny-subset smoke test
- `--clear-cache` — rebuild this run's PersistentDataset cache (scoped to img_size / depth_size / bag_size)

## Training recipe

Both phases use:

- `CrossEntropyLoss(label_smoothing=0.1)`.
- `BCEWithLogitsLoss * 0.5` segmentation aux loss when masks are present (3D `swin_unetr` only).
- `AdamW`, cosine schedule, AMP, grad clipping at norm 1.0, gradient accumulation.
- Patient-level 70 / 15 / 15 stratified split. DAPT uses `WeightedRandomSampler`; BigLunge uses plain shuffle.
- LP-FT: backbone frozen for `finetune_freeze_backbone_epochs` epochs, then unfrozen with a 10x lower LR.
- **Dual pbest**: best-by-rolling-3-mean validation accuracy (`*_pbest_roll`) and best single-epoch (`*_pbest_raw`) are both saved per phase. Test inference runs from `*_pbest_raw`.

## Outputs

```
results/output/<pipeline>/<model_type>/
├── metrics.jsonl                          # one row per epoch + DAPT-test + BL-test
├── inference_probabilities_*.json         # per-patient softmax + labels
├── misclassifications_*.csv
└── *.log

results/thesis/<pipeline>/
├── per_model/<model_type>/                # CSVs, confusion matrices, ROC
├── tables/headline.md                     # overall + per-class metrics with bootstrap CIs
├── figures/                               # accuracy / AUC / F1 bar plots, learning curves
└── README.md                              # auto-generated summary
```

`python scripts/build_thesis_results.py --pipeline {2d,mil,3d}` rebuilds the `results/thesis/` tree from `results/output/` and snapshots the previous version under `results/thesis/_archive/` before overwriting.

## See also

- `docs/limitations.md` — thesis limitations / known caveats (tumor-mask multifocality, lung vs tumor anchoring, RadImageNet weight provenance).
- `data_pipeline/notebooks/biglunge_audit.ipynb` — BigLunge tumor-mask audit; `min_tumor_pixels` and the truncated-lung-mask exclusion list (`sclc/data/exclusions.py`) are derived from it.

## Acknowledgments

- Boban Vesin (NTNU) — thesis supervisor.
- David Bouget, Erlend Fagertun Hofstad (SINTEF, BigLunge group) — dataset and clinical guidance.
- Håkon Leira (St. Olavs Hospital) and the interviewed radiologists.
- MONAI for the 3D imaging transforms and networks.
- timm, torchvision, RadImageNet for pretrained weights.
