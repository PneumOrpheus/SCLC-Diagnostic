# SCLC-Classification

A deep learning pipeline for **3D lung cancer subtype classification** from CT volumes, with a focus on **Small Cell Lung Cancer (SCLC)**. The system performs volumetric classification with an optional auxiliary segmentation head, supports several 3D backbones (SwinUNETR, ResNet, ViT, DenseNet, Models Genesis), and follows a two-phase transfer learning recipe: **Domain-Adaptive Pre-Training (DAPT)** on Lung-PET-CT-Dx followed by **fine-tuning** on the BigLunge dataset.

## 🔬 Overview

End-to-end pipeline for:
- **DAPT** on Lung-PET-CT-Dx to learn lung-CT-specific features
- **Fine-tuning** on BigLunge for target classification
- **3D volumetric preprocessing** with RAS reorientation, isotropic-ish resampling, HU windowing, and depth cropping around lesion masks when available
- **Optional PET input** as a second channel (CT + PET co-registered)
- **Optional auxiliary segmentation loss** when per-sample lesion masks are provided

### Target Classes

The model classifies CT scans into **3 lung cancer subtypes**, aligned across both training phases:

| Index | Class | DAPT Label (Lung-PET-CT-Dx) | Fine-tune Label (BigLunge) |
|-------|-------|------------------------------|----------------------------|
| 0 | Adenocarcinoma | `A` | `Adenokarsinom` |
| 1 | Small Cell Carcinoma | `B` | `Småcelletkarsinom` |
| 2 | Squamous Cell Carcinoma | `G` | `Plateepitelkarsinom` |

> Other classes (e.g. Large Cell Carcinoma `E`) are filtered out during data loading to keep class alignment consistent between both phases.

## 🏗️ Architecture

### Model factory

`model_selection.get_sclc_model(checkpoint_path, model_type, in_channels, depth_size)` returns one of:

| `--model-type` | Wrapper class | Backbone |
|---|---|---|
| `swin_unetr` (default) | `SwinUNETRClassifier` | MONAI SwinUNETR (v2), bottleneck features via forward hook |
| `resnet50` | `ResNetClassifier` | MONAI 3D ResNet-50 (MedicalNet weights) |
| `resnet18` | `ResNet18Classifier` | MONAI 3D ResNet-18 |
| `vit` | `ViTClassifier` | MONAI 3D ViT (tiny: hidden 384, 6 layers, 6 heads) |
| `densenet121` | `DenseNetClassifier` | MONAI 3D DenseNet-121 |
| `models_genesis` | `ModelsGenesisClassifier` | UNet3D encoder (requires sibling `ModelsGenesis` repo) |

All wrappers expose `forward(x, return_segmentation=False)`. When `return_segmentation=True`, they return `(cls_logits, seg_logits)`; models without a native decoder return a zero-tensor mask so the training loop stays uniform. SwinUNETR uses its real decoder output for segmentation and a forward hook on `swinViT` for classification features, bypassing the decoder entirely on classification-only runs for speed.

Checkpoint loading is tolerant: it strips `state_dict` / `module.` wrappers, drops mismatched final-layer keys, and adapts the first-conv / patch-embed weights when switching from 1 to 2 input channels (CT → CT+PET) by copying the CT channel and mean-initializing the PET channel.

### Training recipe

```
Pretrained weights ──► DAPT (Lung-PET-CT-Dx) ──► Fine-tune (BigLunge) ──► Test
```

**Phase 1 — DAPT (default 40 epochs, lr 1e-4)** on Lung-PET-CT-Dx. A `WeightedRandomSampler` with inverse-frequency weights compensates for the heavy class imbalance in this dataset.

**Phase 2 — Fine-tune (default 40 epochs, lr 3e-5)** on BigLunge. Plain shuffling; class-weighted loss is recomputed from the fine-tuning split.

Both phases use:
- `AdamW` optimizer, weight decay `1e-3`
- `CosineAnnealingLR` scheduler
- `CrossEntropyLoss(label_smoothing=0.1)` (class-weighted from the training split) plus optional BCE-with-logits segmentation loss weighted at 0.5, applied only to samples that actually carry a non-empty mask
- Mixed precision via `torch.amp.GradScaler` (disable with `--disable-amp`)
- Gradient accumulation (default 4 steps — effective batch size `batch_size * accumulation_steps`)
- Gradient clipping at norm 1.0
- Best checkpoint by **validation accuracy**, saved as `best_{model_type}_{phase}_new.pth`; periodic checkpoints every 10 epochs

### 3D data preprocessing

Defined in `data/transforms.py` (`get_train_transforms_3d` / `get_val_transforms_3d`):

1. `LoadNiftiWithRGBSupportd` — robust NIfTI loader handling structured RGB dtypes, 4D time-series, and degenerate shapes
2. `EnsureChannelFirstd` + `Orientationd(axcodes="RAS")`
3. `ResampleToMatchd` for PET → CT grid (when `--use-pet`)
4. `Spacingd(pixdim=(1.5, 1.5, 2.0))`
5. `ScaleIntensityRanged(a_min=-1024, a_max=3071)` on CT; `ScaleIntensityd` on PET
6. `AsDiscreted(threshold=0.5)` on masks
7. `ExtractSubVolumed` — crops `depth_size` slices centered on the lesion (uses mask extent when present, otherwise volume center)
8. `Resized` to `(img_size, img_size, depth_size)` — default `224×224×128`
9. **Train-only** augmentations: random flips on all 3 spatial axes (p=0.5 each), `RandScaleIntensityd` (±0.1, p=0.3), `RandShiftIntensityd` (±0.1, p=0.3)
10. `NormalizeIntensityd(nonzero=True, channel_wise=True)`
11. `ConcatItemsd` to fuse CT+PET into a 2-channel `image` when PET is enabled

Samples are cached with MONAI `PersistentDataset` under `~/.cache/{monai_biglunge|monai_lung_pet_ct_clean}/3d_img{SZ}_d{D}[_testing]/{split}/`. On the first run the loader validates every sample, stores the surviving list in `valid_data.json`, and silently drops unreadable volumes.

## 📁 Active pipeline files

```
SCLC-Classification/
├── main.py                 # Pipeline orchestrator (DAPT → fine-tune → inference)
├── model_selection.py      # get_sclc_model() factory + classifier wrappers
├── logger.py               # Logging utilities
├── data/
│   ├── data_loader.py      # create_dataset(), split logic, class maps
│   ├── transforms.py       # MONAI 3D train/val transforms
│   └── data_preprocessing.py
├── training/
│   └── train.py            # train_epoch, validate_epoch, simple_collate_fn
├── models/                 # Legacy Swin Transformer V1/V2 2D implementation (not used by main.py)
├── environment.yaml
├── requirements.txt
└── Dockerfile
```

### Dataset layout

**Lung-PET-CT-Dx** (DAPT):
```
<dapt-dataset>/
└── <patient_id>/                       # patient_id must contain "-A", "-B", or "-G"
    ├── <series_uid>_image.nii.gz
    └── <series_uid>_mask.nii.gz        # optional, enables segmentation aux loss
```

Optional PET (`--use-pet`) lives in a separate directory: `<pet-dir>/<patient_id>_*.nii.gz`. Patients without a PET file are skipped when PET is enabled.

> Patient IDs containing `A` are Adenocarcinoma, `B` Small Cell Carcinoma, `G` Squamous Cell Carcinoma. IDs containing `E` (Large Cell) are excluded.

**BigLunge** (fine-tune): patient folders named with digits, labels sourced from `patients_parameters.csv` via the `MorphologicalGroup` column. Any NIfTI file under a patient folder is used as a scan, except files containing `_label_` in their name.

Splits are **patient-level** and stratified (default 80/10/10) via `sklearn.model_selection.train_test_split`.

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- CUDA 12.x GPU (recommended)
- Conda

### Installation

```bash
git clone https://github.com/Hansstem/SCLC-Classification.git
cd SCLC-Classification

conda env create -f environment.yaml
conda activate sclc
```

Docker:

```bash
docker build -t sclc-classification .
docker run --gpus all -it --rm \
    --shm-size=8g \
    -v /path/to/SCLC-Classification:/workspace/SCLC-Classification \
    -v /path/to/data:/workspace/data \
    sclc-classification
```

### Data formats

NIfTI (`.nii`, `.nii.gz`). Volumes should contain Hounsfield Units for CT. PET volumes are rescaled to `[0, 1]`.

## 💻 Usage

### Full pipeline (DAPT → fine-tune → test)

```bash
python main.py --mode full \
    --model-type swin_unetr \
    --dapt-dataset /path/to/Lung-PET-CT-Dx-Clean \
    --finetune-dataset /path/to/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1 \
    --finetune-csv /path/to/BigLunge/patients_parameters.csv
```

### DAPT only

```bash
python main.py --mode dapt \
    --model-type swin_unetr \
    --dapt-dataset /path/to/Lung-PET-CT-Dx-Clean \
    --dapt-epochs 40 --dapt-lr 1e-4
```

### Fine-tune only (from an existing DAPT checkpoint)

```bash
python main.py --mode finetune \
    --pretrained-checkpoint /path/to/best_swin_unetr_dapt_new.pth \
    --finetune-dataset /path/to/BigLunge/.../1 \
    --finetune-csv /path/to/BigLunge/patients_parameters.csv \
    --finetune-epochs 40 --finetune-lr 3e-5
```

### Inference only

```bash
python main.py --mode inference \
    --model-checkpoint /path/to/best_swin_unetr_finetune_new.pth \
    --finetune-dataset /path/to/BigLunge/.../1 \
    --finetune-csv /path/to/BigLunge/patients_parameters.csv
```

### Enabling PET (CT + PET two-channel input)

```bash
python main.py --mode full --use-pet \
    --dapt-dataset /path/to/Lung-PET-CT-Dx-Clean \
    --pet-dir /path/to/Lung-PET-CT-Dx_PET
```

The first-conv / patch-embed layer is automatically adapted from 1 → 2 channels when loading single-channel pretrained weights.

### Quick smoke test

`--testing` truncates each split to a handful of samples so you can validate wiring end-to-end in a few minutes:

```bash
python main.py --mode full --testing
```

### Command-line arguments

#### Mode & model

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | str | `full` | `full`, `dapt`, `finetune`, or `inference` |
| `--model-type` | str | `swin_unetr` | `swin_unetr`, `resnet50`, `resnet18`, `vit`, `densenet121`, `models_genesis` |
| `--depth-size` | int | `128` | Number of Z-slices extracted per volume |
| `--anno` | flag | `True` | Use lesion masks for auxiliary segmentation loss |

#### Datasets

| Argument | Default | Description |
|---|---|---|
| `--dapt-dataset` | `/home/data/Lung-PET-CT-Dx-Clean` | DAPT dataset root |
| `--pet-dir` | `/home/data/Lung-PET-CT-Dx_PET` | PET NIfTI directory |
| `--use-pet` | `False` | Enable CT+PET 2-channel input |
| `--finetune-dataset` | `/home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1` | Fine-tune dataset root |
| `--finetune-csv` | `/home/data/BigLunge/patients_parameters.csv` | BigLunge labels CSV |

#### Checkpoints

| Argument | Description |
|---|---|
| `--initial-checkpoint` | Initial backbone weights. Defaults to `/home/data/temp/model_swin_unetr_btcv_segmentation_v1.pt` when `--model-type swin_unetr` |
| `--pretrained-checkpoint` | DAPT checkpoint to load in `finetune` mode |
| `--model-checkpoint` | Final model checkpoint for `inference` mode |
| `--checkpoint-dir` | `/home/data/trained_models` | Where best/periodic checkpoints are saved |

#### Training hyperparameters

| Argument | Default |
|---|---|
| `--dapt-epochs` | `40` |
| `--dapt-lr` | `1e-4` |
| `--finetune-epochs` | `40` |
| `--finetune-lr` | `3e-5` |
| `--batch-size` | `2` |
| `--accumulation-steps` | `4` |
| `--weight-decay` | `1e-3` |
| `--num-workers` | `4` |
| `--seed` | `42` |
| `--disable-amp` | off |

## 📊 Checkpoints

Saved under `--checkpoint-dir`:

- `best_{model_type}_{phase}_new.pth` — state dict of the best validation-accuracy model for each phase (`dapt`, `finetune`)
- `{phase}_{model_type}_epoch_{N}_new.pth` — periodic state-dict snapshots every 10 epochs

All saves are `model.state_dict()` only (no optimizer/scheduler state).

## 📦 Dependencies

Core:

- `torch >= 2.0.0`, `torchvision`
- `monai` — transforms, datasets, networks (SwinUNETR, ResNet, ViT, DenseNet)
- `nibabel >= 5.0.0`, `scipy`, `numpy == 1.26.4`
- `scikit-learn` — stratified patient-level splits
- `pandas` — BigLunge CSV parsing
- `tqdm`

Medical imaging (optional): `pydicom`, `dicom2nifti`.

## 🖥️ Running long jobs with tmux

```bash
tmux
conda activate /home/hansstem/anaconda3/envs/sclc
python main.py --mode full --model-type swin_unetr
# Detach: Ctrl+B, D
# Re-attach: tmux attach -t 0
```

## 📝 License

This project builds upon the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) codebase, licensed under the MIT License.

## 🙏 Acknowledgments

- Our Master's thesis supervisor, Boban Vesin, for guidance and support
- Researchers at SINTEF and St. Olavs Hospital for medical and technical insight
- Project MONAI for the 3D imaging transforms and networks
- Microsoft Research for the Swin Transformer architecture
- MedicalNet and Models Genesis for 3D pretrained weights
