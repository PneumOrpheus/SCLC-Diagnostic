# SCLC-Classification

A deep learning diagnostic system for **lung cancer subtype classification** with a special focus on **Small Cell Lung Cancer (SCLC)** using CT scan data. The system implements a dual-head architecture combining object detection with global image classification, supporting multiple state-of-the-art backbone models including Swin Transformer V2.

## 🔬 Overview

This project provides an end-to-end pipeline for:
- **Domain-Adaptive Pre-Training (DAPT)** on Lung-PET-CT-Dx to learn lung CT-specific features
- **Fine-tuning** on the BigLunge dataset for target classification
- **CT scan preprocessing** with multi-channel windowing optimized for lung imaging
- **Lesion detection** using Faster R-CNN with Feature Pyramid Networks (FPN)
- **Global classification** for image-level lung cancer subtype diagnosis
- **Transfer learning** from RadImageNet-pretrained or ImageNet-pretrained models

### Target Classes

The model classifies CT scans into **3 lung cancer subtypes**, aligned across both the DAPT and fine-tuning datasets:

| Index | Class | DAPT Label (Lung-PET-CT-Dx) | Fine-tune Label (BigLunge) |
|-------|-------|------------------------------|----------------------------|
| 0 | Adenocarcinoma | `A` | `Adenokarsinom` |
| 1 | Small Cell Carcinoma | `B` | `Småcelletkarsinom` |
| 2 | Squamous Cell Carcinoma | `G` | `Plateepitelkarsinom` |

> Irrelevant classes (e.g., Large Cell Carcinoma, Non-small Cell NOS) are excluded during data loading to ensure consistent class alignment between both training phases.

## 🏗️ Architecture

### Dual-Head Model

The system uses a composite architecture (`DualHeadSCLCModel`) that combines:

1. **Flexible Backbone** - Supports multiple architectures:
   - `swinv2` - Swin Transformer V2 (default, recommended)
   - `swin` - Swin Transformer V1
   - `resnet50` - ResNet-50
   - `densenet121` - DenseNet-121

2. **Feature Pyramid Network (FPN)** - Multi-scale feature extraction dynamically adapted to backbone output channels

3. **Detection Head** - Faster R-CNN for lesion localization with:
   - Region Proposal Network (RPN)
   - Multi-scale ROI pooling

4. **Global Classification Head** - For image-level cancer subtype classification with:
   - Class-weighted cross-entropy loss (inverse-frequency weighting)
   - Label smoothing (0.1) to prevent overconfident predictions and class collapse
   - Automatic weight computation from training data distribution

### Training Pipeline

The training follows a two-phase transfer learning strategy:

```
RadImageNet Weights ──► DAPT (Lung-PET-CT-Dx) ──► Fine-tune (BigLunge) ──► Test
```

**Phase 1 — Domain-Adaptive Pre-Training (DAPT):**
- Trains backbone + FPN + detection head on Lung-PET-CT-Dx (global classifier frozen)
- **Real bounding box annotations** loaded from PASCAL VOC XML files and aggregated into per-patient union boxes
- 30 epochs default, learning rate 1e-4
- Linear warmup (5 epochs) followed by cosine annealing
- Early stopping with configurable patience

**Phase 2 — Fine-tuning on BigLunge:**
- Unfreezes all layers for full model training
- 50 epochs default, base learning rate 5e-5
- **Differential learning rates**: backbone at 0.1× base LR, FPN/heads at 1× base LR
- Linear warmup (5 epochs) followed by cosine annealing
- Early stopping with configurable patience
- Class-weighted loss recomputed for the fine-tuning dataset

### Data Augmentation

Training transforms include spatial and intensity augmentations (validation/test use no augmentation):

| Augmentation | Parameters | Probability |
|---|---|---|
| Random Horizontal Flip | — | 0.5 |
| Random Vertical Flip | — | 0.5 |
| Random 90° Rotation | axes (0,1) | 0.5 |
| Random Affine | rotate ±0.15 rad, scale ±0.1, translate ±10 px | 0.3 |
| Random Scale Intensity | factors ±0.1 | 0.5 |
| Random Shift Intensity | offsets ±0.1 | 0.5 |
| Random Gaussian Noise | std=0.02 | 0.2 |
| Random Adjust Contrast | gamma 0.8–1.2 | 0.3 |

### CT Preprocessing Pipeline

The preprocessing module supports specialized multi-channel windowing for CT scans:

| Channel | Window Center | Window Width | Purpose |
|---------|---------------|--------------|---------|
| Lung | -600 HU | 1500 HU | Nodules and parenchyma |
| Mediastinal | 50 HU | 350 HU | Lymph nodes and soft tissue |
| Bone/Wide | 300 HU | 2000 HU | Chest wall and spine context |

## 📁 Project Structure

```
data/ 
├── Lung-PET-CT-Dx 
│   └── {patient_id}_{series_uid}.nii.gz
SCLC-Classification/
├── main.py                     # Main pipeline (DAPT → Fine-tune → Inference)
├── logger.py                   # Logging utilities
├── lr_scheduler.py             # Learning rate schedulers
├── optimizer.py                # Optimizer utilities
├── data/
│   ├── biglunge_loader.py      # BigLunge dataset loader with class mapping
│   ├── data_preprocessing.py   # CT preprocessing utilities
│   └── transforms.py           # MONAI transforms (train/val/test pipelines)
├── models/
│   ├── build.py                # Model builder
│   ├── config.py               # Configuration management (YACS)
│   ├── model_selection.py      # Backbone & dual-head architecture
│   ├── swin_transformer.py     # Swin Transformer V1
│   └── swin_transformer_v2.py  # Swin Transformer V2
├── training/
│   └── train.py                # Training/validation epoch functions & dataset creation
├── kernels/
│   └── window_process/         # CUDA kernels for Swin window operations
├── environment.yaml            # Conda environment specification
├── requirements.txt            # Pip dependencies
└── Dockerfile                  # Docker container setup
```

### Lung-PET-CT-Dx dataset
Patients with Names/IDs containing the letter 'A' were diagnosed with Adenocarcinoma, 'B' with Small Cell Carcinoma, 'E' with Large Cell Carcinoma, and 'G' with Squamous Cell Carcinoma.

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- CUDA 12.x compatible GPU (recommended)
- Conda (for environment management)

### Installation

#### Option 1: Conda Environment

```bash
# Clone the repository
git clone https://github.com/Hansstem/SCLC-Classification.git
cd SCLC-Classification

# Create conda environment
conda env create -f environment.yaml
conda activate sclc

# For GPU support, install CUDA toolkit
conda install -c nvidia cuda-nvcc cuda-cudart "cuda-version=12.*"
```

#### Option 2: Docker

```bash
# Build the Docker image
docker build -t sclc-classification .

# Run container with GPU support
docker run --gpus all -it --rm \
    --shm-size=8g \
    -v /path/to/repo/location/SCLC-Classification:/workspace/SCLC-Classification \
    -v /path/to/your/data:/workspace/data \
    sclc-classification
```

### Tmux
To run long training sessions without interruption, we recommend using `tmux` to create a persistent terminal session.

Example workflow:

    tmux 
    conda activate /path/to/your/conda/environment
    python main.py \
    --backbone swinv2 \
    --config /path/to/rin_config.yaml \
    --checkpoint /path/to/rin_swintf.pth \
    --data-path /path/to/training/data \
    --epochs 20

Then you can detach from the session by Ctrl + B , D
To attach to the session again:

    tmux ls
    tmux attach -t 0

### Data Format

The system supports the following input formats:
- **NIfTI**: `.nii`, `.nii.gz`
- **NumPy**: `.npy`, `.npz`

Place your CT scan files in a data directory. Files should contain 3D volumetric data in Hounsfield Units.

## 💻 Usage

### Full Pipeline (DAPT + Fine-tune + Test)

Runs the complete training pipeline: domain-adaptive pre-training on Lung-PET-CT-Dx, fine-tuning on BigLunge, and test set evaluation.

```bash
python main.py --mode full \
    --backbone swinv2 \
    --dapt-backbone-dataset /path/to/Lung-PET-CT-Dx \
    --fine-tuning-dataset /path/to/BigLunge/data \
    --fine-tuning-csv /path/to/BigLunge/patients_parameters.csv \
    --annotation-dir /path/to/Annotation
```

### DAPT Only (Backbone Pre-training)

Pre-trains the backbone on Lung-PET-CT-Dx and saves the checkpoint.

```bash
python main.py --mode dapt \
    --backbone swinv2 \
    --dapt-backbone-dataset /path/to/Lung-PET-CT-Dx \
    --dapt-epochs 30 \
    --dapt-lr 1e-4
```

### Fine-tune Only (Requires Pre-trained Checkpoint)

Fine-tunes a pre-trained model on BigLunge.

```bash
python main.py --mode finetune \
    --backbone swinv2 \
    --pretrained-checkpoint /path/to/dapt_swinv2_best.pth \
    --fine-tuning-dataset /path/to/BigLunge/data \
    --fine-tuning-csv /path/to/BigLunge/patients_parameters.csv \
    --finetune-epochs 50 \
    --finetune-lr 5e-5
```

### Inference Only

Runs inference on the BigLunge test set using a trained model.

```bash
python main.py --mode inference \
    --model-checkpoint /path/to/finetune_swinv2_best.pth \
    --fine-tuning-dataset /path/to/BigLunge/data \
    --fine-tuning-csv /path/to/BigLunge/patients_parameters.csv
```

### Command-Line Arguments

#### Mode & Model

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `full` | Pipeline mode: `full`, `dapt`, `finetune`, or `inference` |
| `--backbone` | str | `swinv2` | Backbone architecture: `swin`, `swinv2`, `resnet50`, `densenet121` |
| `--config` | str | *(RadImageNet config)* | Path to YAML config file for Swin models |
| `--initial-checkpoint` | str | *(RadImageNet weights)* | Path to initial backbone checkpoint |

#### Dataset Paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dapt-backbone-dataset` | str | `/home/data/Lung-PET-CT-Dx` | Path to DAPT dataset (Lung-PET-CT-Dx) |
| `--fine-tuning-dataset` | str | `/home/data/BigLunge/...` | Path to fine-tuning dataset (BigLunge) |
| `--fine-tuning-csv` | str | `/home/data/BigLunge/patients_parameters.csv` | Path to BigLunge patient labels CSV |
| `--pretrained-checkpoint` | str | `""` | Path to pre-trained checkpoint (for `finetune` mode) |
| `--model-checkpoint` | str | `""` | Path to final model checkpoint (for `inference` mode) |

#### Training Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dapt-epochs` | int | `30` | Number of DAPT epochs |
| `--dapt-lr` | float | `1e-4` | Learning rate for DAPT phase |
| `--finetune-epochs` | int | `50` | Number of fine-tuning epochs |
| `--finetune-lr` | float | `5e-5` | Base learning rate for fine-tuning (backbone gets 0.1×) |
| `--batch-size` | int | `8` | Batch size for training |
| `--weight-decay` | float | `0.05` | Weight decay for AdamW optimizer |
| `--patience` | int | `10` | Early stopping patience (epochs without val loss improvement) |
| `--num-workers` | int | `4` | Number of data loading workers |
| `--annotation-dir` | str | `/home/data/Annotation` | Path to annotation directory with per-patient XML bounding boxes |

#### Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `output` | Output directory for logs and results |
| `--checkpoint-dir` | str | `/home/data/trained_models` | Directory for saving model checkpoints |


### Tmux
````
    tmux 
    conda activate /home/hansstem/anaconda3/envs/sclc
    python main.py \
    --backbone swinv2 \
    --config /path/to/rin_config.yaml \
    --checkpoint /path/to/rin_swintf.pth \
    --data-path /path/to/training/data \
    --epochs 20

````
Then you can detach from the session by Ctrl + B , D
To attach to the session again
````
    tmux ls
    tmux attach -t 0
````


## 📦 Dependencies

### Core Dependencies

- `torch >= 2.0.0` - Deep learning framework
- `torchvision >= 0.17.0` - Computer vision models
- `timm >= 0.9.10` - Pretrained image models
- `monai` - Medical Open Network for AI (transforms, data loading)
- `numpy == 1.26.4` - Numerical computing
- `nibabel >= 5.0.0` - NIfTI file handling
- `scipy == 1.13.1` - Scientific computing
- `yacs >= 0.1.8` - Configuration management

### Medical Imaging

- `pydicom == 2.4.4` - DICOM file handling
- `dicom2nifti == 2.4.9` - DICOM to NIfTI conversion

## 📊 Checkpoints

Checkpoints are saved in two formats:

1. **Best model weights** (`dapt_{backbone}_best.pth`, `finetune_{backbone}_best.pth`): Model state dict for the best validation loss, used for inference
2. **Periodic full checkpoints** (every 5 epochs): Complete training state (model, optimizer, scheduler, best loss) for resuming interrupted training

## 🔧 Configuration

The system uses YACS for configuration management. Key configuration sections:

- `DATA`: Dataset settings (batch size, image size, augmentation)
- `MODEL`: Architecture settings (backbone type, number of classes, dropout)
- `TRAIN`: Training hyperparameters (optimizer, learning rate, scheduler)

## 📝 License

This project builds upon the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) codebase, licensed under the MIT License.

## 🙏 Acknowledgments

- Our Master's thesis supervisor, Boban Vesin, for guidance and support
- Researchers at SINTEF and St.Olavs Hospital for medical and technical insight and support
- Microsoft Research for the Swin Transformer architecture
- RadImageNet for medical imaging pretrained weights
- PyTorch and torchvision teams

