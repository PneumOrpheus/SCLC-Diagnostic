# SCLC-Classification

A deep learning diagnostic system for **Small Cell Lung Cancer (SCLC)** detection and classification using CT scan data. The system implements a dual-head architecture combining object detection with global image classification, supporting multiple state-of-the-art backbone models including Swin Transformer V2.

## 🔬 Overview

This project provides an end-to-end pipeline for:
- **CT scan preprocessing** with multi-channel windowing optimized for lung imaging
- **Lesion detection** using Faster R-CNN with Feature Pyramid Networks (FPN)
- **Global classification** for image-level SCLC diagnosis
- **Transfer learning** from RadImageNet-pretrained or ImageNet-pretrained models

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

4. **Global Classification Head** - For image-level SCLC classification

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
├── main.py                 # Training entry point
├── data/
│   └── data_preprocessing.py   # CT preprocessing utilities
├── models/
│   ├── build.py            # Model builder
│   ├── config.py           # Configuration management
│   ├── model_selection.py  # Backbone & dual-head architecture
│   ├── swin_transformer.py # Swin Transformer V1
│   └── swin_transformer_v2.py  # Swin Transformer V2
├── training/
│   └── train.py            # Training loop and dataset
├── kernels/
│   └── window_process/     # CUDA kernels for Swin window operations
├── environment.yaml        # Conda environment specification
├── requirements.txt        # Pip dependencies
└── Dockerfile              # Docker container setup
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

### Data Format

The system supports the following input formats:
- **NIfTI**: `.nii`, `.nii.gz`
- **NumPy**: `.npy`, `.npz`

Place your CT scan files in a data directory. Files should contain 3D volumetric data in Hounsfield Units.

## 💻 Usage

### Training

```bash
python main.py \
    --backbone swinv2 \
    --data-path /path/to/training/data \
    --epochs 20 \
    --batch-size 8 \
    --lr 1e-4
```

### Training with RadImageNet Pretrained Weights

To use RadImageNet-pretrained Swin Transformer weights for better medical imaging transfer learning:

```bash
python main.py \
    --backbone swinv2 \
    --config /path/to/rin_config.yaml \
    --checkpoint /path/to/rin_swintf.pth \
    --data-path /path/to/training/data \
    --epochs 20
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backbone` | str | `swinv2` | Backbone model: `swin`, `swinv2`, `resnet50`, `densenet121` |
| `--data-path` | str | `""` | Path to training data directory |
| `--checkpoint` | str | `""` | Path to pretrained `.pth` checkpoint file |
| `--config` | str | `""` | Path to YAML config file for Swin models |
| `--epochs` | int | `20` | Number of training epochs |
| `--batch-size` | int | `8` | Batch size for training |
| `--lr` | float | `1e-4` | Learning rate |

## 📦 Dependencies

### Core Dependencies

- `torch >= 2.0.0` - Deep learning framework
- `torchvision >= 0.17.0` - Computer vision models
- `timm >= 0.9.10` - Pretrained image models
- `numpy == 1.26.4` - Numerical computing
- `nibabel >= 5.0.0` - NIfTI file handling
- `scipy == 1.13.1` - Scientific computing
- `yacs >= 0.1.8` - Configuration management

### Medical Imaging

- `pydicom == 2.4.4` - DICOM file handling
- `dicom2nifti == 2.4.9` - DICOM to NIfTI conversion

## 📊 Checkpoints

Training checkpoints are saved in two formats:

1. **Weights only** (`checkpoint_weights/`): Model state dict for inference
2. **Full checkpoint** (`full_checkpoints/`): Complete training state for resuming

Checkpoints are saved every 5 epochs.

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

