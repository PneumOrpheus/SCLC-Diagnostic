import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Any, Dict, List, Sequence, Optional

from monai.data import CacheDataset, DataLoader, list_data_collate  # type: ignore[attr-defined]
from monai.transforms import Compose  # type: ignore[attr-defined]

# MONAI transforms for preprocessing
from data.transforms import get_train_transforms, get_val_transforms

"""
SCLC Diagnostic System Training
-------------------------------
Implements the training pipeline for the SCLC diagnostic system with support for:
- Backbone model selection
- Resuming from checkpoints for fine-tuning
- Multi-task loss aggregation
- MONAI CacheDataset for efficient data loading
"""


def sclc_collate_fn(batch: List[Dict[str, Any]]) -> tuple:
    """Custom collate function for detection-style training.
    
    Converts MONAI's dictionary batch format to the (scans, targets) tuple format
    expected by the detection model.
    
    Args:
        batch: List of dictionaries with 'image' and target keys.
        
    Returns:
        Tuple of (scans, targets) where scans is a list of tensors and
        targets is a list of dictionaries.
    """
    scans = []
    targets = []
    
    for i, item in enumerate(batch):
        scans.append(item["image"])

        # Ensure scan_label is a tensor
        scan_label = item.get("scan_label", 0)
        if not isinstance(scan_label, torch.Tensor):
            scan_label = torch.tensor(scan_label, dtype=torch.int64)

        target = {
            "boxes": item.get("boxes", torch.zeros((0, 4), dtype=torch.float32)),
            "labels": item.get("labels", torch.zeros((0,), dtype=torch.int64)),
            "scan_label": scan_label,
            "scan_id": torch.tensor(i, dtype=torch.int64),
        }
        targets.append(target)
    
    return tuple(scans), tuple(targets)


def get_data_list(
    data_path: str, 
    split: str = "train", 
    val_frac: float = 0.1, 
    test_frac: float = 0.1, 
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Create a list of data dictionaries for MONAI dataset with patient-level splitting.
    
    Args:
        data_path: Path to directory containing scan files.
        split: One of 'train', 'val', or 'test'.
        val_frac: Fraction of patients to use for validation.
        test_frac: Fraction of patients to use for testing.
        seed: Random seed for reproducibility.
        
    Returns:
        List of dictionaries with 'image' key pointing to file paths.
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")
    
    try:
        all_files = os.listdir(data_path)
    except OSError as e:
        raise ValueError(f"Unable to list contents of data path '{data_path}': {e}") from e
    
    # Collect all valid samples for NIfTI and numpy formats
    # TODO: Implement logic to prefer _Eq files if both versions exist
    valid_extensions = ('.nii.gz', '.nii', '.npy', '.npz')
    samples = [f for f in all_files if any(f.endswith(ext) for ext in valid_extensions)]
    
    if not samples:
        raise ValueError(
            f"No valid data files found in data path '{data_path}'. "
            f"Supported formats: {valid_extensions}"
        )
    
    # A: Adenocarcinoma (0), B: Small Cell Carcinoma (1), E: Large Cell Carcinoma (2), G: Squamous Cell Carcinoma (3)
    class_map = {'A': 0, 'B': 1, 'E': 2, 'G': 3}

    # Filename format: Lung_Dx-A0126_1.3.6.1... -> Patient ID: Lung_Dx-A0126
    patient_files = {}
    for f in samples:
        parts = f.split('_')
        if len(parts) >= 2:
            # Reconstruct "Lung_Dx-A0126"
            patient_id = f"{parts[0]}_{parts[1]}"
            if patient_id not in patient_files:
                patient_files[patient_id] = []
            patient_files[patient_id].append(f)

    all_patients = sorted(list(patient_files.keys()))
    
    # Shuffle patients deterministically
    rng = np.random.default_rng(seed)
    rng.shuffle(all_patients)
    
    # Calculate split indices
    n_total = len(all_patients)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val
    
    if split == 'train':
        selected_patients = set(all_patients[:n_train])
    elif split == 'val':
        selected_patients = set(all_patients[n_train:n_train+n_val])
    elif split == 'test':
        selected_patients = set(all_patients[n_train+n_val:])
    else:
        # Fallback to all if split name is unknown (e.g. for legacy behavior)
        selected_patients = set(all_patients)

    print(f"Data Split '{split}': {len(selected_patients)} patients.")

    # Create list of dictionaries for MONAI
    data_list = []
    
    for pid in selected_patients:
        files = patient_files.get(pid, [])
        for f in files:
            # Determine label from filename (e.g. Lung_Dx-A0126_...)
            label = -1
            for key, val in class_map.items():
                if f"-{key}" in f:
                    label = val
                    break
            
            # Adds to data list with image path and class label
            if label != -1:
                data_list.append({
                    "image": os.path.join(data_path, f),
                    "scan_label": label
                })
    
    print(f"  -> {len(data_list)} images found for split '{split}'.")
    return data_list


def create_dataset(
    data_path: str,
    split: str = "train",
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False,
    cache_rate: float = 1.0,
    num_workers: int = 4
) -> CacheDataset:
    """Create a MONAI CacheDataset.
    
    Args:
        data_path: Path to directory containing scan files.
        split: One of 'train', 'val', or 'test'.
        img_size: Target image size (height and width).
        convert_to_rgb: Whether to convert grayscale to 3-channel RGB.
        use_multichannel_windowing: Whether to use multi-channel CT windowing.
        cache_rate: Fraction of data to cache (0.0 to 1.0).
        num_workers: Number of workers for caching.
        
    Returns:
        CacheDataset configured for the specified split.
    """
    data_list = get_data_list(data_path, split=split)
    
    if split == "train":
        transforms = get_train_transforms(
            img_size=img_size,
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing
        )
    else:
        # val and test use val transforms (no augmentation)
        transforms = get_val_transforms(
            img_size=img_size,
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing
        )
    
    dataset = CacheDataset(
        data=data_list,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    
    return dataset


def train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0
    running_det_loss = 0.0
    running_global_loss = 0.0
    
    num_batches = len(data_loader)
    if num_batches == 0:
        print("Warning: Train data loader is empty.")
        return {"loss": 0.0, "det_loss": 0.0, "global_loss": 0.0}

    for i, (scans, targets) in enumerate(data_loader):
        scans = list(scan.to(device) for scan in scans)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(scans, targets)
        
        # Loss aggregation
        global_loss = loss_dict.pop("global_classification_loss")

        # Ensure detection loss is always a tensor on the correct device
        if loss_dict:
            detection_losses = []
            for loss in loss_dict.values():
                if isinstance(loss, torch.Tensor):
                    detection_losses.append(loss.to(device))
                else:
                    detection_losses.append(torch.as_tensor(loss, device=device))
            loss_detection = sum(detection_losses, torch.zeros((), device=device))
        else:
            loss_detection = torch.zeros((), device=device)
        
        # Weighted sum
        total_loss = loss_detection + 0.5 * global_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        running_det_loss += loss_detection.item()
        running_global_loss += global_loss.item()
        
        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Iteration [{i}/{len(data_loader)}], "
                  f"Total loss: {total_loss.item():.4f}, "
                  f"Detection loss: {loss_detection.item():.4f}, "
                  f"Global loss: {global_loss.item():.4f}")

    metrics = {
        "loss": running_loss / num_batches,
        "det_loss": running_det_loss / num_batches,
        "global_loss": running_global_loss / num_batches
    }
    return metrics


@torch.no_grad()
def validate_epoch(model, data_loader, device, phase="val"):
    """
    Computes validation loss. 
    NOTE: Sets model to train() mode with no_grad() because standard torchvision 
    detection models only return losses in train mode (and predictions in eval mode).
    """
    # Preserve original training state 
    was_training = model.training
    model.train() 
    
    running_loss = 0.0
    running_det_loss = 0.0
    running_global_loss = 0.0
    
    num_batches = len(data_loader)
    if num_batches == 0:
        print(f"Warning: {phase} data loader is empty.")
        model.train(was_training)
        return {"loss": 0.0, "det_loss": 0.0, "global_loss": 0.0}
    
    print(f"Starting {phase} evaluation...")
    
    for scans, targets in data_loader:
        scans = list(scan.to(device) for scan in scans)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(scans, targets)
        
        global_loss = loss_dict.pop("global_classification_loss")

        if loss_dict:
            detection_losses = [l.to(device) if isinstance(l, torch.Tensor) else torch.as_tensor(l, device=device) for l in loss_dict.values()]
            loss_detection = sum(detection_losses, torch.zeros((), device=device))
        else:
            loss_detection = torch.zeros((), device=device)
            
        total_loss = loss_detection + 0.5 * global_loss
        
        running_loss += total_loss.item()
        running_det_loss += loss_detection.item()
        running_global_loss += global_loss.item()
        
    avg_loss = running_loss / num_batches
    avg_det = running_det_loss / num_batches
    avg_global = running_global_loss / num_batches
    
    print(f"  {phase.capitalize()} Loss: {avg_loss:.4f} (Det: {avg_det:.4f}, Global: {avg_global:.4f})")
    
    # Restore model state
    model.train(was_training)
    
    return {"loss": avg_loss, "det_loss": avg_det, "global_loss": avg_global}
