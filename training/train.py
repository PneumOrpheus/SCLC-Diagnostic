import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os

"""
SCLC Diagnostic System Training
===============================
Implements the training pipeline for the SCLC diagnostic system with support for:
- Backbone model selection
- Resuming from checkpoints for fine-tuning
- Multi-task loss aggregation
"""

def detection_collate_fn(batch):
    """Custom collate function for DataLoader that unzips (scan, target) pairs."""
    return tuple(zip(*batch))

class SCLCTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, img_size=224):
        # Initialize dataset
        self.data_path = data_path
        self.img_size = img_size

        # Validate that the data path exists and is a directory
        if not os.path.isdir(self.data_path):
            raise ValueError(f"Data path '{self.data_path}' does not exist or is not a directory.")

        try:
            all_files = os.listdir(self.data_path)
        except OSError as e:
            raise ValueError(f"Unable to list contents of data path '{self.data_path}': {e}") from e

        # Collect all valid sample for both NIfTI and numpy formats
        self.samples = [f for f in all_files if f.endswith('.nii.gz') or f.endswith('.nii') or f.endswith('.npy') or f.endswith('.npz')]

        if not self.samples:
            raise ValueError(
                f"No valid data files found in data path '{self.data_path}'. "
                "Supported formats: .nii.gz, .nii, .npy, .npz"
            )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.samples[idx])
        
        try:
            if path.endswith('.nii.gz') or path.endswith('.nii'):
                # Load NIfTI file
                nii_img = nib.load(path)
                scan_data = nii_img.get_fdata().astype(np.float32)
            else:
                # Load numpy file
                data = np.load(path, allow_pickle=True)
                if hasattr(data, "item"):
                    data_dict = data.item()
                    scan_data = data_dict['scan']
                else:
                    scan_data = data
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{path}': {e}") from e

        # Convert to tensor (C, H, W) for 2D or (C, D, H, W) for 3D
        scan = torch.tensor(scan_data, dtype=torch.float32)
        
        # Handle different dimensionalities
        if scan.ndim == 2: 
            # (H, W) -> (1, H, W)
            scan = scan.unsqueeze(0)
        elif scan.ndim == 3:
            # Distinguish 2D images with channels from true 3D volumes
            if scan.shape[-1] in (1, 3):
                # Likely channel-last image: (H, W, C) -> (C, H, W)
                scan = scan.permute(2, 0, 1)
            elif scan.shape[0] in (1, 3):
                # Likely channel-first image: (C, H, W), keep as-is
                pass
            else:
                # Likely 3D volume: select middle slice along the largest axis
                depth_axis = int(np.argmax(scan.shape))
                mid_slice = scan.shape[depth_axis] // 2
                scan = scan.select(dim=depth_axis, index=mid_slice).unsqueeze(0)
        
        # Normalize scan
        if scan.max() > 1.0:
            scan_min = scan.min()
            scan_max = scan.max()
            # Avoid dividing by a near-zero range for uniform scans
            if scan_max > scan_min:
                scan = (scan - scan_min) / (scan_max - scan_min)
            else:
                # For uniform scans, map to a stable constant (all zeros)
                scan = torch.zeros_like(scan)
        
        # TODO: Only apply for ImageNet-pretrained backbones, not for RadImageNet
        # Convert grayscale to RGB for compatibility with pretrained backbones
        if scan.shape[0] == 1:
            scan = scan.repeat(3, 1, 1)
        
        # Resize to model's expected input size
        scan = F.interpolate(
            scan.unsqueeze(0), 
            size=(self.img_size, self.img_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Create placeholder targets for NIfTI files without annotation data 
        targets = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'scan_label': torch.tensor(0, dtype=torch.int64),
            'scan_id': torch.tensor(idx, dtype=torch.int64),
        }
        return scan, targets       


def train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    
    for i, (scans, targets) in enumerate(data_loader):
        scans = list(scan.to(device) for scan in scans)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(scans, targets)
        
        # Loss aggregation
        global_loss = loss_dict.pop("global_classification_loss")

        # Ensure detection loss is always a tensor on the correct device,
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
        
        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Iteration [{i}/{len(data_loader)}], "
                  f"Total loss: {total_loss.item():.4f}, "
                  f"Detection loss: {loss_detection.item():.4f}, "
                  f"Global loss: {global_loss.item():.4f}")
