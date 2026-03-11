import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Any, Dict, List, Sequence, Optional

from monai.data import DataLoader, list_data_collate  # type: ignore[attr-defined]
from monai.transforms import Compose  # type: ignore[attr-defined]

from data.transforms import get_train_transforms, get_val_transforms


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
        
        # Weighted sum - global classification is the primary task
        total_loss = loss_detection + 1.0 * global_loss
        
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
            
        total_loss = loss_detection + 1.0 * global_loss
        
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
