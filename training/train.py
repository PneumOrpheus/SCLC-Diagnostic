import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Any, Dict, List, Sequence, Optional

# MONAI transforms for preprocessing

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

        boxes = item.get("boxes", [])
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)

        labels = item.get("labels", [])
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "scan_label": scan_label,
            "scan_id": torch.tensor(i, dtype=torch.int64),
        }
        targets.append(target)
    
    return tuple(scans), tuple(targets)

def train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                use_mixup=False, mixup_alpha=0.2,
                scaler=None, accumulation_steps=1, clip_grad=1.0):
    """Train one epoch with AMP, and gradient accumulation.
    (Note: Mixup is disabled as it is structurally incompatible with Object Detection bounded targets)
    """
    if use_mixup:
        print("Warning: Mixup is disabled. It is incompatible with bounding box targets in Dual-Head networks.")
        
    model.train()
    running_loss = 0.0
    running_det_loss = 0.0
    running_global_loss = 0.0
    running_grad_norm = 0.0
    
    amp_enabled = scaler is not None and scaler.is_enabled()
    last_grad_norm = 0.0
    
    num_batches = len(data_loader)
    if num_batches == 0:
        print("Warning: Train data loader is empty.")
        return {"loss": 0.0, "det_loss": 0.0, "global_loss": 0.0, "grad_norm": 0.0}

    optimizer.zero_grad()

    for i, (scans, targets) in enumerate(data_loader):
        scans = [scan.to(device, non_blocking=True) for scan in scans]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        with torch.amp.autocast(enabled=amp_enabled, device_type=device.type):
            
            loss_dict = model(scans, targets)
            
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
            
            # --- DYNAMIC RE-WEIGHTING LOSSES ---
            # Warmup Schedule: start with 0 for classification and gradually increase
            # to let the detector learn spatial representation first.
            warmup_epochs = 5
            ramp_factor = (epoch - 1) / warmup_epochs if epoch <= warmup_epochs else 1.0
            
            det_weight = 0.25
            cls_weight = 2.0 * ramp_factor
            
            total_loss = (det_weight * loss_detection) + (cls_weight * global_loss)
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / accumulation_steps
        
        # Backward pass with AMP
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Step optimizer every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            # Gradient clipping + norm monitoring
            if clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            last_grad_norm = grad_norm.item()
            running_grad_norm += last_grad_norm
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Statistics
        running_loss += total_loss.item() * accumulation_steps
        running_det_loss += loss_detection.item()
        running_global_loss += global_loss.item()
        
        if i % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == "cuda" else 0
            print(f"Epoch [{epoch}], Iteration [{i}/{num_batches}], "
                  f"Total loss: {total_loss.item() * accumulation_steps:.4f}, "
                  f"Detection loss: {loss_detection.item():.4f}, "
                  f"Global loss: {global_loss.item():.4f}, "
                  f"Grad norm: {last_grad_norm:.4f}, "
                  f"Mem: {memory_used:.0f}MB")

    num_steps = (num_batches + accumulation_steps - 1) // accumulation_steps
    metrics = {
        "loss": running_loss / num_batches,
        "det_loss": running_det_loss / num_batches,
        "global_loss": running_global_loss / num_batches,
        "grad_norm": running_grad_norm / max(num_steps, 1),
    }
    return metrics


@torch.no_grad()
def validate_epoch(model, data_loader, device, epoch=None, phase="val"):
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
        scans = [scan.to(device, non_blocking=True) for scan in scans]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        loss_dict = model(scans, targets)
        
        global_loss = loss_dict.pop("global_classification_loss")

        if loss_dict:
            detection_losses = [l.to(device) if isinstance(l, torch.Tensor) else torch.as_tensor(l, device=device) for l in loss_dict.values()]
            loss_detection = sum(detection_losses, torch.zeros((), device=device))
        else:
            loss_detection = torch.zeros((), device=device)
            
        # Match training weights so eval metrics align
        warmup_epochs = 5
        ramp_factor = (epoch - 1) / warmup_epochs if epoch is not None and epoch <= warmup_epochs else 1.0
        
        total_loss = (0.25 * loss_detection) + (2.0 * ramp_factor * global_loss)
        
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
