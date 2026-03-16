import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Any, Dict, List, Sequence, Optional

# MONAI transforms for preprocessing
from models.model_selection import focal_loss

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

def train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                use_mixup=False, mixup_alpha=0.2,
                scaler=None, accumulation_steps=1, clip_grad=1.0,
                label_smoothing=0.0):
    """Train one epoch with optional Mixup augmentation, AMP, and gradient accumulation.
    
    Args:
        model: The model to train.
        optimizer: Optimizer.
        data_loader: Training data loader.
        device: Device to train on.
        epoch: Current epoch number.
        print_freq: Print frequency.
        use_mixup: Whether to apply Mixup data augmentation.
        mixup_alpha: Beta distribution parameter for Mixup.
        scaler: GradScaler for AMP (None = no AMP).
        accumulation_steps: Number of mini-batches to accumulate before stepping.
        clip_grad: Max gradient norm for clipping (0 = disabled).
        label_smoothing: Label smoothing factor (0.0 = disabled).
    """
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
        
        batch_size = len(scans)
        apply_mixup = use_mixup and batch_size > 1 and np.random.random() < 0.5
        
        with torch.amp.autocast(enabled=amp_enabled, device_type=device.type):
            if apply_mixup:
                # Blend pairs of images and compute mixed loss
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(batch_size)
                
                mixed_scans = [lam * scans[j] + (1 - lam) * scans[index[j]] for j in range(batch_size)]
                
                # Forward with mixed images, request logits for proper mixup loss
                loss_dict = model(mixed_scans, targets, return_logits=True)
                
                global_logits = loss_dict.pop("global_logits")
                _ = loss_dict.pop("global_classification_loss")
                
                # Compute proper mixup focal loss
                gt_a = torch.stack([targets[j]["scan_label"] for j in range(batch_size)])
                gt_b = torch.stack([targets[index[j]]["scan_label"] for j in range(batch_size)])
                
                class_w = model.class_weights if hasattr(model, 'class_weights') else None
                global_loss = (
                    lam * focal_loss(global_logits, gt_a, alpha=class_w) +
                    (1 - lam) * focal_loss(global_logits, gt_b, alpha=class_w)
                )
            else:
                # Request logits when label smoothing is needed
                need_logits = label_smoothing > 0.0
                loss_dict = model(scans, targets, return_logits=need_logits)
                
                if need_logits:
                    global_logits = loss_dict.pop("global_logits")
                    _ = loss_dict.pop("global_classification_loss")
                    
                    # Compute label-smoothed loss via cross-entropy
                    gt_labels = torch.stack([t["scan_label"] for t in targets])
                    num_classes = global_logits.size(-1)
                    log_probs = torch.log_softmax(global_logits, dim=-1)
                    
                    # One-hot targets blended with uniform distribution
                    one_hot = torch.zeros_like(log_probs).scatter_(1, gt_labels.unsqueeze(1), 1.0)
                    smooth_targets = (1.0 - label_smoothing) * one_hot + label_smoothing / num_classes
                    global_loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
                else:
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
            total_loss = loss_detection + global_loss
            
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
            extra = " [mixup]" if apply_mixup else ""
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == "cuda" else 0
            print(f"Epoch [{epoch}], Iteration [{i}/{num_batches}]{extra}, "
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
        scans = [scan.to(device, non_blocking=True) for scan in scans]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
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
