import time
import torch
import torch.nn as nn
import numpy as np
from monai.metrics import ConfusionMatrixMetric
from monai.losses import DiceLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def simple_collate_fn(batch):
    # Extract the volume and the single target class logic
    scans = torch.stack([item["image"] for item in batch], dim=0)
    
    # We want a 1D tensor of class indices for CrossEntropyLoss
    labels = torch.tensor([item["scan_label"] for item in batch], dtype=torch.long)
    
    # Extract segmentation masks if available, filling missing ones with zeros
    masks = None
    if any("mask" in item for item in batch):
        # Find the shape and type of the first valid mask in the batch
        mask_shape = None
        mask_dtype = None
        mask_device = None
        for item in batch:
            if "mask" in item:
                mask_shape = item["mask"].shape
                mask_dtype = item["mask"].dtype
                mask_device = item["mask"].device
                break
                
        # Stack masks, generating an empty (all zeros) mask for items without one
        masks_list = []
        has_mask_list = []
        for item in batch:
            if "mask" in item:
                # Also check to make sure it's not an empty mask (all zeros)
                is_empty = (item["mask"].sum() == 0)
                masks_list.append(item["mask"])
                has_mask_list.append(not is_empty)
            else:
                masks_list.append(torch.zeros(mask_shape, dtype=mask_dtype, device=mask_device))
                has_mask_list.append(False)
                
        masks = torch.stack(masks_list, dim=0)
        has_mask = torch.tensor(has_mask_list, dtype=torch.bool)
        
        return scans, labels, masks, has_mask
    
    return scans, labels, None, None


def train_epoch(model, loader, optimizer, epoch, device, logger, scaler=None, use_segmentation=False, class_weights=None, accumulation_steps=1):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_cls_loss = AverageMeter()
    run_seg_loss = AverageMeter()
    
    # Apply label smoothing due to visual overlap in NSCLC/SCLC subtypes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer.zero_grad()  # 1. Zero gradients before the loop starts

    for idx, batch_data in enumerate(loader):
        has_mask_tensor = None
        if len(batch_data) == 4:
            data, target, masks, has_mask_tensor = batch_data
            masks = masks.to(device) if masks is not None else None
        elif len(batch_data) == 3:
            data, target, masks = batch_data
            masks = masks.to(device) if masks is not None else None
        else:
            data, target = batch_data
            masks = None
            
        data, target = data.to(device), target.to(device)
        
        with torch.amp.autocast(enabled=(scaler is not None), device_type='cuda'):
            if use_segmentation:
                logits, seg_outputs = model(data, return_segmentation=True)
                cls_loss = criterion(logits, target)
                
                seg_loss_val = 0.0
                if masks is not None and has_mask_tensor is not None and has_mask_tensor.any():
                    valid_mask_indices = has_mask_tensor.to(device)
                    masks = masks.float()  # Make sure masks are float
                    # Only compute BCE on samples that genuinely have a mask
                    seg_loss = nn.functional.binary_cross_entropy_with_logits(
                        seg_outputs[valid_mask_indices], 
                        masks[valid_mask_indices]
                    )
                    loss = cls_loss + (0.5 * seg_loss)
                    seg_loss_val = seg_loss.item()
                else:
                    loss = cls_loss
            else:
                logits = model(data, return_segmentation=False)
                loss = criterion(logits, target)
                cls_loss = loss
                seg_loss_val = 0.0
            
            unscaled_loss = loss.item()  # Save true loss value for metric tracking
            loss = loss / accumulation_steps  # 2. Scale the loss down

        # 3. Accumulate gradients (no optimizer.zero_grad() before this!)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # 4. Only step the optimizer after 'accumulation_steps' batches OR at the end of the loader
        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()  # Zero gradients AFTER stepping
            
        run_loss.update(unscaled_loss, n=data.size(0))
        run_cls_loss.update(cls_loss.item(), n=data.size(0))
        run_seg_loss.update(seg_loss_val, n=data.size(0))
        
        # Log every 10 steps and on the last step
        if (idx + 1) % 10 == 0 or (idx + 1) == len(loader):
            if use_segmentation:
                msg = (f"Epoch {epoch} [{idx + 1}/{len(loader)}] "
                       f"Loss: {run_loss.avg:.4f} (Cls: {run_cls_loss.avg:.4f} / Seg: {run_seg_loss.avg:.4f}) "
                       f"Time: {time.time() - start_time:.2f}s")
            else:
                msg = (f"Epoch {epoch} [{idx + 1}/{len(loader)}] "
                       f"Loss: {run_loss.avg:.4f} "
                       f"Time: {time.time() - start_time:.2f}s")
            print(msg)
            logger.info(msg)
            start_time = time.time()
            
    return run_loss.avg


@torch.no_grad()
def validate_epoch(model, loader, device, logger):
    model.eval()
    run_loss = AverageMeter()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    all_preds = []
    all_targets = []
    
    print("\nStarting validation...")
    logger.info("Starting validation...")

    for batch_data in loader:
        if len(batch_data) == 4:
            data, target, _, _ = batch_data
        elif len(batch_data) == 3:
            data, target, _ = batch_data
        else:
            data, target = batch_data
            
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = criterion(logits, target)
        
        run_loss.update(loss.item(), n=data.size(0))
        
        # Calculate Classification Accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate precision, recall, and f1 (dice for classification)
    if len(all_targets) > 0:
        metric = ConfusionMatrixMetric(
            include_background=True,
            metric_name=["precision", "recall", "f1 score"], 
            compute_sample=False, 
            reduction="mean"
        )
        all_targets_t = torch.tensor(all_targets, dtype=torch.long)
        all_preds_t = torch.tensor(all_preds, dtype=torch.long)
        
        max_observed = max(all_targets_t.max().item(), all_preds_t.max().item()) + 1
        num_classes = max(3, max_observed)
        
        target_onehot = nn.functional.one_hot(all_targets_t, num_classes=num_classes)
        pred_onehot = nn.functional.one_hot(all_preds_t, num_classes=num_classes)
        
        metric(y_pred=pred_onehot, y=target_onehot)
        metrics_res = metric.aggregate()
        
        precision = metrics_res[0].item() if not torch.isnan(metrics_res[0]) else 0.0
        recall = metrics_res[1].item() if not torch.isnan(metrics_res[1]) else 0.0
        f1 = metrics_res[2].item() if not torch.isnan(metrics_res[2]) else 0.0
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
        num_classes = 3
        
    val_msg = f"Validation Complete => Loss: {run_loss.avg:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Dice/F1: {f1:.4f}"
    print(val_msg)
    logger.info(val_msg)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        mem_msg = f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        print(mem_msg)
        logger.info(mem_msg)
    
    # Generate and print confusion matrix
    if len(all_targets) > 0:
        names = ["Adenocarcinoma", "Small Cell", "Squamous"]
        
        # Ensure we cover at least 3 classes
        max_observed = max(max(all_targets), max(all_preds)) + 1
        num_classes = max(3, max_observed)
            
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_targets, all_preds):
            conf_matrix[t, p] += 1
            
        print("\nConfusion Matrix (Rows: Actual Class, Columns: Predicted Class):")
        logger.info("\nConfusion Matrix (Rows: Actual Class, Columns: Predicted Class):")
        
        # Format names safely based on indices 
        display_names = [names[i] if i < len(names) else f"Class{i}" for i in range(num_classes)]
        
        # Header row (Columns = Model's Predictions)
        header = f"{'True \\ Pred':<18}" + "".join([f"{display_names[j]:<16}" for j in range(num_classes)])
        print(header)
        logger.info(header)
        
        # Rows (Rows = Actual Ground Truth)
        for i in range(num_classes):
            row_str = f"Y:{display_names[i]:<16}"
            for j in range(num_classes):
                # conf_matrix[True, Pred]
                count = conf_matrix[i, j]
                row_str += f"{count:<16}"
            print(row_str)
            logger.info(row_str)
        print("\n")
    
    # Return metrics as a dict (matching what main.py expects)
    return {"loss": run_loss.avg, "accuracy": accuracy}
