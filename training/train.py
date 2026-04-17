import time
import torch
import torch.nn as nn
import numpy as np
from monai.losses import DiceLoss

NUM_CLASSES = 3
CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]

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


def _compute_classification_metrics(targets, preds, min_num_classes=NUM_CLASSES):
    """Compute confusion-matrix-derived metrics for multiclass single-label data."""
    if len(targets) == 0 or len(preds) == 0:
        num_classes = int(min_num_classes)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    else:
        max_observed = int(max(max(targets), max(preds))) + 1
        num_classes = max(int(min_num_classes), max_observed)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(targets, preds):
            conf_matrix[int(t), int(p)] += 1

    per_class_precision = np.zeros(num_classes, dtype=np.float64)
    per_class_recall = np.zeros(num_classes, dtype=np.float64)
    per_class_f1 = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        per_class_precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class_recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p, r = per_class_precision[c], per_class_recall[c]
        per_class_f1[c] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    support = conf_matrix.sum(axis=1)
    present = support > 0
    if present.any():
        macro_precision = per_class_precision[present].mean()
        macro_recall = per_class_recall[present].mean()
        macro_f1 = per_class_f1[present].mean()
        balanced_accuracy = per_class_recall[present].mean()
    else:
        macro_precision = macro_recall = macro_f1 = balanced_accuracy = 0.0

    total = int(conf_matrix.sum())
    accuracy = float(np.trace(conf_matrix) / total) if total > 0 else 0.0

    return {
        "num_classes": num_classes,
        "conf_matrix": conf_matrix,
        "support": support,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }


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


def train_epoch(
    model,
    loader,
    optimizer,
    epoch,
    device,
    logger,
    scaler=None,
    use_segmentation=False,
    accumulation_steps=1,
    seg_loss_weight=0.1,
):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_cls_loss = AverageMeter()
    run_seg_loss = AverageMeter()
    seg_loss_weight = max(0.0, float(seg_loss_weight))
    all_preds = []
    all_targets = []

    # Apply label smoothing due to visual overlap in NSCLC/SCLC subtypes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Tumor masks are extremely sparse; raw BCE pushes the decoder toward
    # "predict all zeros". Combining BCE with Dice gives a foreground-aware
    # signal that actually teaches the encoder *where* the lesion is —
    # which is the whole reason DAPT exists (BigLunge has no masks).
    dice_loss_fn = DiceLoss(sigmoid=True)

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
                    # Compute the aux seg loss only on samples that genuinely
                    # have a mask. BCE gives a pixel-wise signal, Dice gives
                    # a region-overlap signal — combined they are substantially
                    # more stable on sparse tumor foregrounds than either alone.
                    seg_logits_valid = seg_outputs[valid_mask_indices]
                    masks_valid = masks[valid_mask_indices]
                    bce_seg = nn.functional.binary_cross_entropy_with_logits(
                        seg_logits_valid, masks_valid
                    )
                    dice_seg = dice_loss_fn(seg_logits_valid, masks_valid)
                    seg_loss = 0.5 * bce_seg + 0.5 * dice_seg
                    loss = cls_loss + (seg_loss_weight * seg_loss)
                    seg_loss_val = seg_loss.item()
                else:
                    loss = cls_loss
            else:
                logits = model(data, return_segmentation=False)
                loss = criterion(logits, target)
                cls_loss = loss
                seg_loss_val = 0.0

            preds = torch.argmax(logits.detach(), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(target.detach().cpu().tolist())
            
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
        if (idx + 1) % 20 == 0 or (idx + 1) == len(loader):
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

    train_metrics = _compute_classification_metrics(all_targets, all_preds)
    train_summary = (
        f"Train Epoch {epoch} Summary => Loss: {run_loss.avg:.4f}, "
        f"Accuracy: {train_metrics['accuracy']:.4f}, "
        f"MacroF1: {train_metrics['macro_f1']:.4f}"
    )
    print(train_summary)
    logger.info(train_summary)

    return run_loss.avg, train_metrics["macro_f1"]


@torch.no_grad()
def validate_epoch(model, loader, device, logger, return_probabilities=False):
    model.eval()
    run_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
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

        if return_probabilities:
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().tolist())
        
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(target.cpu().tolist())

    metrics = _compute_classification_metrics(all_targets, all_preds)
    num_classes = metrics["num_classes"]
    conf_matrix = metrics["conf_matrix"]
    support = metrics["support"]
    accuracy = metrics["accuracy"]
    balanced_accuracy = metrics["balanced_accuracy"]
    macro_precision = metrics["macro_precision"]
    macro_recall = metrics["macro_recall"]
    macro_f1 = metrics["macro_f1"]
    per_class_precision = metrics["per_class_precision"]
    per_class_recall = metrics["per_class_recall"]
    per_class_f1 = metrics["per_class_f1"]

    val_msg = (
        f"Validation Complete => Loss: {run_loss.avg:.4f}, "
        f"Accuracy: {accuracy:.4f}, BalancedAcc: {balanced_accuracy:.4f}, "
        f"MacroPrecision: {macro_precision:.4f}, MacroRecall: {macro_recall:.4f}, "
        f"MacroF1: {macro_f1:.4f}"
    )
    print(val_msg)
    logger.info(val_msg)

    # Per-class breakdown — the actionable signal for the imbalanced val set.
    for c in range(num_classes):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class{c}"
        logger.info(
            f"  [{name:<16}] support={int(support[c]):<4} "
            f"precision={per_class_precision[c]:.4f} "
            f"recall={per_class_recall[c]:.4f} "
            f"f1={per_class_f1[c]:.4f}"
        )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        mem_msg = f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        print(mem_msg)
        logger.info(mem_msg)

    # Pretty-print the confusion matrix.
    if len(all_targets) > 0:
        display_names = [
            CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class{i}"
            for i in range(num_classes)
        ]
        print("\nConfusion Matrix (Rows: Actual Class, Columns: Predicted Class):")
        logger.info("\nConfusion Matrix (Rows: Actual Class, Columns: Predicted Class):")
        header = f"{'True \\ Pred':<18}" + "".join([f"{display_names[j]:<16}" for j in range(num_classes)])
        print(header)
        logger.info(header)
        for i in range(num_classes):
            row_str = f"Y:{display_names[i]:<16}"
            for j in range(num_classes):
                row_str += f"{int(conf_matrix[i, j]):<16}"
            print(row_str)
            logger.info(row_str)
        print("\n")

    inference_payload = None
    if return_probabilities:
        class_prob_sums = np.zeros(num_classes, dtype=np.float64)
        pred_hist = np.zeros(num_classes, dtype=np.int64)
        samples = []

        for idx, (true_label, pred_label, probs_row) in enumerate(zip(all_targets, all_preds, all_probs)):
            probs_np = np.asarray(probs_row, dtype=np.float64)
            if probs_np.shape[0] != num_classes:
                aligned = np.zeros(num_classes, dtype=np.float64)
                n_copy = min(num_classes, probs_np.shape[0])
                aligned[:n_copy] = probs_np[:n_copy]
                probs_np = aligned

            class_prob_sums += probs_np
            if 0 <= int(pred_label) < num_classes:
                pred_hist[int(pred_label)] += 1

            samples.append({
                "sample_index": int(idx),
                "true_label": int(true_label),
                "true_name": display_names[int(true_label)] if 0 <= int(true_label) < num_classes else f"Class{int(true_label)}",
                "pred_label": int(pred_label),
                "pred_name": display_names[int(pred_label)] if 0 <= int(pred_label) < num_classes else f"Class{int(pred_label)}",
                "confidence": float(np.max(probs_np)) if probs_np.size > 0 else 0.0,
                "probabilities": {
                    display_names[c]: float(probs_np[c])
                    for c in range(num_classes)
                },
            })

        num_samples = len(samples)
        if num_samples > 0:
            mean_probs = class_prob_sums / float(num_samples)
            pred_fracs = pred_hist.astype(np.float64) / float(num_samples)
        else:
            mean_probs = np.zeros(num_classes, dtype=np.float64)
            pred_fracs = np.zeros(num_classes, dtype=np.float64)

        mean_probs_dict = {display_names[c]: float(mean_probs[c]) for c in range(num_classes)}
        pred_counts_dict = {display_names[c]: int(pred_hist[c]) for c in range(num_classes)}
        pred_fracs_dict = {display_names[c]: float(pred_fracs[c]) for c in range(num_classes)}

        logger.info(
            "Inference probability summary => " +
            ", ".join([f"mean P({name})={mean_probs_dict[name]:.4f}" for name in display_names])
        )
        logger.info(
            "Inference predicted class distribution => " +
            ", ".join([f"{name}: {pred_counts_dict[name]} ({pred_fracs_dict[name]:.3f})" for name in display_names])
        )

        inference_payload = {
            "num_samples": int(num_samples),
            "class_names": display_names,
            "mean_probability_per_class": mean_probs_dict,
            "predicted_class_counts": pred_counts_dict,
            "predicted_class_fractions": pred_fracs_dict,
            "samples": samples,
        }

    result = {
        "loss": run_loss.avg,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
    }

    if inference_payload is not None:
        result["inference_probabilities"] = inference_payload

    return result
