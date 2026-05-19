import time
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from monai.losses import DiceLoss
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
from sclc.training.bbox_utils import bbox_loss_3d

NUM_CLASSES = 3
CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]


def _macro_f1(yt, yp):
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _bacc(yt, yp):
    return float(balanced_accuracy_score(yt, yp))

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


def _extract_volume_id(item: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of a string volume identifier post-load.

    Order of preference:
      1. Explicit string ``volume_id`` field set upstream (2D / MIL pipelines).
      2. ``image`` field if it's still a string (pre-transform).
      3. MONAI MetaTensor's ``filename_or_obj`` metadata (post-transform).
      4. ``None`` — caller falls back to a synthetic id.

    Why this exists: in the 3D pipeline the data list entry has
    ``"image": "<path>.nii.gz"`` as a string, but by the time the collate
    function sees it the LoadNifti transform has replaced ``"image"`` with
    a ``MetaTensor``. The previous fallback ``item.get("volume_id") or
    item.get("image")`` then returned the Tensor — JSON-unserializable —
    and crashed the inference-probabilities dump at end of DAPT-test.
    """
    vid = item.get("volume_id")
    if isinstance(vid, str):
        return vid
    if vid is not None and not hasattr(vid, "shape"):
        return str(vid)
    img = item.get("image")
    if isinstance(img, str):
        return img
    meta = getattr(img, "meta", None) if img is not None else None
    if isinstance(meta, dict):
        fn = meta.get("filename_or_obj")
        if isinstance(fn, str):
            return fn
        if isinstance(fn, (list, tuple)) and fn and isinstance(fn[0], str):
            return fn[0]
    return None


def _collect_meta(batch):
    """Per-sample metadata for patient/volume-level eval aggregation.

    ``patient_id`` is set in the data list (data_loader.py) for both
    Lung-PET-CT-Dx and BigLunge. ``volume_id`` defaults to the original image
    path so multi-scan patients still produce distinct volume buckets in
    validate_epoch's slice→volume→patient rollup.
    """
    meta: List[Dict[str, Any]] = []
    for item in batch:
        meta.append({
            "patient_id": item.get("patient_id"),
            "volume_id":  _extract_volume_id(item),
        })
    return meta


def simple_collate_fn(batch):
    """Collate volumes + labels (+ optional masks) + per-sample meta.

    Always emits the ``meta`` list as the last tuple element so validate_epoch
    can aggregate multi-scan patients to a single patient-level prediction.
    """
    # Extract the volume and the single target class logic
    scans = torch.stack([item["image"] for item in batch], dim=0)

    # We want a 1D tensor of class indices for CrossEntropyLoss
    labels = torch.tensor([item["scan_label"] for item in batch], dtype=torch.long)
    meta = _collect_meta(batch)

    # Extract segmentation masks if available, filling missing ones with zeros
    masks = None
    has_mask = None
    if any("mask" in item for item in batch):
        masks_list = []
        has_mask_list = []
        for item in batch:
            mask = item.get("mask")
            if mask is None:
                mask = torch.zeros_like(item["image"])
                has_mask_list.append(False)
            else:
                has_mask_list.append(bool(mask.sum() > 0))
            masks_list.append(mask)
        masks = torch.stack(masks_list, dim=0)
        has_mask = torch.tensor(has_mask_list, dtype=torch.bool)

    bboxes = None
    has_bbox = None
    if any("bbox" in item for item in batch):
        bbox_list = []
        has_bbox_list = []
        for item in batch:
            bbox = item.get("bbox")
            if bbox is None:
                bbox_t = torch.zeros(6, dtype=torch.float32)
                hb = False
            else:
                bbox_t = bbox if torch.is_tensor(bbox) else torch.as_tensor(bbox, dtype=torch.float32)
                hb = bool(item.get("has_bbox", True))
            bbox_list.append(bbox_t)
            has_bbox_list.append(hb)
        bboxes = torch.stack(bbox_list, dim=0)
        has_bbox = torch.tensor(has_bbox_list, dtype=torch.bool)

    return scans, labels, masks, has_mask, bboxes, has_bbox, meta


def _mixup_3d(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Bag-style mixup for 3D volumes. Mixes (B, C, D, H, W) along B.

    Same convention as the 2D / MIL mixup helpers: ``lam ~ Beta(alpha, alpha)``,
    flipped to ``lam = max(lam, 1-lam)`` so y_a is the dominant target.
    Resamples the permutation up to 8 times to avoid identity perms at
    small B (sleeper bug for B=2).
    """
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)
    for _ in range(8):
        if not torch.any(idx == torch.arange(B, device=x.device)):
            break
        idx = torch.randperm(B, device=x.device)
    else:
        return x, y, y, 1.0
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def train_epoch(
    model,
    loader,
    optimizer,
    epoch,
    device,
    logger,
    scaler=None,
    use_segmentation=False,
    use_det_seg: bool = False,
    accumulation_steps=1,
    seg_loss_weight=0.1,
    bbox_loss_weight: float = 0.1,
    mixup_alpha=0.0,
    **_unused,  # keep signature tolerant of pipeline-specific kwargs (bag_dropout, etc.)
):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_cls_loss = AverageMeter()
    run_seg_loss = AverageMeter()
    run_box_loss = AverageMeter()
    seg_loss_weight = max(0.0, float(seg_loss_weight))
    bbox_loss_weight = max(0.0, float(bbox_loss_weight))
    all_preds = []
    all_targets = []

    # Apply label smoothing due to visual overlap in NSCLC/SCLC subtypes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Tumor masks are extremely sparse; raw BCE pushes the decoder toward
    # "predict all zeros". Combining BCE with Dice gives a foreground-aware
    # signal that actually teaches the encoder *where* the lesion is —
    # which is the whole reason DAPT exists (BigLunge has no masks).
    dice_loss_fn = DiceLoss(sigmoid=True)

    # Mixup is applied to classification only — when seg/det losses are
    # active we skip mixup entirely (mixing would desync image and mask/box).
    mixup_active = float(mixup_alpha) > 0.0 and not use_segmentation and not use_det_seg
    if mixup_active and epoch == 1:
        logger.info(f"[3D] Mixup active with alpha={mixup_alpha:.3f}")
    elif float(mixup_alpha) > 0.0 and (use_segmentation or use_det_seg) and epoch == 1:
        logger.info(
            f"[3D] Mixup requested (alpha={mixup_alpha:.3f}) but suppressed because "
            f"seg/det loss active (image/mask desync would corrupt aux loss)."
        )

    optimizer.zero_grad()  # 1. Zero gradients before the loop starts

    for idx, batch_data in enumerate(loader):
        has_mask_tensor = None
        has_bbox_tensor = None
        bboxes = None
        if len(batch_data) >= 7:
            data, target, masks, has_mask_tensor, bboxes, has_bbox_tensor, _meta = batch_data
            masks = masks.to(device) if masks is not None else None
            bboxes = bboxes.to(device) if bboxes is not None else None
        elif len(batch_data) == 5:
            data, target, masks, has_mask_tensor, _meta = batch_data
            masks = masks.to(device) if masks is not None else None
        elif len(batch_data) == 4:
            data, target, masks, has_mask_tensor = batch_data
            masks = masks.to(device) if masks is not None else None
        elif len(batch_data) == 3:
            data, target, masks = batch_data
            masks = masks.to(device) if masks is not None else None
        else:
            data, target = batch_data
            masks = None
            
        data, target = data.to(device), target.to(device)

        # Mixup only when seg-aux loss is OFF — see mixup_active gate above.
        if mixup_active:
            data, y_a, y_b, lam = _mixup_3d(data, target, alpha=float(mixup_alpha))
        else:
            y_a, y_b, lam = target, target, 1.0

        with torch.amp.autocast(enabled=(scaler is not None), device_type='cuda'):
            if use_segmentation or use_det_seg:
                outputs = model(data, return_segmentation=True, return_detection=use_det_seg)
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        logits, seg_outputs, box_pred = outputs
                    else:
                        logits, seg_outputs = outputs
                        box_pred = None
                else:
                    logits, seg_outputs, box_pred = outputs, None, None
                cls_loss = criterion(logits, target)
                loss = cls_loss

                seg_loss_val = 0.0
                if (
                    seg_outputs is not None
                    and seg_outputs.requires_grad
                    and masks is not None
                    and has_mask_tensor is not None
                    and has_mask_tensor.any()
                ):
                    valid_mask_indices = has_mask_tensor.to(device)
                    masks = masks.float()
                    seg_logits_valid = seg_outputs[valid_mask_indices]
                    masks_valid = masks[valid_mask_indices]
                    bce_seg = nn.functional.binary_cross_entropy_with_logits(
                        seg_logits_valid, masks_valid
                    )
                    dice_seg = dice_loss_fn(seg_logits_valid, masks_valid)
                    seg_loss = 0.5 * bce_seg + 0.5 * dice_seg
                    loss = loss + (seg_loss_weight * seg_loss)
                    seg_loss_val = seg_loss.item()

                box_loss_val = 0.0
                if (
                    use_det_seg
                    and box_pred is not None
                    and box_pred.requires_grad
                    and bboxes is not None
                    and has_bbox_tensor is not None
                    and has_bbox_tensor.any()
                ):
                    valid_box_indices = has_bbox_tensor.to(device)
                    pred_boxes = box_pred[valid_box_indices]
                    tgt_boxes = bboxes[valid_box_indices]
                    l1, giou = bbox_loss_3d(pred_boxes, tgt_boxes)
                    box_loss = l1.mean() + giou.mean()
                    loss = loss + (bbox_loss_weight * box_loss)
                    box_loss_val = box_loss.item()
                run_box_loss.update(box_loss_val, n=data.size(0))
            else:
                logits = model(data, return_segmentation=False)
                if mixup_active:
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, target)
                cls_loss = loss
                seg_loss_val = 0.0

            preds = torch.argmax(logits.detach(), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y_a.detach().cpu().tolist())
            
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
            if use_segmentation or use_det_seg:
                msg = (f"Epoch {epoch} [{idx + 1}/{len(loader)}] "
                       f"Loss: {run_loss.avg:.4f} (Cls: {run_cls_loss.avg:.4f} / "
                       f"Seg: {run_seg_loss.avg:.4f} / Box: {run_box_loss.avg:.4f}) "
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
def validate_epoch(
    model, loader, device, logger,
    return_probabilities: bool = False,
    compute_ci: bool = True,
    n_boot: int = 1000,
):
    """Validate the 3D pipeline at volume level AND aggregate to patient level.

    Volume-level metrics: argmax of per-volume logits — what historical 3D
    runs reported. Patient-level metrics: mean of per-volume softmax over a
    patient's volumes, then argmax. For Lung-PET-CT-Dx (max 2 scans/patient)
    this collapses 100ish volumes to 52ish patients; for BigLunge (1 scan
    per patient) volume-level == patient-level.

    Bootstrap CIs (stratified, n_boot=1000) are computed on the patient-level
    targets/preds for thesis-grade headline numbers.
    """
    model.eval()
    run_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    all_preds = []
    all_targets = []
    all_probs = []
    all_patient_ids: List[Optional[Any]] = []
    all_volume_ids: List[Optional[Any]] = []

    print("\nStarting validation...")
    logger.info("Starting validation...")

    for batch_data in loader:
        meta_batch: List[Dict[str, Any]] = []
        if len(batch_data) >= 7:
            data, target, _, _, _, _, meta_batch = batch_data
        elif len(batch_data) == 5:
            data, target, _, _, meta_batch = batch_data
        elif len(batch_data) == 4:
            data, target, _, _ = batch_data
        elif len(batch_data) == 3:
            data, target, _ = batch_data
        else:
            data, target = batch_data

        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = criterion(logits, target)

        run_loss.update(loss.item(), n=data.size(0))

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        if return_probabilities:
            all_probs.extend(probs.tolist())

        preds = probs.argmax(axis=1)
        tgts = target.cpu().numpy()
        for i in range(len(tgts)):
            all_preds.append(int(preds[i]))
            all_targets.append(int(tgts[i]))
            if meta_batch and i < len(meta_batch):
                all_patient_ids.append(meta_batch[i].get("patient_id"))
                all_volume_ids.append(meta_batch[i].get("volume_id"))
            else:
                all_patient_ids.append(None)
                all_volume_ids.append(None)
            if not return_probabilities:
                # Need probs for patient-level mean-of-softmax even if caller
                # doesn't want the per-sample probability dump.
                all_probs.append(probs[i].tolist())

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

    # Patient-level aggregation: mean of per-volume softmax over a patient's
    # volumes, then argmax. Volumes with no patient_id get a synthetic
    # per-index key so they contribute independently rather than collapsing
    # into one fake "None" patient bucket.
    patient_prob_sum: Dict[Any, np.ndarray] = {}
    patient_volume_count: Dict[Any, int] = {}
    patient_label: Dict[Any, int] = {}
    for i, pid in enumerate(all_patient_ids):
        if pid is not None:
            key: Any = pid
        else:
            vid = all_volume_ids[i] if i < len(all_volume_ids) else None
            key = f"__vol__:{vid}" if vid is not None else f"__idx__:{i}"
        if key not in patient_prob_sum:
            patient_prob_sum[key] = np.zeros(num_classes, dtype=np.float64)
            patient_volume_count[key] = 0
            patient_label[key] = int(all_targets[i])
        # all_probs[i] is a list/array of length num_classes from this volume
        p = np.asarray(all_probs[i], dtype=np.float64)
        if p.shape[0] != num_classes:
            aligned = np.zeros(num_classes, dtype=np.float64)
            n_copy = min(num_classes, p.shape[0])
            aligned[:n_copy] = p[:n_copy]
            p = aligned
        patient_prob_sum[key] += p
        patient_volume_count[key] += 1

    patient_keys = list(patient_prob_sum.keys())
    patient_preds: List[int] = []
    patient_targets: List[int] = []
    for key in patient_keys:
        mean_p = patient_prob_sum[key] / max(1, patient_volume_count[key])
        patient_preds.append(int(mean_p.argmax()))
        patient_targets.append(int(patient_label[key]))

    patient_metrics = _compute_classification_metrics(patient_targets, patient_preds)

    val_msg = (
        f"Validation (volume-level, n={len(all_targets)}) => Loss: {run_loss.avg:.4f}, "
        f"Accuracy: {accuracy:.4f}, BalancedAcc: {balanced_accuracy:.4f}, "
        f"MacroPrecision: {macro_precision:.4f}, MacroRecall: {macro_recall:.4f}, "
        f"MacroF1: {macro_f1:.4f}"
    )
    pat_msg = (
        f"Validation (patient-level, n={len(patient_targets)}) => "
        f"Acc: {patient_metrics['accuracy']:.4f}, BalancedAcc: {patient_metrics['balanced_accuracy']:.4f}, "
        f"MacroPrecision: {patient_metrics['macro_precision']:.4f}, MacroRecall: {patient_metrics['macro_recall']:.4f}, "
        f"MacroF1: {patient_metrics['macro_f1']:.4f}"
    )
    print(val_msg); print(pat_msg)
    logger.info(val_msg); logger.info(pat_msg)

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

            # patient_id / volume_id come from all_patient_ids / all_volume_ids
            # populated above. Including them brings the 3D samples up to parity
            # with the MIL and 2D paths and lets the misclassifications CSV
            # writer in main.py emit a clickable volume path per error row.
            pid = all_patient_ids[idx] if idx < len(all_patient_ids) else None
            vid = all_volume_ids[idx] if idx < len(all_volume_ids) else None
            samples.append({
                "sample_index": int(idx),
                "patient_id": pid,
                "volume_id": vid,
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
        "patient_level": {
            "accuracy": patient_metrics["accuracy"],
            "balanced_accuracy": patient_metrics["balanced_accuracy"],
            "macro_precision": patient_metrics["macro_precision"],
            "macro_recall": patient_metrics["macro_recall"],
            "macro_f1": patient_metrics["macro_f1"],
            "per_class_precision": patient_metrics["per_class_precision"].tolist(),
            "per_class_recall": patient_metrics["per_class_recall"].tolist(),
            "per_class_f1": patient_metrics["per_class_f1"].tolist(),
            "num_patients": len(patient_targets),
        },
    }

    # Stratified bootstrap CIs on patient-level predictions. Lazy-imported so
    # this file doesn't pull sklearn at module-load when bootstrap isn't used.
    if compute_ci and len(patient_targets) > 0:
        from sclc.training.bootstrap import bootstrap_ci, per_class_f1_ci
        _, mf1_lo, mf1_hi = bootstrap_ci(
            patient_targets, patient_preds, _macro_f1, n_boot=n_boot, rng_seed=0,
        )
        _, bacc_lo, bacc_hi = bootstrap_ci(
            patient_targets, patient_preds, _bacc, n_boot=n_boot, rng_seed=0,
        )
        pc_f1_ci = per_class_f1_ci(
            patient_targets, patient_preds, num_classes=num_classes,
            n_boot=n_boot, rng_seed=0,
        )
        result["patient_level"]["macro_f1_ci95"] = [mf1_lo, mf1_hi]
        result["patient_level"]["balanced_accuracy_ci95"] = [bacc_lo, bacc_hi]
        result["patient_level"]["per_class_f1_ci95"] = [list(pair) for pair in pc_f1_ci]
        result["patient_level"]["ci_n_boot"] = int(n_boot)
        logger.info(
            f"Bootstrap CI (n_boot={n_boot}): "
            f"patient MacroF1={patient_metrics['macro_f1']:.4f} [{mf1_lo:.4f}, {mf1_hi:.4f}], "
            f"BalAcc={patient_metrics['balanced_accuracy']:.4f} [{bacc_lo:.4f}, {bacc_hi:.4f}]"
        )

    if inference_payload is not None:
        result["inference_probabilities"] = inference_payload

    return result
