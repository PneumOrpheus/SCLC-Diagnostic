"""Training loop for the 2D per-slice pipeline.

Key differences from the 3D/2.5D ``training/train.py`` loop:

- No segmentation auxiliary loss (the 2D baseline is classification-only).
- Each training sample is one axial slice; each volume contributes many slices.
- Validation aggregates slice-level softmax probabilities per volume (mean),
  then argmax, so the reported metrics are volume/patient-level — directly
  comparable to the 3D and 2.5D pipelines.
"""
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceLoss

from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)

from sclc.training.bootstrap import bootstrap_ci, per_class_f1_ci
from sclc.training.train_3d import (
    AverageMeter,
    CLASS_NAMES,
    NUM_CLASSES,
    _compute_classification_metrics,
)
from sclc.training.bbox_utils import bbox_loss_2d


def _macro_f1(yt, yp):
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _bacc(yt, yp):
    return float(balanced_accuracy_score(yt, yp))


def _get_class_weight_tensor(loader, device, logger=None) -> Optional[torch.Tensor]:
    """Build inverse-frequency class weights from loader.dataset.data.

    We normalize weights to mean=1 and clip the extremes so weighting stays
    stable even on small classes.
    """
    cached = getattr(loader, "_class_weight_tensor", None)
    if cached is not None:
        return cached.to(device)

    labels: List[int] = []
    dataset = getattr(loader, "dataset", None)
    entries = getattr(dataset, "data", None)
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            y = item.get("scan_label")
            if y is None:
                continue
            yi = int(y)
            if 0 <= yi < NUM_CLASSES:
                labels.append(yi)

    if not labels:
        if logger is not None:
            logger.warning("[2D] Class-weighted loss disabled: could not infer labels from loader.dataset.data.")
        return None

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=NUM_CLASSES).astype(np.float32)
    inv_freq = counts.sum() / np.clip(counts, 1.0, None)
    weights = inv_freq / max(float(inv_freq.mean()), 1e-8)
    weights = np.clip(weights, 0.25, 4.0).astype(np.float32)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    loader._class_weight_tensor = weight_tensor

    if logger is not None:
        cls_msg = []
        for i in range(NUM_CLASSES):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class{i}"
            cls_msg.append(f"{name}: n={int(counts[i])}, w={weights[i]:.3f}")
        logger.info("[2D] Class-weighted CE active -> " + " | ".join(cls_msg))

    return weight_tensor.to(device)


def simple_collate_fn_2d(batch):
    """Collate per-slice samples.

    Returns
    -------
    images   : (B, 1, H, W) float tensor
    labels   : (B,) long tensor
    masks    : (B, 1, H, W) float tensor or None
    has_mask : (B,) bool tensor or None
    bboxes   : (B, 4) float tensor or None
    has_bbox : (B,) bool tensor or None
    meta     : list[dict] with ``volume_id``, ``patient_id`` (if present) and
               ``slice_idx`` for patient-level aggregation in validate.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([int(item["scan_label"]) for item in batch], dtype=torch.long)

    masks = None
    has_mask = None
    if any("tumor_mask" in item for item in batch):
        masks_list = []
        has_mask_list = []
        for item in batch:
            mask = item.get("tumor_mask")
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
                bbox_t = torch.zeros(4, dtype=torch.float32)
                hb = False
            else:
                bbox_t = bbox if torch.is_tensor(bbox) else torch.as_tensor(bbox, dtype=torch.float32)
                hb = bool(item.get("has_bbox", True))
            bbox_list.append(bbox_t)
            has_bbox_list.append(hb)
        bboxes = torch.stack(bbox_list, dim=0)
        has_bbox = torch.tensor(has_bbox_list, dtype=torch.bool)

    meta: List[Dict[str, Any]] = []
    for item in batch:
        meta.append({
            "volume_id": item.get("volume_id") or item.get("image"),
            "patient_id": item.get("patient_id"),
            "slice_idx": int(item.get("slice_idx", -1)),
        })
    return images, labels, masks, has_mask, bboxes, has_bbox, meta


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Sample lam ~ Beta(alpha, alpha), mix x, return (x_mix, y_a, y_b, lam).

    We flip ``lam = max(lam, 1-lam)`` so the primary target y_a is always the
    larger-weighted sample — keeps on-the-fly train-metric tracking against
    y_a interpretable as "how well we predict the dominant label of the mix".
    alpha<=0 returns the identity (no mixing) so the training loop can stay
    uniform whether MixUp is on or off. For tiny batches resample the
    permutation until it has no fixed points; see flaws.md 1.4.
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


def train_epoch_2d(
    model,
    loader,
    optimizer,
    epoch,
    device,
    logger,
    scaler=None,
    accumulation_steps: int = 1,
    mixup_alpha: float = 0.0,
    use_det_seg: bool = False,
    seg_loss_weight: float = 0.1,
    bbox_loss_weight: float = 0.1,
    **_unused,  # keep a shared signature with train_epoch so run_training_phase can swap them
):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_cls_loss = AverageMeter()
    run_seg_loss = AverageMeter()
    run_box_loss = AverageMeter()
    all_preds: List[int] = []
    all_targets: List[int] = []

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    dice_loss_fn = DiceLoss(sigmoid=True)
    optimizer.zero_grad()
    use_det_seg = bool(use_det_seg)
    seg_loss_weight = max(0.0, float(seg_loss_weight))
    bbox_loss_weight = max(0.0, float(bbox_loss_weight))
    mixup_active = float(mixup_alpha) > 0.0 and not use_det_seg
    if mixup_active and epoch == 1:
        logger.info(f"[2D] MixUp active with alpha={mixup_alpha:.3f} (Beta({mixup_alpha},{mixup_alpha}))")
    elif float(mixup_alpha) > 0.0 and use_det_seg and epoch == 1:
        logger.info(
            f"[2D] MixUp requested (alpha={mixup_alpha:.3f}) but suppressed because "
            f"use_det_seg=True (image/mask alignment would break)."
        )

    for idx, batch_data in enumerate(loader):
        if len(batch_data) >= 7:
            data, target, masks, has_mask, bboxes, has_bbox, _ = batch_data
        else:
            data, target, _ = batch_data
            masks = has_mask = bboxes = has_bbox = None
        data, target = data.to(device), target.to(device)
        masks = masks.to(device) if masks is not None else None
        bboxes = bboxes.to(device) if bboxes is not None else None

        if mixup_active:
            data_mix, y_a, y_b, lam = _mixup_batch(data, target, alpha=float(mixup_alpha))
        else:
            data_mix, y_a, y_b, lam = data, target, target, 1.0

        with torch.amp.autocast(enabled=(scaler is not None), device_type="cuda"):
            if use_det_seg:
                outputs = model(data_mix, return_segmentation=True, return_detection=True)
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        logits, seg_logits, box_pred = outputs
                    else:
                        logits, seg_logits = outputs
                        box_pred = None
                else:
                    logits, seg_logits, box_pred = outputs, None, None
                cls_loss = criterion(logits, y_a)
                loss = cls_loss

                seg_loss_val = 0.0
                if (
                    seg_logits is not None
                    and seg_logits.requires_grad
                    and masks is not None
                    and has_mask is not None
                    and has_mask.any()
                ):
                    valid_mask_indices = has_mask.to(device)
                    seg_logits_valid = seg_logits[valid_mask_indices]
                    masks_valid = masks[valid_mask_indices].float()
                    bce_seg = nn.functional.binary_cross_entropy_with_logits(seg_logits_valid, masks_valid)
                    dice_seg = dice_loss_fn(seg_logits_valid, masks_valid)
                    seg_loss = 0.5 * bce_seg + 0.5 * dice_seg
                    loss = loss + (seg_loss_weight * seg_loss)
                    seg_loss_val = seg_loss.item()
                run_seg_loss.update(seg_loss_val, n=data.size(0))

                box_loss_val = 0.0
                if (
                    box_pred is not None
                    and box_pred.requires_grad
                    and bboxes is not None
                    and has_bbox is not None
                    and has_bbox.any()
                ):
                    valid_box_indices = has_bbox.to(device)
                    pred_boxes = box_pred[valid_box_indices]
                    tgt_boxes = bboxes[valid_box_indices]
                    l1, giou = bbox_loss_2d(pred_boxes, tgt_boxes)
                    box_loss = l1.mean() + giou.mean()
                    loss = loss + (bbox_loss_weight * box_loss)
                    box_loss_val = box_loss.item()
                run_box_loss.update(box_loss_val, n=data.size(0))
            else:
                logits = model(data_mix, return_segmentation=False)
                if mixup_active:
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, y_a)
                cls_loss = loss

            preds = torch.argmax(logits.detach(), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y_a.detach().cpu().tolist())

            unscaled_loss = loss.item()
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        run_loss.update(unscaled_loss, n=data.size(0))
        run_cls_loss.update(cls_loss.item(), n=data.size(0))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(loader):
            if use_det_seg:
                msg = (
                    f"[2D] Epoch {epoch} [{idx + 1}/{len(loader)}] "
                    f"Loss: {run_loss.avg:.4f} (Cls: {run_cls_loss.avg:.4f} / "
                    f"Seg: {run_seg_loss.avg:.4f} / Box: {run_box_loss.avg:.4f}) "
                    f"Time: {time.time() - start_time:.2f}s"
                )
            else:
                msg = (
                    f"[2D] Epoch {epoch} [{idx + 1}/{len(loader)}] "
                    f"Loss: {run_loss.avg:.4f} Time: {time.time() - start_time:.2f}s"
                )
            print(msg)
            logger.info(msg)
            start_time = time.time()

    metrics = _compute_classification_metrics(all_targets, all_preds)
    summary = (
        f"[2D] Train Epoch {epoch} (slice-level) => Loss: {run_loss.avg:.4f}, "
        f"Acc: {metrics['accuracy']:.4f}, MacroF1: {metrics['macro_f1']:.4f}"
    )
    print(summary)
    logger.info(summary)
    return run_loss.avg, metrics["macro_f1"]


@torch.no_grad()
def validate_epoch_2d(
    model, loader, device, logger,
    return_probabilities: bool = False,
    compute_ci: bool = True,
    n_boot: int = 1000,
):
    """Validate at volume level: mean softmax over a volume's slices, argmax.

    Also reports slice-level metrics for sanity. The volume-level metrics are
    the ones that matter — that's the comparable number vs. 3D/2.5D.
    """
    model.eval()
    run_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    slice_preds: List[int] = []
    slice_targets: List[int] = []

    # per-volume accumulation
    volume_prob_sum: Dict[str, np.ndarray] = {}
    volume_slice_count: Dict[str, int] = {}
    volume_label: Dict[str, int] = {}
    volume_patient: Dict[str, Any] = {}

    print("\n[2D] Starting validation...")
    logger.info("[2D] Starting validation...")

    for batch_data in loader:
        if len(batch_data) >= 7:
            data, target, _, _, _, _, meta = batch_data
        else:
            data, target, meta = batch_data
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = criterion(logits, target)
        run_loss.update(loss.item(), n=data.size(0))

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        tgts = target.cpu().numpy()
        slice_preds.extend(int(p) for p in preds)
        slice_targets.extend(int(t) for t in tgts)

        for i, m in enumerate(meta):
            vid = m["volume_id"]
            if vid not in volume_prob_sum:
                volume_prob_sum[vid] = np.zeros(probs.shape[1], dtype=np.float64)
                volume_slice_count[vid] = 0
                volume_label[vid] = int(tgts[i])
                volume_patient[vid] = m.get("patient_id")
            volume_prob_sum[vid] += probs[i]
            volume_slice_count[vid] += 1

    volume_ids = list(volume_prob_sum.keys())
    volume_preds: List[int] = []
    volume_targets: List[int] = []
    volume_probs: List[np.ndarray] = []
    for vid in volume_ids:
        mean_p = volume_prob_sum[vid] / max(1, volume_slice_count[vid])
        volume_probs.append(mean_p)
        volume_preds.append(int(mean_p.argmax()))
        volume_targets.append(volume_label[vid])

    # Patient-level aggregation: equal weight per volume (mean of each volume's
    # mean-softmax), then argmax. For datasets where one patient == one volume
    # (e.g. BigLunge) this reduces to the volume-level numbers. For Lung-PET-CT-Dx
    # (multi-scan patients) it's the one clinically meaningful number.
    # Volumes with an unknown patient_id get their own bucket keyed on volume_id
    # so they contribute independently rather than collapsing into one fake
    # "None" patient.
    patient_prob_sum: Dict[Any, np.ndarray] = {}
    patient_volume_count: Dict[Any, int] = {}
    patient_slice_count: Dict[Any, int] = {}
    patient_label: Dict[Any, int] = {}
    patient_volume_ids: Dict[Any, List[str]] = {}
    for i, vid in enumerate(volume_ids):
        pid = volume_patient.get(vid)
        key = pid if pid is not None else f"__vol__:{vid}"
        if key not in patient_prob_sum:
            patient_prob_sum[key] = np.zeros(volume_probs[i].shape[0], dtype=np.float64)
            patient_volume_count[key] = 0
            patient_slice_count[key] = 0
            patient_label[key] = volume_label[vid]
            patient_volume_ids[key] = []
        patient_prob_sum[key] += volume_probs[i]
        patient_volume_count[key] += 1
        patient_slice_count[key] += volume_slice_count[vid]
        patient_volume_ids[key].append(vid)

    patient_keys = list(patient_prob_sum.keys())
    patient_preds: List[int] = []
    patient_targets: List[int] = []
    patient_probs: List[np.ndarray] = []
    for key in patient_keys:
        mean_p = patient_prob_sum[key] / max(1, patient_volume_count[key])
        patient_probs.append(mean_p)
        patient_preds.append(int(mean_p.argmax()))
        patient_targets.append(patient_label[key])

    slice_metrics = _compute_classification_metrics(slice_targets, slice_preds)
    metrics = _compute_classification_metrics(volume_targets, volume_preds)
    patient_metrics = _compute_classification_metrics(patient_targets, patient_preds)
    num_classes = metrics["num_classes"]

    slice_msg = (
        f"[2D] Val (slice-level, n={len(slice_targets)}) => "
        f"Acc: {slice_metrics['accuracy']:.4f}, "
        f"MacroF1: {slice_metrics['macro_f1']:.4f}"
    )
    val_msg = (
        f"[2D] Val (volume-level, n={len(volume_targets)}) => Loss: {run_loss.avg:.4f}, "
        f"Acc: {metrics['accuracy']:.4f}, BalancedAcc: {metrics['balanced_accuracy']:.4f}, "
        f"MacroPrecision: {metrics['macro_precision']:.4f}, MacroRecall: {metrics['macro_recall']:.4f}, "
        f"MacroF1: {metrics['macro_f1']:.4f}"
    )
    patient_msg = (
        f"[2D] Val (patient-level, n={len(patient_targets)}) => "
        f"Acc: {patient_metrics['accuracy']:.4f}, BalancedAcc: {patient_metrics['balanced_accuracy']:.4f}, "
        f"MacroPrecision: {patient_metrics['macro_precision']:.4f}, MacroRecall: {patient_metrics['macro_recall']:.4f}, "
        f"MacroF1: {patient_metrics['macro_f1']:.4f}"
    )
    print(slice_msg); print(val_msg); print(patient_msg)
    logger.info(slice_msg); logger.info(val_msg); logger.info(patient_msg)

    display_names = [
        CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class{i}"
        for i in range(num_classes)
    ]
    for c in range(num_classes):
        logger.info(
            f"  [{display_names[c]:<16}] support={int(metrics['support'][c]):<4} "
            f"precision={metrics['per_class_precision'][c]:.4f} "
            f"recall={metrics['per_class_recall'][c]:.4f} "
            f"f1={metrics['per_class_f1'][c]:.4f}"
        )

    # Confusion matrices (volume-level and patient-level)
    def _log_conf_matrix(title: str, matrix) -> None:
        print(f"\n[2D] {title}:")
        logger.info(f"\n[2D] {title}:")
        header = f"{'True \\ Pred':<18}" + "".join([f"{display_names[j]:<16}" for j in range(num_classes)])
        print(header); logger.info(header)
        for i in range(num_classes):
            row = f"Y:{display_names[i]:<16}" + "".join(f"{int(matrix[i, j]):<16}" for j in range(num_classes))
            print(row); logger.info(row)
        print("")

    if len(volume_targets) > 0:
        _log_conf_matrix("Volume-level Confusion Matrix", metrics["conf_matrix"])
    if len(patient_targets) > 0:
        _log_conf_matrix("Patient-level Confusion Matrix", patient_metrics["conf_matrix"])

    result = {
        "loss": run_loss.avg,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_class_precision": metrics["per_class_precision"].tolist(),
        "per_class_recall": metrics["per_class_recall"].tolist(),
        "per_class_f1": metrics["per_class_f1"].tolist(),
        "slice_level": {
            "accuracy": slice_metrics["accuracy"],
            "macro_f1": slice_metrics["macro_f1"],
            "num_slices": len(slice_targets),
        },
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

    # Stratified bootstrap CIs on patient-level metrics. n_patients=52 on
    # Lung-PET-CT-Dx DAPT val → per-class F1 swings ~0.1 per single-patient
    # flip, so the CI is the only honest thing to report.
    if compute_ci and len(patient_targets) > 0:
        _, mf1_lo, mf1_hi = bootstrap_ci(
            patient_targets, patient_preds, _macro_f1, n_boot=n_boot, rng_seed=0
        )
        _, bacc_lo, bacc_hi = bootstrap_ci(
            patient_targets, patient_preds, _bacc, n_boot=n_boot, rng_seed=0
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
            f"[2D] Bootstrap CI (n_boot={n_boot}): "
            f"patient MacroF1={patient_metrics['macro_f1']:.4f} [{mf1_lo:.4f}, {mf1_hi:.4f}], "
            f"BalAcc={patient_metrics['balanced_accuracy']:.4f} [{bacc_lo:.4f}, {bacc_hi:.4f}]"
        )

    if return_probabilities:
        samples = []
        class_prob_sums = np.zeros(num_classes, dtype=np.float64)
        pred_hist = np.zeros(num_classes, dtype=np.int64)
        for i, vid in enumerate(volume_ids):
            mean_p = volume_probs[i]
            if mean_p.shape[0] != num_classes:
                aligned = np.zeros(num_classes, dtype=np.float64)
                n_copy = min(num_classes, mean_p.shape[0])
                aligned[:n_copy] = mean_p[:n_copy]
                mean_p = aligned
            class_prob_sums += mean_p
            pl = volume_preds[i]
            if 0 <= pl < num_classes:
                pred_hist[pl] += 1
            samples.append({
                "sample_index": i,
                "volume_id": vid,
                "patient_id": volume_patient.get(vid),
                "num_slices": int(volume_slice_count[vid]),
                "true_label": int(volume_targets[i]),
                "true_name": display_names[int(volume_targets[i])] if 0 <= int(volume_targets[i]) < num_classes else f"Class{int(volume_targets[i])}",
                "pred_label": int(volume_preds[i]),
                "pred_name": display_names[int(volume_preds[i])] if 0 <= int(volume_preds[i]) < num_classes else f"Class{int(volume_preds[i])}",
                "confidence": float(np.max(mean_p)) if mean_p.size > 0 else 0.0,
                "probabilities": {display_names[c]: float(mean_p[c]) for c in range(num_classes)},
            })
        n = max(1, len(samples))
        mean_probs_dict = {display_names[c]: float(class_prob_sums[c] / n) for c in range(num_classes)}
        pred_counts_dict = {display_names[c]: int(pred_hist[c]) for c in range(num_classes)}
        pred_fracs_dict = {display_names[c]: float(pred_hist[c] / n) for c in range(num_classes)}
        # Patient-level rollup for the inference JSON. Keys beginning with
        # "__vol__:" are synthetic fallbacks for volumes whose patient_id was
        # missing — we surface them as patient_id=None in the payload so
        # downstream analysis can tell them apart from real patient groupings.
        patient_samples = []
        pat_class_prob_sums = np.zeros(num_classes, dtype=np.float64)
        pat_pred_hist = np.zeros(num_classes, dtype=np.int64)
        for i, key in enumerate(patient_keys):
            mean_p = patient_probs[i]
            if mean_p.shape[0] != num_classes:
                aligned = np.zeros(num_classes, dtype=np.float64)
                n_copy = min(num_classes, mean_p.shape[0])
                aligned[:n_copy] = mean_p[:n_copy]
                mean_p = aligned
            pat_class_prob_sums += mean_p
            pl = patient_preds[i]
            if 0 <= pl < num_classes:
                pat_pred_hist[pl] += 1
            is_fallback = isinstance(key, str) and key.startswith("__vol__:")
            true_lbl = int(patient_targets[i])
            pred_lbl = int(patient_preds[i])
            patient_samples.append({
                "sample_index": i,
                "patient_id": None if is_fallback else key,
                "volume_ids": list(patient_volume_ids[key]),
                "num_volumes": int(patient_volume_count[key]),
                "num_slices": int(patient_slice_count[key]),
                "true_label": true_lbl,
                "true_name": display_names[true_lbl] if 0 <= true_lbl < num_classes else f"Class{true_lbl}",
                "pred_label": pred_lbl,
                "pred_name": display_names[pred_lbl] if 0 <= pred_lbl < num_classes else f"Class{pred_lbl}",
                "confidence": float(np.max(mean_p)) if mean_p.size > 0 else 0.0,
                "probabilities": {display_names[c]: float(mean_p[c]) for c in range(num_classes)},
            })
        n_p = max(1, len(patient_samples))
        pat_mean_probs_dict = {display_names[c]: float(pat_class_prob_sums[c] / n_p) for c in range(num_classes)}
        pat_pred_counts_dict = {display_names[c]: int(pat_pred_hist[c]) for c in range(num_classes)}
        pat_pred_fracs_dict = {display_names[c]: float(pat_pred_hist[c] / n_p) for c in range(num_classes)}

        result["inference_probabilities"] = {
            "num_samples": len(samples),
            "class_names": display_names,
            "mean_probability_per_class": mean_probs_dict,
            "predicted_class_counts": pred_counts_dict,
            "predicted_class_fractions": pred_fracs_dict,
            "samples": samples,
            "patient_level": {
                "num_patients": len(patient_samples),
                "mean_probability_per_class": pat_mean_probs_dict,
                "predicted_class_counts": pat_pred_counts_dict,
                "predicted_class_fractions": pat_pred_fracs_dict,
                "samples": patient_samples,
            },
        }

    return result
