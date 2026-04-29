"""Bag-level training + validation for the MIL pipeline.

Key differences from ``training/train.py`` and ``training/train_2d.py``:

- One loss per bag. No per-slice / per-instance supervision.
- Validation is patient-level by construction (one bag = one patient). No
  slice-to-volume or volume-to-patient aggregation needed.
- No segmentation auxiliary loss (MIL has no decoder).
- Mixup is applied at the bag level (mix whole bags with a per-batch λ),
  following the same convention as ``train_2d.py``.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import balanced_accuracy_score, f1_score

from sclc.training.bootstrap import bootstrap_ci, per_class_f1_ci
from sclc.training.train_3d import (
    AverageMeter,
    CLASS_NAMES,
    NUM_CLASSES,
    _compute_classification_metrics,
)


def _macro_f1(yt, yp):
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _bacc(yt, yp):
    return float(balanced_accuracy_score(yt, yp))


def simple_collate_fn_mil(batch):
    """Stack bag tensors and labels; preserve per-sample metadata.

    Returns
    -------
    images : (B, N, 1, H, W) float tensor
    labels : (B,) long tensor
    meta   : list[dict] per sample with ``patient_id``, ``volume_id``.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([int(item["scan_label"]) for item in batch], dtype=torch.long)
    meta: List[Dict[str, Any]] = []
    for item in batch:
        meta.append({
            "patient_id": item.get("patient_id"),
            "volume_id": item.get("volume_id") or item.get("image"),
        })
    return images, labels, meta


def _bag_instance_dropout(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Zero a Bernoulli(drop_prob) subset of bag instances per sample.

    Acts as bag-level cutout: forces attention to distribute mass across
    multiple slices since any single slice may be dropped on a given step.
    Guarantees at least one surviving instance per bag — an all-zero bag
    feeds zero features into MIL attention and NaNs the softmax.
    """
    if drop_prob <= 0.0:
        return x
    B, N = x.shape[0], x.shape[1]
    keep = (torch.rand(B, N, device=x.device) >= drop_prob).float()
    all_drop = keep.sum(dim=1) == 0
    if bool(all_drop.any()):
        for i in torch.nonzero(all_drop, as_tuple=False).flatten().tolist():
            keep[i, torch.randint(N, (1,), device=x.device)] = 1.0
    return x * keep.view(B, N, 1, 1, 1)


def _mixup_bags(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Mixup at the bag level.

    ``x`` shape is ``(B, N, C, H, W)`` — we mix along B, keeping each bag's
    instance count N fixed. Sampling is identical to the per-slice mixup in
    ``train_2d.py``: Beta(alpha, alpha) with lam = max(lam, 1-lam).

    For B<4, ``torch.randperm(B)`` returns the identity with non-negligible
    probability (50% at B=2), so the "mix" degenerates to the original sample.
    Resample the permutation up to a few times until it is a derangement; if
    we still can't get one (B==1), skip mixup. See flaws.md 1.4.
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
        # No derangement found in 8 tries — skip mixup this batch.
        return x, y, y, 1.0
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def train_epoch_mil(
    model,
    loader,
    optimizer,
    epoch,
    device,
    logger,
    scaler=None,
    accumulation_steps: int = 1,
    mixup_alpha: float = 0.0,
    bag_dropout: float = 0.0,
    **_unused,
):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    all_preds: List[int] = []
    all_targets: List[int] = []

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer.zero_grad()
    mixup_active = float(mixup_alpha) > 0.0
    bag_dropout = float(bag_dropout)
    if mixup_active and epoch == 1:
        logger.info(f"[MIL] Mixup active with alpha={mixup_alpha:.3f} (Beta({mixup_alpha},{mixup_alpha}))")
    if bag_dropout > 0.0 and epoch == 1:
        logger.info(f"[MIL] Bag-instance dropout active: drop_prob={bag_dropout:.3f}")
    nonfinite_count = 0
    nonfinite_grad_steps = 0

    for idx, batch_data in enumerate(loader):
        data, target, _ = batch_data
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if bag_dropout > 0.0:
            data = _bag_instance_dropout(data, drop_prob=bag_dropout)
        data_mix, y_a, y_b, lam = _mixup_bags(data, target, alpha=float(mixup_alpha))

        with torch.amp.autocast(enabled=(scaler is not None), device_type="cuda"):
            logits = model(data_mix, return_segmentation=False)
            if mixup_active:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y_a)

            preds = torch.argmax(logits.detach(), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y_a.detach().cpu().tolist())

            unscaled_loss = loss.item()
            loss = loss / accumulation_steps

        # Skip non-finite losses so one overflow doesn't poison the running
        # mean (AverageMeter is sticky-NaN) and to avoid propagating NaN
        # grads into the optimizer on the next accumulation boundary.
        if not math.isfinite(unscaled_loss):
            nonfinite_count += 1
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(loader):
                optimizer.zero_grad()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            # clip_grad_norm_ returns NaN-total when any grad is NaN/Inf but
            # does NOT zero the bad grads, so the optimizer step would still
            # corrupt the weights. Check once across the whole parameter set
            # and skip the step if anything is non-finite.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(total_norm):
                nonfinite_grad_steps += 1
                optimizer.zero_grad()
                if scaler is not None:
                    # Tell the scaler this step was bad so it adjusts its scale.
                    scaler.update()
                continue
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        run_loss.update(unscaled_loss, n=data.size(0))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(loader):
            msg = (
                f"[MIL] Epoch {epoch} [{idx + 1}/{len(loader)}] "
                f"Loss: {run_loss.avg:.4f} Time: {time.time() - start_time:.2f}s"
            )
            print(msg)
            logger.info(msg)
            start_time = time.time()

    metrics = _compute_classification_metrics(all_targets, all_preds)
    summary = (
        f"[MIL] Train Epoch {epoch} (bag-level) => Loss: {run_loss.avg:.4f}, "
        f"Acc: {metrics['accuracy']:.4f}, MacroF1: {metrics['macro_f1']:.4f}"
    )
    print(summary)
    logger.info(summary)
    if nonfinite_count > 0 or nonfinite_grad_steps > 0:
        logger.warning(
            f"[MIL] Epoch {epoch}: skipped {nonfinite_count}/{len(loader)} batches with "
            f"non-finite loss and {nonfinite_grad_steps} optimizer steps with non-finite grads."
        )
    return run_loss.avg, metrics["macro_f1"]


@torch.no_grad()
def validate_epoch_mil(
    model, loader, device, logger,
    return_probabilities: bool = False,
    compute_ci: bool = True,
    n_boot: int = 1000,
):
    """Validate at bag level. One patient = one bag on BigLunge, so
    bag-level metrics ARE patient-level metrics. We still emit a
    ``patient_level`` sub-dict so the generic ``run_training_phase`` in
    ``main.py`` can monitor patient metrics for checkpoint selection without
    special-casing MIL.
    """
    model.eval()
    run_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []
    all_patient_ids: List[Any] = []
    all_volume_ids: List[Any] = []
    # Attention diagnostics — only valid for att / att_trans MIL modes.
    # entropy is normalized by log(N) so 1.0 == uniform attention,
    # 0.0 == one-hot. top1_mass / top3_mass quantify concentration.
    attn_entropies: List[float] = []
    attn_top1_mass: List[float] = []
    attn_top3_mass: List[float] = []
    attn_warned: bool = False

    print("\n[MIL] Starting validation...")
    logger.info("[MIL] Starting validation...")

    for batch_data in loader:
        data, target, meta = batch_data
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits = model(data)
        loss = criterion(logits, target)
        run_loss.update(loss.item(), n=data.size(0))

        if hasattr(model, "attention_weights"):
            try:
                a = model.attention_weights(data)  # (B, N), softmax already applied
                N = int(a.shape[1])
                if N >= 2:
                    log_n = float(np.log(N))
                    ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(dim=1) / log_n
                    topk = a.sort(dim=1, descending=True).values
                    attn_entropies.extend(ent.cpu().tolist())
                    attn_top1_mass.extend(topk[:, 0].cpu().tolist())
                    if N >= 3:
                        attn_top3_mass.extend(topk[:, :3].sum(dim=1).cpu().tolist())
            except Exception as e:
                if not attn_warned:
                    logger.debug(f"[MIL] attention diagnostics unavailable: {e}")
                    attn_warned = True

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        tgts = target.cpu().numpy()
        for i in range(len(meta)):
            all_preds.append(int(preds[i]))
            all_targets.append(int(tgts[i]))
            all_probs.append(probs[i])
            all_patient_ids.append(meta[i].get("patient_id"))
            all_volume_ids.append(meta[i].get("volume_id"))

    metrics = _compute_classification_metrics(all_targets, all_preds)
    num_classes = metrics["num_classes"]

    val_msg = (
        f"[MIL] Val (bag-level, n={len(all_targets)}) => Loss: {run_loss.avg:.4f}, "
        f"Acc: {metrics['accuracy']:.4f}, BalancedAcc: {metrics['balanced_accuracy']:.4f}, "
        f"MacroPrecision: {metrics['macro_precision']:.4f}, MacroRecall: {metrics['macro_recall']:.4f}, "
        f"MacroF1: {metrics['macro_f1']:.4f}"
    )
    print(val_msg)
    logger.info(val_msg)

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

    if len(all_targets) > 0:
        print("\n[MIL] Bag-level Confusion Matrix:")
        logger.info("\n[MIL] Bag-level Confusion Matrix:")
        header = f"{'True \\ Pred':<18}" + "".join([f"{display_names[j]:<16}" for j in range(num_classes)])
        print(header); logger.info(header)
        for i in range(num_classes):
            row = f"Y:{display_names[i]:<16}" + "".join(f"{int(metrics['conf_matrix'][i, j]):<16}" for j in range(num_classes))
            print(row); logger.info(row)
        print("")

    if attn_entropies:
        mean_ent = float(np.mean(attn_entropies))
        mean_t1 = float(np.mean(attn_top1_mass))
        mean_t3 = float(np.mean(attn_top3_mass)) if attn_top3_mass else float("nan")
        attn_msg = (
            f"[MIL] Attention (n_bags={len(attn_entropies)}): "
            f"entropy={mean_ent:.3f} (1.0=uniform, 0.0=one-hot), "
            f"top1_mass={mean_t1:.3f}, top3_mass={mean_t3:.3f}"
        )
        print(attn_msg)
        logger.info(attn_msg)

    # The monitor loop in main.py reads from result['patient_level'] when it
    # exists. For MIL, patient == bag, so mirror the top-level numbers here.
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
        "patient_level": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "per_class_precision": metrics["per_class_precision"].tolist(),
            "per_class_recall": metrics["per_class_recall"].tolist(),
            "per_class_f1": metrics["per_class_f1"].tolist(),
            "num_patients": len(all_targets),
        },
    }
    if attn_entropies:
        result["attention"] = {
            "entropy_mean": float(np.mean(attn_entropies)),
            "top1_mass_mean": float(np.mean(attn_top1_mass)),
            "top3_mass_mean": float(np.mean(attn_top3_mass)) if attn_top3_mass else None,
            "n_bags": len(attn_entropies),
        }
        # Surface in patient_level too so metrics.jsonl rows pick it up via
        # the existing val_patient capture in run_training_phase.
        result["patient_level"]["attention"] = result["attention"]

    if compute_ci and len(all_targets) > 0:
        _, mf1_lo, mf1_hi = bootstrap_ci(
            all_targets, all_preds, _macro_f1, n_boot=n_boot, rng_seed=0
        )
        _, bacc_lo, bacc_hi = bootstrap_ci(
            all_targets, all_preds, _bacc, n_boot=n_boot, rng_seed=0
        )
        pc_f1_ci = per_class_f1_ci(
            all_targets, all_preds, num_classes=num_classes,
            n_boot=n_boot, rng_seed=0,
        )
        result["patient_level"]["macro_f1_ci95"] = [mf1_lo, mf1_hi]
        result["patient_level"]["balanced_accuracy_ci95"] = [bacc_lo, bacc_hi]
        result["patient_level"]["per_class_f1_ci95"] = [list(pair) for pair in pc_f1_ci]
        result["patient_level"]["ci_n_boot"] = int(n_boot)
        logger.info(
            f"[MIL] Bootstrap CI (n_boot={n_boot}): "
            f"MacroF1={metrics['macro_f1']:.4f} [{mf1_lo:.4f}, {mf1_hi:.4f}], "
            f"BalAcc={metrics['balanced_accuracy']:.4f} [{bacc_lo:.4f}, {bacc_hi:.4f}]"
        )

    if return_probabilities:
        samples = []
        class_prob_sums = np.zeros(num_classes, dtype=np.float64)
        pred_hist = np.zeros(num_classes, dtype=np.int64)
        for i, probs in enumerate(all_probs):
            probs_np = np.asarray(probs, dtype=np.float64)
            if probs_np.shape[0] != num_classes:
                aligned = np.zeros(num_classes, dtype=np.float64)
                n_copy = min(num_classes, probs_np.shape[0])
                aligned[:n_copy] = probs_np[:n_copy]
                probs_np = aligned
            class_prob_sums += probs_np
            pl = all_preds[i]
            if 0 <= pl < num_classes:
                pred_hist[pl] += 1
            samples.append({
                "sample_index": i,
                "patient_id": all_patient_ids[i],
                "volume_id": all_volume_ids[i],
                "true_label": int(all_targets[i]),
                "true_name": display_names[int(all_targets[i])] if 0 <= int(all_targets[i]) < num_classes else f"Class{int(all_targets[i])}",
                "pred_label": int(all_preds[i]),
                "pred_name": display_names[int(all_preds[i])] if 0 <= int(all_preds[i]) < num_classes else f"Class{int(all_preds[i])}",
                "confidence": float(np.max(probs_np)) if probs_np.size > 0 else 0.0,
                "probabilities": {display_names[c]: float(probs_np[c]) for c in range(num_classes)},
            })
        n = max(1, len(samples))
        mean_probs_dict = {display_names[c]: float(class_prob_sums[c] / n) for c in range(num_classes)}
        pred_counts_dict = {display_names[c]: int(pred_hist[c]) for c in range(num_classes)}
        pred_fracs_dict = {display_names[c]: float(pred_hist[c] / n) for c in range(num_classes)}
        result["inference_probabilities"] = {
            "num_samples": len(samples),
            "class_names": display_names,
            "mean_probability_per_class": mean_probs_dict,
            "predicted_class_counts": pred_counts_dict,
            "predicted_class_fractions": pred_fracs_dict,
            "samples": samples,
        }

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        mem_msg = f"[MIL] VRAM - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        print(mem_msg)
        logger.info(mem_msg)

    return result
