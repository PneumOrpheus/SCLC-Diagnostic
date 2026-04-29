"""Bootstrap confidence intervals for classification metrics on small val sets.

Used for the 52-patient Lung-PET-CT-Dx validation split, where a 2-3 percentage
point difference in MacroF1 is well inside sampling noise. Reporting point
estimates without a CI on a set that small is not defensible.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple

import numpy as np


def bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    metric_fn: Callable[[List[int], List[int]], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    stratified: bool = True,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (point, lo, hi) where lo/hi are alpha/2 and 1-alpha/2 quantiles.

    stratified=True resamples within each class so rare classes (SC=6, Sq=9
    on the DAPT val split) are always represented in every resample. For macro
    metrics this is the correct default — unstratified resamples can drop a
    rare class entirely and blow up the CI.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(rng_seed)

    if stratified:
        idx_by_cls: Dict[int, np.ndarray] = {
            int(c): np.where(y_true_arr == c)[0] for c in np.unique(y_true_arr)
        }

    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        if stratified:
            idx_parts = [
                rng.choice(idx, size=len(idx), replace=True)
                for idx in idx_by_cls.values()
                if len(idx) > 0
            ]
            idx = np.concatenate(idx_parts)
        else:
            idx = rng.choice(n, size=n, replace=True)
        samples[b] = metric_fn(y_true_arr[idx].tolist(), y_pred_arr[idx].tolist())

    point = float(metric_fn(y_true_arr.tolist(), y_pred_arr.tolist()))
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return point, lo, hi


def per_class_f1_ci(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> List[Tuple[float, float]]:
    """Stratified bootstrap CIs for per-class F1. Returns [(lo, hi), ...].

    Each class's F1 is a function of the full prediction array (needs false
    positives from *other* classes), so we resample the full vector and
    recompute all class F1s per replicate rather than resampling per class.
    """
    from sklearn.metrics import f1_score

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)
    if n == 0:
        return [(0.0, 0.0)] * num_classes
    rng = np.random.default_rng(rng_seed)

    idx_by_cls: Dict[int, np.ndarray] = {
        int(c): np.where(y_true_arr == c)[0] for c in np.unique(y_true_arr)
    }

    samples = np.zeros((n_boot, num_classes), dtype=np.float64)
    labels = list(range(num_classes))
    for b in range(n_boot):
        idx_parts = [
            rng.choice(idx, size=len(idx), replace=True)
            for idx in idx_by_cls.values()
            if len(idx) > 0
        ]
        idx = np.concatenate(idx_parts)
        f1s = f1_score(
            y_true_arr[idx], y_pred_arr[idx],
            labels=labels, average=None, zero_division=0,
        )
        samples[b] = f1s

    return [
        (float(np.quantile(samples[:, c], alpha / 2.0)),
         float(np.quantile(samples[:, c], 1.0 - alpha / 2.0)))
        for c in range(num_classes)
    ]
