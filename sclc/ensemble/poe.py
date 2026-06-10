"""Product-of-experts ensemble for the SCLC binary + ternary MIL pipeline.

Ternary model: 3 classes — 0=Adenocarcinoma, 1=Small Cell (SCLC), 2=Squamous
Binary  model: 2 classes — 0=non-SCLC,       1=SCLC

Symmetric PoE (single alpha):

    unnorm[ADC]  = p_tern[0] * p_bin[nonSCLC]^alpha
    unnorm[SCLC] = p_tern[1] * p_bin[SCLC]^alpha
    unnorm[SCC]  = p_tern[2] * p_bin[nonSCLC]^alpha
    p_final      = unnorm / sum(unnorm)

Asymmetric PoE (alpha_sclc ≠ alpha_nonsclc):

    unnorm[ADC]  = p_tern[0] * p_bin[nonSCLC]^alpha_nonsclc
    unnorm[SCLC] = p_tern[1] * p_bin[SCLC]^alpha_sclc
    unnorm[SCC]  = p_tern[2] * p_bin[nonSCLC]^alpha_nonsclc

Calibration: grid-search (alpha_sclc, alpha_nonsclc) maximising val macro-F1.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

# --------------------------------------------------------------------------- #
# Class-index conventions (must match the models' output ordering)            #
# --------------------------------------------------------------------------- #
TERNARY_CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]
BINARY_CLASS_NAMES  = ["non-SCLC", "SCLC"]
SCLC_IDX_TERNARY = 1   # index of SCLC in 3-class output
SCLC_IDX_BINARY  = 1   # index of SCLC in 2-class output


# --------------------------------------------------------------------------- #
# Core PoE maths                                                               #
# --------------------------------------------------------------------------- #

def poe_combine(
    p_ternary: np.ndarray,
    p_binary: np.ndarray,
    alpha: float = 1.0,
    alpha_sclc: Optional[float] = None,
    alpha_nonsclc: Optional[float] = None,
) -> np.ndarray:
    """Combine ternary (N, 3) and binary (N, 2) softmax probabilities.

    alpha_sclc    — power applied to p_bin[SCLC] for the SCLC unnorm term.
                    Defaults to alpha (symmetric PoE).
    alpha_nonsclc — power applied to p_bin[nonSCLC] for ADC/SCC terms.
                    Defaults to alpha (symmetric PoE).

    Setting alpha_sclc > alpha_nonsclc lets the binary arm preside over
    SCLC decisions while the ternary governs ADC vs SCC discrimination.

    Returns (N, 3) combined probabilities in the same class order as p_ternary.
    """
    p_t = np.asarray(p_ternary, dtype=np.float64)   # (N, 3)
    p_b = np.asarray(p_binary,  dtype=np.float64)   # (N, 2)

    if p_t.ndim == 1:
        p_t = p_t[None]
    if p_b.ndim == 1:
        p_b = p_b[None]

    a_sclc = float(alpha_sclc   if alpha_sclc    is not None else alpha)
    a_ns   = float(alpha_nonsclc if alpha_nonsclc is not None else alpha)

    p_nonSCLC = p_b[:, [0]] ** a_ns    # (N, 1) — ADC/SCC weight
    p_SCLC_b  = p_b[:, [1]] ** a_sclc  # (N, 1) — SCLC weight

    # ADC→non-SCLC factor, SCLC→SCLC factor, SCC→non-SCLC factor
    factors = np.concatenate([p_nonSCLC, p_SCLC_b, p_nonSCLC], axis=1)  # (N, 3)

    unnorm = p_t * factors
    z = unnorm.sum(axis=1, keepdims=True)
    z = np.maximum(z, 1e-12)
    return (unnorm / z).squeeze()


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #

def calibrate_alpha(
    val_ternary: np.ndarray,
    val_binary: np.ndarray,
    val_labels: Sequence[int],
    alpha_range: Optional[np.ndarray] = None,
    asymmetric: bool = False,
) -> Tuple[float, float, float]:
    """Grid-search alpha that maximises val macro-F1.

    asymmetric=False (default): 1-D search over a single alpha (backward-
        compatible).  Returns (best_alpha, best_alpha, best_macro_f1) so the
        caller can always unpack three values.

    asymmetric=True: 2-D grid search over (alpha_sclc, alpha_nonsclc)
        independently.  Returns (best_alpha_sclc, best_alpha_nonsclc, best_f1).
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.0, 3.0, 61)

    y = np.asarray(val_labels, dtype=np.int64)
    best_a_sclc, best_a_ns, best_f1 = 1.0, 1.0, -1.0

    if not asymmetric:
        for a in alpha_range:
            combined = poe_combine(val_ternary, val_binary, alpha=float(a))
            preds = np.argmax(combined, axis=-1) if combined.ndim > 1 else np.array([np.argmax(combined)])
            f1 = float(f1_score(y, preds, average="macro", zero_division=0))
            if f1 > best_f1:
                best_f1, best_a_sclc, best_a_ns = f1, float(a), float(a)
    else:
        for a_sclc in alpha_range:
            for a_ns in alpha_range:
                combined = poe_combine(val_ternary, val_binary,
                                       alpha_sclc=float(a_sclc),
                                       alpha_nonsclc=float(a_ns))
                preds = np.argmax(combined, axis=-1) if combined.ndim > 1 else np.array([np.argmax(combined)])
                f1 = float(f1_score(y, preds, average="macro", zero_division=0))
                if f1 > best_f1:
                    best_f1, best_a_sclc, best_a_ns = f1, float(a_sclc), float(a_ns)

    return best_a_sclc, best_a_ns, best_f1


# --------------------------------------------------------------------------- #
# Probability JSON I/O                                                         #
# --------------------------------------------------------------------------- #

def load_prob_json(path: str | Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load an inference-probability JSON produced by sclc.main.

    Returns:
        patient_ids  — list of N patient ID strings
        probs        — (N, C) float64 array of softmax probabilities
        true_labels  — (N,) int64 array of ground-truth class indices
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # The JSON may be saved at top-level or nested under "patient_level"
    samples = payload.get("samples")
    if not samples:
        pl = payload.get("patient_level") or {}
        samples = pl.get("samples") or []
    if not samples:
        raise ValueError(f"No 'samples' found in {path}")

    class_names = payload.get("class_names") or []
    if not class_names and samples:
        class_names = list((samples[0].get("probabilities") or {}).keys())

    n = len(samples)
    c = len(class_names)

    patient_ids = []
    probs       = np.zeros((n, c), dtype=np.float64)
    true_labels = np.zeros(n, dtype=np.int64)

    prob_lookup = {name: i for i, name in enumerate(class_names)}

    for i, s in enumerate(samples):
        patient_ids.append(str(s.get("patient_id", s.get("volume_id", i))))
        true_labels[i] = int(s.get("true_label", 0))
        p_dict = s.get("probabilities") or {}
        for name, col in prob_lookup.items():
            probs[i, col] = float(p_dict.get(name, 0.0))

    return patient_ids, probs, true_labels


def _align_patients(
    ids_a: List[str], probs_a: np.ndarray, labels_a: np.ndarray,
    ids_b: List[str], probs_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs_a, probs_b, labels) aligned on the intersection of patient IDs."""
    set_a = {pid: i for i, pid in enumerate(ids_a)}
    common = [pid for pid in ids_b if pid in set_a]
    if not common:
        raise ValueError("No overlapping patient IDs between ternary and binary probability files.")

    idx_a = [set_a[pid]                    for pid in common]
    idx_b = [ids_b.index(pid)              for pid in common]

    return probs_a[idx_a], probs_b[idx_b], labels_a[idx_a]


# --------------------------------------------------------------------------- #
# Per-fold evaluation                                                          #
# --------------------------------------------------------------------------- #

def poe_evaluate_fold(
    ternary_prob_path: str | Path,
    binary_prob_path: str | Path,
    alpha: Optional[float] = None,
    alpha_sclc: Optional[float] = None,
    alpha_nonsclc: Optional[float] = None,
    val_ternary_path: Optional[str | Path] = None,
    val_binary_path: Optional[str | Path] = None,
    asymmetric: bool = False,
) -> Dict:
    """Evaluate PoE ensemble on one fold's test set.

    alpha          — symmetric alpha (used when alpha_sclc/alpha_nonsclc are None)
    alpha_sclc     — override: binary power for the SCLC class only
    alpha_nonsclc  — override: binary power for ADC/SCC classes only
    asymmetric     — when calibrating from val, do a 2-D grid search

    If alpha/alpha_sclc/alpha_nonsclc are all None and val paths are provided,
    alpha is calibrated on the val set.  Otherwise defaults to alpha=1.0.
    """
    t_ids, t_probs, t_labels = load_prob_json(ternary_prob_path)
    b_ids, b_probs, _        = load_prob_json(binary_prob_path)
    t_probs_a, b_probs_a, labels = _align_patients(t_ids, t_probs, t_labels,
                                                    b_ids, b_probs)

    # Explicit asymmetric alphas override calibration
    if alpha_sclc is not None or alpha_nonsclc is not None:
        a_s  = float(alpha_sclc   if alpha_sclc    is not None else (alpha or 1.0))
        a_ns = float(alpha_nonsclc if alpha_nonsclc is not None else (alpha or 1.0))
    elif alpha is None:
        if val_ternary_path and val_binary_path:
            vt_ids, vt_probs, vt_labels = load_prob_json(val_ternary_path)
            vb_ids, vb_probs, _         = load_prob_json(val_binary_path)
            vt_a, vb_a, vl = _align_patients(vt_ids, vt_probs, vt_labels,
                                              vb_ids, vb_probs)
            a_s, a_ns, _ = calibrate_alpha(vt_a, vb_a, vl, asymmetric=asymmetric)
            mode = "asymmetric" if asymmetric else "symmetric"
            print(f"[PoE] Calibrated ({mode}) alpha_sclc={a_s:.2f} alpha_nonsclc={a_ns:.2f} (n={len(vl)})")
        else:
            a_s, a_ns = 1.0, 1.0
    else:
        a_s = a_ns = float(alpha)

    combined = poe_combine(t_probs_a, b_probs_a, alpha_sclc=a_s, alpha_nonsclc=a_ns)
    preds    = np.argmax(combined, axis=-1)

    mf1  = float(f1_score(labels, preds, average="macro",    zero_division=0))
    acc  = float(accuracy_score(labels, preds))
    bacc = float(balanced_accuracy_score(labels, preds))
    pcf1 = f1_score(labels, preds, average=None, zero_division=0).tolist()
    cm   = confusion_matrix(labels, preds, labels=list(range(len(TERNARY_CLASS_NAMES)))).tolist()

    return {
        "alpha_used":        a_s,        # kept for backward compat
        "alpha_sclc":        a_s,
        "alpha_nonsclc":     a_ns,
        "n_patients":        int(len(labels)),
        "macro_f1":          mf1,
        "accuracy":          acc,
        "balanced_accuracy": bacc,
        "per_class_f1":      pcf1,
        "confusion_matrix":  cm,
    }


# --------------------------------------------------------------------------- #
# CV aggregate helper                                                          #
# --------------------------------------------------------------------------- #

def poe_cv_summary(fold_results: List[Dict]) -> Dict:
    """Compute mean ± std across folds from a list of poe_evaluate_fold outputs."""
    mf1s  = [r["macro_f1"]          for r in fold_results]
    accs  = [r["accuracy"]           for r in fold_results]
    baccs = [r["balanced_accuracy"]  for r in fold_results]
    alphas = [r["alpha_used"]        for r in fold_results]

    def _stats(vals):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    mf1_m,  mf1_s  = _stats(mf1s)
    acc_m,  acc_s  = _stats(accs)
    bacc_m, bacc_s = _stats(baccs)

    n_classes = len(fold_results[0]["per_class_f1"]) if fold_results else 3
    pcf1_means = []
    for c in range(n_classes):
        vals = [r["per_class_f1"][c] for r in fold_results if c < len(r["per_class_f1"])]
        pcf1_means.append(statistics.mean(vals) if vals else 0.0)

    return {
        "n_folds":            len(fold_results),
        "macro_f1_mean":      mf1_m,
        "macro_f1_std":       mf1_s,
        "accuracy_mean":      acc_m,
        "accuracy_std":       acc_s,
        "balanced_accuracy_mean": bacc_m,
        "balanced_accuracy_std":  bacc_s,
        "per_class_f1_mean":  pcf1_means,
        "class_names":        TERNARY_CLASS_NAMES,
        "alpha_per_fold":     alphas,
        "alpha_mean":         statistics.mean(alphas),
    }
