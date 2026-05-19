"""Build the final thesis-results tree from the 2026-05 master CV runs.

This consumes the per-fold artefacts written by the new ``run_master.sh``
orchestrator (and its resume / gap-fill siblings) and emits the LaTeX
tables and PDF figures cited in ``Sections/Results.tex``.

Reads from (per arm):
  - results/output_master_base/<pipeline>/<model>/                   (baseline arm)
  - results/output_master_fpn/<pipeline>/<model>/                    (FPN arm)

Each model directory contains, per fold k in {0..4}:
  - metrics.jsonl with the per-fold ``test_fold_k`` / ``dapt_test_fold_k`` rows
  - <model>_<ts>_dapt_fold<k>_inference_probabilities.json    (LPCD test)
  - <model>_<ts>__fold<k>_inference_probabilities.json        (BigLunge test)
  - checkpoints/fold_<k>/...

Pools per-fold patient predictions into a single union-of-folds set
(each LPCD or BigLunge patient appears exactly once, matching the
methodology in Section 4.7) and computes patient-level metrics with a
stratified non-parametric bootstrap 95% CI (``n_boot=1000``).

Also emits, separately, the per-fold variance table (mean / SE across
the five folds) that Section 4.10 promises.

Writes to:
  results/thesis_final/
  ├── per_config/<arm>/<pipeline>/<model>/
  │     ├── union_predictions_lpcd.csv          y_true, y_pred, prob_*
  │     ├── union_predictions_biglunge.csv
  │     ├── fold_coverage.json                  which folds are present
  │     ├── union_metrics.json                  pooled metrics + bootstrap CI
  │     ├── per_fold_metrics.json               per-fold metrics for the variance view
  │     ├── confusion_lpcd.csv / confusion_biglunge.csv
  │     └── (symlinks to source inference JSONs for traceability)
  ├── tables/
  │     ├── table_overall.{csv,tex}             headline (acc, bal_acc, macro_f1, macro_auc)
  │     ├── table_per_class_f1.{csv,tex}
  │     ├── table_per_class_auc.{csv,tex}
  │     ├── table_fpn_ablation.{csv,tex}        baseline vs FPN paired comparison
  │     ├── table_per_fold_variance.{csv,tex}   5-fold macro_f1 + mean ± SE
  │     └── table_literature_anchors.{csv,tex}  LPCD: our row + Honda/Dunn/Jacob
  ├── figures/
  │     ├── fig_overall_macro_f1.pdf
  │     ├── fig_per_class_f1.pdf
  │     ├── fig_per_class_auc.pdf
  │     ├── fig_dapt_test_gap.pdf
  │     ├── fig_confusion_matrices.pdf
  │     ├── fig_training_curves.pdf
  │     └── fig_fpn_delta.pdf
  ├── coverage_report.md                        which (arm, model, fold, cohort) is present
  └── README.md                                 provenance: git SHA, dates, n_boot, etc.

Idempotent — re-run after gap-fills to refresh in place.

Usage::

    python -m sclc.scripts.build_final_results            # or:
    python scripts/build_final_results.py
    python scripts/build_final_results.py --arms base     # baseline-only
    python scripts/build_final_results.py --skip-figures  # tables only
    python scripts/build_final_results.py --n-boot 5000   # tighter CIs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Make repo importable when called directly.
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# =============================================================================
# Active matrix
# =============================================================================

CLASS_NAMES: List[str] = ["Adenocarcinoma", "Small Cell", "Squamous"]
CLASS_SHORT: List[str] = ["ADC", "SCLC", "SCC"]
CLASS_COLORS: Dict[str, str] = {
    "Adenocarcinoma": "#4E79A7",
    "Small Cell":     "#F28E2B",
    "Squamous":       "#59A14F",
}

@dataclass(frozen=True)
class Config:
    """One row of the experimental matrix."""
    arm: str             # "base" or "fpn"
    pipeline: str        # "2d" / "mil" / "3d"
    model_type: str      # e.g. "efficientnet_b0_2d"
    backbone_label: str  # e.g. "EfficientNet-B0"
    pretrain_label: str  # e.g. "ImageNet" / "BTCV"
    pipeline_label: str  # e.g. "2D"


def _matrix() -> List[Config]:
    rows: List[Config] = []
    base_specs = [
        ("2d", "efficientnet_b0_2d", "EfficientNet-B0", "ImageNet", "2D"),
        ("2d", "resnet50_2d",        "ResNet-50",       "ImageNet", "2D"),
        ("2d", "densenet121_2d",     "DenseNet-121",    "ImageNet", "2D"),
        ("2d", "swinv2_tiny_2d",     "SwinV2-Tiny",     "ImageNet", "2D"),
        ("mil", "mil_swinv2_tiny",   "SwinV2-Tiny",     "ImageNet", "MIL"),
        ("3d", "swin_unetr",         "Swin\\,UNETR",    "BTCV",     "3D"),
    ]
    for arm in ("base", "fpn"):
        for pipeline, mtype, bb, pre, plabel in base_specs:
            rows.append(Config(arm=arm, pipeline=pipeline, model_type=mtype,
                               backbone_label=bb, pretrain_label=pre,
                               pipeline_label=plabel))
    return rows


# =============================================================================
# Paths
# =============================================================================

REPO_ROOT = Path(_REPO)
OUTPUT_ROOTS: Dict[str, Path] = {
    "base": REPO_ROOT / "results" / "output_master_base",
    "fpn":  REPO_ROOT / "results" / "output_master_fpn",
}
RESULTS_ROOT = REPO_ROOT / "results" / "thesis_final"
N_FOLDS = 5


def _model_dir(cfg: Config) -> Path:
    return OUTPUT_ROOTS[cfg.arm] / cfg.pipeline / cfg.model_type


def _per_config_dir(cfg: Config) -> Path:
    return RESULTS_ROOT / "per_config" / cfg.arm / cfg.pipeline / cfg.model_type


# =============================================================================
# Loading helpers
# =============================================================================

def _load_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[metrics] skip malformed row in {path}: {exc}", file=sys.stderr)
    return rows


def _latest_glob(d: Path, pattern: str) -> Optional[Path]:
    """Latest-by-mtime match of ``pattern`` under ``d``, or None."""
    if not d.is_dir():
        return None
    matches = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def find_fold_inference_files(cfg: Config) -> Dict[str, Dict[int, Optional[Path]]]:
    """Return ``{cohort: {fold: path or None}}`` for both cohorts and 5 folds."""
    out: Dict[str, Dict[int, Optional[Path]]] = {"lpcd": {}, "biglunge": {}}
    d = _model_dir(cfg)
    if not d.is_dir():
        for k in range(N_FOLDS):
            out["lpcd"][k] = None
            out["biglunge"][k] = None
        return out
    for k in range(N_FOLDS):
        # LPCD test files carry "_dapt_foldK_" in the name.
        out["lpcd"][k] = _latest_glob(
            d, f"{cfg.model_type}_*_dapt_fold{k}_inference_probabilities.json"
        )
        # BigLunge test files carry "__foldK_" (no phase token between the
        # timestamp and the fold suffix; double underscore is the marker).
        out["biglunge"][k] = _latest_glob(
            d, f"{cfg.model_type}_*__fold{k}_inference_probabilities.json"
        )
    return out


def _extract_patient_samples(probs_path: Path) -> List[Dict[str, Any]]:
    """Patient-level samples from an inference-probabilities JSON.

    Three input shapes are handled uniformly:

    1. ``patient_level.samples`` is non-empty (the 2D pipeline on either
       cohort) — returned verbatim.
    2. Top-level ``samples`` is already patient-level (the MIL and 3D
       pipelines on BigLunge, where there is exactly one volume per
       patient) — returned verbatim.
    3. Top-level ``samples`` is volume-level with patient IDs repeated
       (the MIL and 3D pipelines on Lung-PET-CT-Dx, where the multi-scan
       cap of two volumes per ADC/SCC patient means a patient may
       contribute one or two volumes) — aggregated to patient-level by
       averaging the per-volume softmax vectors within each patient_id
       and re-taking the argmax. This matches the patient-level
       aggregation rule in Methodology §4.7.
    """
    try:
        payload = json.loads(probs_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[probs] failed to parse {probs_path}: {exc}", file=sys.stderr)
        return []
    pat = (payload.get("patient_level") or {}).get("samples") or []
    if pat:
        return pat
    samples = payload.get("samples") or []
    if not samples:
        return []
    # Are patient IDs repeated? If yes, fall back to manual per-patient
    # aggregation; if no, the top-level rows are already patient-level.
    pids = [s.get("patient_id") for s in samples]
    if len(set(pids)) == len(pids):
        return samples
    return _aggregate_to_patient_level(samples)


def _aggregate_to_patient_level(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Average per-volume softmax probabilities within each patient_id,
    then take the argmax to obtain a patient-level prediction. Drops
    rows that share a patient_id but disagree on the true label (a
    data-integrity guard — should not happen in practice)."""
    by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        pid = s.get("patient_id")
        if pid is None:
            continue
        by_pid.setdefault(pid, []).append(s)
    out: List[Dict[str, Any]] = []
    for pid, group in by_pid.items():
        true_labels = {int(g.get("true_label", -1)) for g in group}
        if len(true_labels) != 1:
            print(f"[aggregate] skipping {pid}: inconsistent true labels "
                  f"{true_labels}", file=sys.stderr)
            continue
        true_label = next(iter(true_labels))
        # Average per-class probabilities across volumes.
        sums = {c: 0.0 for c in CLASS_NAMES}
        for g in group:
            probs = g.get("probabilities") or {}
            for c in CLASS_NAMES:
                sums[c] += float(probs.get(c, 0.0))
        n = float(len(group))
        avg = {c: sums[c] / n for c in CLASS_NAMES}
        pred_label = int(np.argmax([avg[c] for c in CLASS_NAMES]))
        out.append({
            "patient_id": pid,
            "true_label": true_label,
            "true_name": CLASS_NAMES[true_label] if 0 <= true_label < 3 else "?",
            "pred_label": pred_label,
            "pred_name": CLASS_NAMES[pred_label],
            "confidence": float(avg[CLASS_NAMES[pred_label]]),
            "probabilities": avg,
            "num_volumes": len(group),
        })
    return out


def samples_to_arrays(
    samples: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Returns ``(y_true, y_pred, y_score, patient_ids)``.

    ``y_score`` is ``(N, 3)`` with columns indexed by ``CLASS_NAMES``.
    Rows with malformed labels are dropped silently.
    """
    y_t: List[int] = []
    y_p: List[int] = []
    y_s: List[List[float]] = []
    ids: List[str] = []
    for s in samples:
        t = s.get("true_label")
        p = s.get("pred_label")
        probs = s.get("probabilities") or {}
        if t is None or p is None:
            continue
        try:
            ti, pi = int(t), int(p)
        except (TypeError, ValueError):
            continue
        if not (0 <= ti < 3 and 0 <= pi < 3):
            continue
        row = [float(probs.get(c, 0.0)) for c in CLASS_NAMES]
        y_t.append(ti)
        y_p.append(pi)
        y_s.append(row)
        ids.append(str(s.get("patient_id", "")))
    return (np.asarray(y_t, dtype=np.int64),
            np.asarray(y_p, dtype=np.int64),
            np.asarray(y_s, dtype=np.float64),
            ids)


# =============================================================================
# Metric helpers
# =============================================================================

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).sum()) / float(len(y_true))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    if classes.size == 0:
        return float("nan")
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).sum()) / float(mask.sum()))
    return float(np.mean(recalls)) if recalls else float("nan")


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    out: List[float] = []
    for c in range(3):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = 2 * tp + fp + fn
        out.append(float(2 * tp) / denom if denom > 0 else 0.0)
    return out


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = _per_class_f1(y_true, y_pred)
    return float(np.mean(f1s))


def _per_class_auc_and_macro(
    y_true: np.ndarray, y_score: np.ndarray,
) -> Tuple[List[float], float]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return ([float("nan")] * 3, float("nan"))
    per_class: List[float] = []
    for c in range(3):
        y_bin = (y_true == c).astype(np.int64)
        if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
            per_class.append(float("nan"))
            continue
        try:
            per_class.append(float(roc_auc_score(y_bin, y_score[:, c])))
        except Exception:
            per_class.append(float("nan"))
    # Macro AUC over classes that are computable.
    finite = [v for v in per_class if np.isfinite(v)]
    macro = float(np.mean(finite)) if finite else float("nan")
    return per_class, macro


def _stratified_bootstrap(
    metric_fn,
    y_true: np.ndarray,
    *score_args: np.ndarray,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Stratified non-parametric bootstrap CI on a metric function.

    Resamples within each true-label class so all classes are present in
    every replicate. ``metric_fn`` is called as ``metric_fn(y_true_b, *args_b)``
    where each arg in ``*score_args`` is index-aligned with ``y_true``.
    """
    point = float(metric_fn(y_true, *score_args))
    if y_true.size == 0 or not np.isfinite(point):
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(rng_seed)
    classes, class_counts = np.unique(y_true, return_counts=True)
    class_idx = {int(c): np.where(y_true == c)[0] for c in classes}

    boot_values: List[float] = []
    for _ in range(n_boot):
        parts = []
        for c, idx in class_idx.items():
            if idx.size == 0:
                continue
            parts.append(rng.choice(idx, size=idx.size, replace=True))
        if not parts:
            continue
        idx_b = np.concatenate(parts)
        try:
            v = float(metric_fn(y_true[idx_b], *[a[idx_b] for a in score_args]))
        except Exception:
            continue
        if np.isfinite(v):
            boot_values.append(v)
    if not boot_values:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boot_values, 2.5))
    hi = float(np.percentile(boot_values, 97.5))
    return point, lo, hi


def _per_class_f1_with_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Bootstrap each per-class F1 individually so the CI is class-specific."""
    points = _per_class_f1(y_true, y_pred)
    cis: List[Tuple[float, float]] = []
    for c in range(3):
        def _f1_c(yt, yp, _c=c):
            tp = int(((yt == _c) & (yp == _c)).sum())
            fp = int(((yt != _c) & (yp == _c)).sum())
            fn = int(((yt == _c) & (yp != _c)).sum())
            d = 2 * tp + fp + fn
            return float(2 * tp) / d if d > 0 else 0.0
        _, lo, hi = _stratified_bootstrap(_f1_c, y_true, y_pred, n_boot=n_boot)
        cis.append((lo, hi))
    return points, cis


def _per_class_auc_with_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return [float("nan")] * 3, [(float("nan"), float("nan"))] * 3

    points: List[float] = []
    cis: List[Tuple[float, float]] = []
    for c in range(3):
        scores_c = y_score[:, c]

        def _auc_c(yt, sc, _c=c):
            yb = (yt == _c).astype(np.int64)
            if yb.sum() == 0 or yb.sum() == yb.size:
                raise ValueError("degenerate binary labels")
            return float(roc_auc_score(yb, sc))

        try:
            pt, lo, hi = _stratified_bootstrap(_auc_c, y_true, scores_c, n_boot=n_boot)
        except Exception:
            pt, lo, hi = float("nan"), float("nan"), float("nan")
        points.append(pt)
        cis.append((lo, hi))
    return points, cis


def compute_metrics_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_boot: int,
) -> Dict[str, Any]:
    """All headline metrics + bootstrap CIs for one pooled prediction set."""
    out: Dict[str, Any] = {"n_patients": int(y_true.size)}
    if y_true.size == 0:
        return out

    # Accuracy
    pt, lo, hi = _stratified_bootstrap(_accuracy, y_true, y_pred, n_boot=n_boot)
    out["accuracy"] = pt; out["accuracy_ci95"] = [lo, hi]

    # Balanced accuracy
    pt, lo, hi = _stratified_bootstrap(_balanced_accuracy, y_true, y_pred, n_boot=n_boot)
    out["balanced_accuracy"] = pt; out["balanced_accuracy_ci95"] = [lo, hi]

    # Macro F1 (bootstrap)
    pt, lo, hi = _stratified_bootstrap(_macro_f1, y_true, y_pred, n_boot=n_boot)
    out["macro_f1"] = pt; out["macro_f1_ci95"] = [lo, hi]

    # Per-class F1
    pc_f1, pc_f1_ci = _per_class_f1_with_ci(y_true, y_pred, n_boot=n_boot)
    out["per_class_f1"] = pc_f1
    out["per_class_f1_ci95"] = [list(t) for t in pc_f1_ci]

    # Per-class AUC + Macro AUC
    pc_auc, pc_auc_ci = _per_class_auc_with_ci(y_true, y_score, n_boot=n_boot)
    out["per_class_auc"] = pc_auc
    out["per_class_auc_ci95"] = [list(t) for t in pc_auc_ci]

    def _macro_auc(yt, ys):
        from sklearn.metrics import roc_auc_score
        present = sorted(set(int(c) for c in yt))
        if len(present) < 2:
            raise ValueError("need >=2 classes present")
        return float(roc_auc_score(yt, ys[:, present],
                                   labels=present,
                                   multi_class="ovr", average="macro"))
    try:
        pt, lo, hi = _stratified_bootstrap(_macro_auc, y_true, y_score, n_boot=n_boot)
        out["macro_auc"] = pt
        out["macro_auc_ci95"] = [lo, hi]
    except Exception:
        out["macro_auc"] = float("nan")
        out["macro_auc_ci95"] = [float("nan"), float("nan")]

    out["ci_n_boot"] = int(n_boot)
    return out


def compute_metrics_point(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
) -> Dict[str, Any]:
    """Point estimates only (no bootstrap). Used for per-fold variance."""
    if y_true.size == 0:
        return {"n_patients": 0}
    pc_auc, macro_auc = _per_class_auc_and_macro(y_true, y_score)
    return {
        "n_patients": int(y_true.size),
        "accuracy": _accuracy(y_true, y_pred),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "macro_f1": _macro_f1(y_true, y_pred),
        "per_class_f1": _per_class_f1(y_true, y_pred),
        "macro_auc": macro_auc,
        "per_class_auc": pc_auc,
    }


# =============================================================================
# Per-config processing
# =============================================================================

@dataclass
class ConfigResult:
    cfg: Config
    coverage: Dict[str, Dict[int, bool]]
    union_metrics: Dict[str, Dict[str, Any]]              # cohort -> metrics bundle
    per_fold_metrics: Dict[str, Dict[int, Dict[str, Any]]]  # cohort -> fold -> point dict
    confusion: Dict[str, np.ndarray]                       # cohort -> 3x3 cm
    union_predictions: Dict[str, Dict[str, np.ndarray]]    # cohort -> arrays


def process_config(cfg: Config, n_boot: int) -> ConfigResult:
    """Run the full per-config pipeline: discover folds, pool, compute metrics."""
    files = find_fold_inference_files(cfg)
    coverage: Dict[str, Dict[int, bool]] = {
        cohort: {k: (p is not None) for k, p in folds.items()}
        for cohort, folds in files.items()
    }

    union_metrics: Dict[str, Dict[str, Any]] = {}
    per_fold_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {}
    confusion: Dict[str, np.ndarray] = {}
    union_preds: Dict[str, Dict[str, np.ndarray]] = {}

    for cohort, folds in files.items():
        union_y_t: List[np.ndarray] = []
        union_y_p: List[np.ndarray] = []
        union_y_s: List[np.ndarray] = []
        union_ids: List[str] = []
        per_fold_metrics[cohort] = {}

        for k, p in folds.items():
            if p is None:
                continue
            samples = _extract_patient_samples(p)
            y_t, y_p, y_s, ids = samples_to_arrays(samples)
            if y_t.size == 0:
                continue
            per_fold_metrics[cohort][k] = compute_metrics_point(y_t, y_p, y_s)
            union_y_t.append(y_t)
            union_y_p.append(y_p)
            union_y_s.append(y_s)
            union_ids.extend(ids)

        if union_y_t:
            yT = np.concatenate(union_y_t)
            yP = np.concatenate(union_y_p)
            yS = np.concatenate(union_y_s)
        else:
            yT = np.empty(0, dtype=np.int64)
            yP = np.empty(0, dtype=np.int64)
            yS = np.empty((0, 3), dtype=np.float64)

        union_metrics[cohort] = compute_metrics_bundle(yT, yP, yS, n_boot=n_boot)
        # Confusion matrix on the union.
        cm = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(yT.tolist(), yP.tolist()):
            cm[t, p] += 1
        confusion[cohort] = cm
        union_preds[cohort] = {
            "y_true": yT, "y_pred": yP, "y_score": yS,
            "patient_ids": np.asarray(union_ids),
        }

    return ConfigResult(
        cfg=cfg,
        coverage=coverage,
        union_metrics=union_metrics,
        per_fold_metrics=per_fold_metrics,
        confusion=confusion,
        union_predictions=union_preds,
    )


# =============================================================================
# Per-config disk writers
# =============================================================================

def write_per_config(result: ConfigResult) -> None:
    out_dir = _per_config_dir(result.cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Coverage
    (out_dir / "fold_coverage.json").write_text(
        json.dumps(result.coverage, indent=2) + "\n"
    )

    # Union metrics (JSON-friendly: numpy → list)
    union_clean = {
        cohort: _jsonify(m) for cohort, m in result.union_metrics.items()
    }
    (out_dir / "union_metrics.json").write_text(json.dumps(union_clean, indent=2) + "\n")

    # Per-fold metrics
    pf_clean = {
        cohort: {str(k): _jsonify(v) for k, v in folds.items()}
        for cohort, folds in result.per_fold_metrics.items()
    }
    (out_dir / "per_fold_metrics.json").write_text(json.dumps(pf_clean, indent=2) + "\n")

    # Confusion matrices
    for cohort, cm in result.confusion.items():
        path = out_dir / f"confusion_{cohort}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true \\ pred"] + CLASS_SHORT)
            for i, name in enumerate(CLASS_SHORT):
                w.writerow([name] + [int(x) for x in cm[i]])

    # Union prediction CSV (per cohort)
    for cohort, preds in result.union_predictions.items():
        path = out_dir / f"union_predictions_{cohort}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "true_label", "pred_label",
                        "prob_ADC", "prob_SCLC", "prob_SCC"])
            for i in range(preds["y_true"].size):
                w.writerow([
                    str(preds["patient_ids"][i]),
                    int(preds["y_true"][i]),
                    int(preds["y_pred"][i]),
                    float(preds["y_score"][i, 0]),
                    float(preds["y_score"][i, 1]),
                    float(preds["y_score"][i, 2]),
                ])


def _jsonify(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# =============================================================================
# Table formatting
# =============================================================================

def _fmt_ci(point: Optional[float], lo: Optional[float], hi: Optional[float],
            decimals: int = 3) -> str:
    """``0.607 [0.420, 0.768]`` style. NaNs render as ``--``."""
    if point is None or not np.isfinite(point):
        return "--"
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(point)
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return s
    return f"{s} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _fmt_ci_tex(point: Optional[float], lo: Optional[float], hi: Optional[float],
                decimals: int = 3, bold: bool = False) -> str:
    """LaTeX cell with the bracket part in ``\\scriptsize``."""
    if point is None or not np.isfinite(point):
        return "--"
    fmt = f"{{:.{decimals}f}}"
    pt = fmt.format(point)
    if bold:
        pt = f"$\\mathbf{{{pt}}}$"
    else:
        pt = f"${pt}$"
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return pt
    return f"{pt} {{\\scriptsize $[{fmt.format(lo)}, {fmt.format(hi)}]$}}"


def _bold_mask(values: List[Optional[float]]) -> List[bool]:
    """Mark every cell that equals the column max (within 1e-6)."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return [False] * len(values)
    best = max(finite)
    return [v is not None and np.isfinite(v) and abs(v - best) < 1e-6 for v in values]


# =============================================================================
# Table writers
# =============================================================================

# Section header in LaTeX bodies — distinguishes the two test cohorts.
_LPCD_HEADER = r"\multicolumn{{COLS}}{@{}l}{\textit{Lung-PET-CT-Dx test (DAPT-internal generalisation)}} \\"
_BL_HEADER   = r"\multicolumn{{COLS}}{@{}l}{\textit{BigLunge test (target generalisation)}} \\"


def _table_rows_for_arm(results: List[ConfigResult], arm: str) -> List[ConfigResult]:
    return [r for r in results if r.cfg.arm == arm]


def write_overall_table(results: List[ConfigResult], out_dir: Path) -> None:
    """Headline accuracy / balanced acc / macro F1 / macro AUC — baseline arm."""
    base = _table_rows_for_arm(results, "base")
    cohorts = [("lpcd", _LPCD_HEADER), ("biglunge", _BL_HEADER)]

    # CSV
    csv_path = out_dir / "table_overall.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "pipeline", "backbone", "pretrain", "cohort", "n",
                    "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
                    "balanced_accuracy", "balanced_accuracy_ci_lo", "balanced_accuracy_ci_hi",
                    "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
                    "macro_auc", "macro_auc_ci_lo", "macro_auc_ci_hi"])
        for r in results:
            for cohort, _ in cohorts:
                m = r.union_metrics.get(cohort, {})
                w.writerow([
                    r.cfg.arm, r.cfg.pipeline, r.cfg.backbone_label, r.cfg.pretrain_label,
                    cohort, m.get("n_patients", 0),
                    *_csv_pt_ci(m, "accuracy"),
                    *_csv_pt_ci(m, "balanced_accuracy"),
                    *_csv_pt_ci(m, "macro_f1"),
                    *_csv_pt_ci(m, "macro_auc"),
                ])

    # LaTeX
    lines: List[str] = []
    lines.append(r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.")
    lines.append(r"\begin{tabular}{@{} l l l c c c c c @{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Pipeline} & \textbf{Backbone} & \textbf{Pre-train} & "
                 r"\textbf{n} & \textbf{Accuracy} & \textbf{Balanced Acc.} & "
                 r"\textbf{Macro-F1} & \textbf{Macro AUC} \\")
    lines.append(r"\midrule")

    for cohort, header in cohorts:
        lines.append(header.replace("{COLS}", "8"))
        # Per-column bolding scoped to this cohort.
        acc_vals: List[Optional[float]] = []
        bacc_vals: List[Optional[float]] = []
        f1_vals: List[Optional[float]] = []
        auc_vals: List[Optional[float]] = []
        for r in base:
            m = r.union_metrics.get(cohort, {})
            acc_vals.append(m.get("accuracy"))
            bacc_vals.append(m.get("balanced_accuracy"))
            f1_vals.append(m.get("macro_f1"))
            auc_vals.append(m.get("macro_auc"))
        acc_b, bacc_b, f1_b, auc_b = (_bold_mask(v) for v in
                                      (acc_vals, bacc_vals, f1_vals, auc_vals))
        for i, r in enumerate(base):
            m = r.union_metrics.get(cohort, {})
            n = m.get("n_patients", 0)
            row = " & ".join([
                r.cfg.pipeline_label,
                r.cfg.backbone_label,
                r.cfg.pretrain_label,
                f"${n}$",
                _fmt_ci_tex(m.get("accuracy"), *_ci(m, "accuracy"), bold=acc_b[i]),
                _fmt_ci_tex(m.get("balanced_accuracy"), *_ci(m, "balanced_accuracy"), bold=bacc_b[i]),
                _fmt_ci_tex(m.get("macro_f1"), *_ci(m, "macro_f1"), bold=f1_b[i]),
                _fmt_ci_tex(m.get("macro_auc"), *_ci(m, "macro_auc"), bold=auc_b[i]),
            ])
            lines.append(row + r" \\")
        if cohort == "lpcd":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_overall.tex").write_text("\n".join(lines) + "\n")


def _csv_pt_ci(m: Dict[str, Any], key: str) -> List[Any]:
    v = m.get(key)
    ci = m.get(f"{key}_ci95") or [None, None]
    return [v, ci[0] if ci else None, ci[1] if ci else None]


def _ci(m: Dict[str, Any], key: str) -> Tuple[Optional[float], Optional[float]]:
    ci = m.get(f"{key}_ci95") or [None, None]
    return (ci[0] if ci else None, ci[1] if ci else None)


def write_per_class_f1_table(results: List[ConfigResult], out_dir: Path) -> None:
    base = _table_rows_for_arm(results, "base")
    cohorts = [("lpcd", _LPCD_HEADER), ("biglunge", _BL_HEADER)]

    csv_path = out_dir / "table_per_class_f1.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "pipeline", "backbone", "pretrain", "cohort", "n",
                    "f1_ADC", "f1_ADC_lo", "f1_ADC_hi",
                    "f1_SCLC", "f1_SCLC_lo", "f1_SCLC_hi",
                    "f1_SCC", "f1_SCC_lo", "f1_SCC_hi"])
        for r in results:
            for cohort, _ in cohorts:
                m = r.union_metrics.get(cohort, {})
                f1s = m.get("per_class_f1") or [None, None, None]
                cis = m.get("per_class_f1_ci95") or [[None, None]] * 3
                row = [r.cfg.arm, r.cfg.pipeline, r.cfg.backbone_label,
                       r.cfg.pretrain_label, cohort, m.get("n_patients", 0)]
                for c in range(3):
                    ci = cis[c] if c < len(cis) else [None, None]
                    row += [f1s[c] if c < len(f1s) else None,
                            ci[0] if ci else None, ci[1] if ci else None]
                w.writerow(row)

    lines: List[str] = [
        r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.",
        r"\begin{tabular}{@{} l l l c c c c @{}}",
        r"\toprule",
        r"\textbf{Pipeline} & \textbf{Backbone} & \textbf{Pre-train} & "
        r"\textbf{n} & \textbf{F1 ADC} & \textbf{F1 SCLC} & \textbf{F1 SCC} \\",
        r"\midrule",
    ]
    for cohort, header in cohorts:
        lines.append(header.replace("{COLS}", "7"))
        # Column-bold within cohort.
        cls_vals: List[List[Optional[float]]] = [[] for _ in range(3)]
        for r in base:
            m = r.union_metrics.get(cohort, {})
            f1s = m.get("per_class_f1") or [None, None, None]
            for c in range(3):
                cls_vals[c].append(f1s[c] if c < len(f1s) else None)
        cls_bold = [_bold_mask(v) for v in cls_vals]
        for i, r in enumerate(base):
            m = r.union_metrics.get(cohort, {})
            f1s = m.get("per_class_f1") or [None, None, None]
            cis = m.get("per_class_f1_ci95") or [[None, None]] * 3
            row = " & ".join([
                r.cfg.pipeline_label, r.cfg.backbone_label, r.cfg.pretrain_label,
                f"${m.get('n_patients', 0)}$",
                _fmt_ci_tex(f1s[0], cis[0][0] if cis[0] else None,
                            cis[0][1] if cis[0] else None, bold=cls_bold[0][i]),
                _fmt_ci_tex(f1s[1], cis[1][0] if cis[1] else None,
                            cis[1][1] if cis[1] else None, bold=cls_bold[1][i]),
                _fmt_ci_tex(f1s[2], cis[2][0] if cis[2] else None,
                            cis[2][1] if cis[2] else None, bold=cls_bold[2][i]),
            ])
            lines.append(row + r" \\")
        if cohort == "lpcd":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_per_class_f1.tex").write_text("\n".join(lines) + "\n")


def write_per_class_auc_table(results: List[ConfigResult], out_dir: Path) -> None:
    base = _table_rows_for_arm(results, "base")
    cohorts = [("lpcd", _LPCD_HEADER), ("biglunge", _BL_HEADER)]

    csv_path = out_dir / "table_per_class_auc.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "pipeline", "backbone", "pretrain", "cohort", "n",
                    "auc_ADC", "auc_ADC_lo", "auc_ADC_hi",
                    "auc_SCLC", "auc_SCLC_lo", "auc_SCLC_hi",
                    "auc_SCC", "auc_SCC_lo", "auc_SCC_hi"])
        for r in results:
            for cohort, _ in cohorts:
                m = r.union_metrics.get(cohort, {})
                aucs = m.get("per_class_auc") or [None, None, None]
                cis = m.get("per_class_auc_ci95") or [[None, None]] * 3
                row = [r.cfg.arm, r.cfg.pipeline, r.cfg.backbone_label,
                       r.cfg.pretrain_label, cohort, m.get("n_patients", 0)]
                for c in range(3):
                    ci = cis[c] if c < len(cis) else [None, None]
                    row += [aucs[c] if c < len(aucs) else None,
                            ci[0] if ci else None, ci[1] if ci else None]
                w.writerow(row)

    lines: List[str] = [
        r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.",
        r"\begin{tabular}{@{} l l l c c c c @{}}",
        r"\toprule",
        r"\textbf{Pipeline} & \textbf{Backbone} & \textbf{Pre-train} & "
        r"\textbf{n} & \textbf{AUC ADC} & \textbf{AUC SCLC} & \textbf{AUC SCC} \\",
        r"\midrule",
    ]
    for cohort, header in cohorts:
        lines.append(header.replace("{COLS}", "7"))
        cls_vals: List[List[Optional[float]]] = [[] for _ in range(3)]
        for r in base:
            m = r.union_metrics.get(cohort, {})
            aucs = m.get("per_class_auc") or [None, None, None]
            for c in range(3):
                cls_vals[c].append(aucs[c] if c < len(aucs) else None)
        cls_bold = [_bold_mask(v) for v in cls_vals]
        for i, r in enumerate(base):
            m = r.union_metrics.get(cohort, {})
            aucs = m.get("per_class_auc") or [None, None, None]
            cis = m.get("per_class_auc_ci95") or [[None, None]] * 3
            row = " & ".join([
                r.cfg.pipeline_label, r.cfg.backbone_label, r.cfg.pretrain_label,
                f"${m.get('n_patients', 0)}$",
                _fmt_ci_tex(aucs[0], cis[0][0] if cis[0] else None,
                            cis[0][1] if cis[0] else None, bold=cls_bold[0][i]),
                _fmt_ci_tex(aucs[1], cis[1][0] if cis[1] else None,
                            cis[1][1] if cis[1] else None, bold=cls_bold[1][i]),
                _fmt_ci_tex(aucs[2], cis[2][0] if cis[2] else None,
                            cis[2][1] if cis[2] else None, bold=cls_bold[2][i]),
            ])
            lines.append(row + r" \\")
        if cohort == "lpcd":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_per_class_auc.tex").write_text("\n".join(lines) + "\n")


def write_fpn_ablation_table(results: List[ConfigResult], out_dir: Path) -> None:
    """One row per (pipeline, backbone) showing baseline vs FPN deltas on BL."""
    by_key: Dict[Tuple[str, str, str], Dict[str, ConfigResult]] = {}
    for r in results:
        key = (r.cfg.pipeline, r.cfg.model_type, r.cfg.backbone_label)
        by_key.setdefault(key, {})[r.cfg.arm] = r

    csv_path = out_dir / "table_fpn_ablation.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "backbone",
                    "macro_f1_base", "macro_f1_fpn", "delta_macro_f1",
                    "macro_auc_base", "macro_auc_fpn", "delta_macro_auc",
                    "bal_acc_base", "bal_acc_fpn", "delta_bal_acc"])
        for key, arm_map in by_key.items():
            base_m = (arm_map.get("base") or ConfigResult(_dummy_cfg(), {}, {}, {}, {}, {})).union_metrics.get("biglunge", {})
            fpn_m = (arm_map.get("fpn") or ConfigResult(_dummy_cfg(), {}, {}, {}, {}, {})).union_metrics.get("biglunge", {})
            row = [key[0], key[2]]
            for metric in ["macro_f1", "macro_auc", "balanced_accuracy"]:
                bv = base_m.get(metric); fv = fpn_m.get(metric)
                delta = (fv - bv) if (bv is not None and fv is not None
                                      and np.isfinite(bv) and np.isfinite(fv)) else None
                row += [bv, fv, delta]
            w.writerow(row)

    lines: List[str] = [
        r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.",
        r"\begin{tabular}{@{} l l c c c c c c @{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{Macro-F1}} & \multicolumn{3}{c}{\textbf{Macro AUC}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"\textbf{Pipeline} & \textbf{Backbone} & "
        r"\textbf{Baseline} & \textbf{FPN} & \textbf{$\Delta$} & "
        r"\textbf{Baseline} & \textbf{FPN} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]
    for key, arm_map in by_key.items():
        base_m = arm_map.get("base").union_metrics.get("biglunge", {}) if arm_map.get("base") else {}
        fpn_m  = arm_map.get("fpn").union_metrics.get("biglunge", {}) if arm_map.get("fpn") else {}
        cells = [key[0], key[2]]
        for metric in ["macro_f1", "macro_auc"]:
            bv = base_m.get(metric); fv = fpn_m.get(metric)
            cells.append(_fmt_ci_tex(bv, *_ci(base_m, metric)))
            cells.append(_fmt_ci_tex(fv, *_ci(fpn_m, metric)))
            if bv is not None and fv is not None and np.isfinite(bv) and np.isfinite(fv):
                d = fv - bv
                cells.append(f"${d:+.3f}$")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_fpn_ablation.tex").write_text("\n".join(lines) + "\n")


def _dummy_cfg() -> Config:
    return Config("", "", "", "", "", "")


def write_fpn_paired_table(results: List[ConfigResult], out_dir: Path) -> None:
    """Per-fold ΔMF1 (FPN − baseline) on BigLunge for each backbone.

    Pairs the baseline and FPN ``per_fold_metrics.json`` entries on
    matching fold indices, computes the per-fold delta, and reports
    the mean ± SE across folds present in BOTH arms. This is the
    methodologically defensible comparison when one arm's pooled
    union is over a different fold set than the other's (e.g. when
    the baseline arm is missing a fold that the FPN arm has, or
    vice versa) — see the auxiliary diagnostic adjunct to
    ``table_fpn_ablation``.
    """
    by_key: Dict[Tuple[str, str, str], Dict[str, ConfigResult]] = {}
    for r in results:
        key = (r.cfg.pipeline, r.cfg.model_type, r.cfg.backbone_label)
        by_key.setdefault(key, {})[r.cfg.arm] = r

    csv_path = out_dir / "table_fpn_paired.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "backbone",
                    *[f"fold{k}_delta_macro_f1" for k in range(N_FOLDS)],
                    "mean_delta_macro_f1", "se_delta_macro_f1",
                    *[f"fold{k}_delta_macro_auc" for k in range(N_FOLDS)],
                    "mean_delta_macro_auc", "se_delta_macro_auc",
                    "n_paired_folds"])
        for key, arm_map in by_key.items():
            base = arm_map.get("base")
            fpn  = arm_map.get("fpn")
            if base is None or fpn is None:
                continue
            base_pf = base.per_fold_metrics.get("biglunge", {})
            fpn_pf  = fpn.per_fold_metrics.get("biglunge", {})
            mf1_deltas: List[Optional[float]] = []
            auc_deltas: List[Optional[float]] = []
            for k in range(N_FOLDS):
                bv = base_pf.get(k, {}).get("macro_f1")
                fv = fpn_pf.get(k, {}).get("macro_f1")
                bavu = base_pf.get(k, {}).get("macro_auc")
                favu = fpn_pf.get(k, {}).get("macro_auc")
                d_mf1 = (fv - bv) if (bv is not None and fv is not None
                                      and np.isfinite(bv) and np.isfinite(fv)) else None
                d_auc = (favu - bavu) if (bavu is not None and favu is not None
                                          and np.isfinite(bavu) and np.isfinite(favu)) else None
                mf1_deltas.append(d_mf1)
                auc_deltas.append(d_auc)
            mf1_mean, mf1_se, n_paired = _mean_se(mf1_deltas)
            auc_mean, auc_se, _        = _mean_se(auc_deltas)
            w.writerow([
                key[0], key[2],
                *[(v if v is not None else "") for v in mf1_deltas],
                mf1_mean if mf1_mean is not None else "",
                mf1_se if mf1_se is not None else "",
                *[(v if v is not None else "") for v in auc_deltas],
                auc_mean if auc_mean is not None else "",
                auc_se if auc_se is not None else "",
                n_paired,
            ])

    lines: List[str] = [
        r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.",
        r"\begin{tabular}{@{} l l c c c c c c c @{}}",
        r"\toprule",
        r"\textbf{Pipeline} & \textbf{Backbone} & "
        r"\textbf{F0 $\Delta$} & \textbf{F1 $\Delta$} & "
        r"\textbf{F2 $\Delta$} & \textbf{F3 $\Delta$} & "
        r"\textbf{F4 $\Delta$} & "
        r"\textbf{Mean $\pm$ SE} & \textbf{Paired folds} \\",
        r"\midrule",
    ]
    for key, arm_map in by_key.items():
        base = arm_map.get("base")
        fpn  = arm_map.get("fpn")
        if base is None or fpn is None:
            continue
        base_pf = base.per_fold_metrics.get("biglunge", {})
        fpn_pf  = fpn.per_fold_metrics.get("biglunge", {})
        deltas: List[Optional[float]] = []
        for k in range(N_FOLDS):
            bv = base_pf.get(k, {}).get("macro_f1")
            fv = fpn_pf.get(k, {}).get("macro_f1")
            d = (fv - bv) if (bv is not None and fv is not None
                              and np.isfinite(bv) and np.isfinite(fv)) else None
            deltas.append(d)
        mean, se, n_paired = _mean_se(deltas)
        cells = [
            (f"${d:+.3f}$" if d is not None and np.isfinite(d) else "--")
            for d in deltas
        ]
        if mean is not None and np.isfinite(mean):
            if se is not None and np.isfinite(se):
                mean_cell = f"${mean:+.3f} \\pm {se:.3f}$"
            else:
                mean_cell = f"${mean:+.3f}$"
        else:
            mean_cell = "--"
        lines.append(" & ".join([
            key[0], key[2], *cells, mean_cell, f"${n_paired}/{N_FOLDS}$",
        ]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_fpn_paired.tex").write_text("\n".join(lines) + "\n")


def write_per_fold_variance_table(results: List[ConfigResult], out_dir: Path) -> None:
    """5-fold macro-F1 per config on BigLunge, with mean ± SE across folds."""
    base = _table_rows_for_arm(results, "base")

    csv_path = out_dir / "table_per_fold_variance.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "backbone", "cohort",
                    *[f"fold{k}_macro_f1" for k in range(N_FOLDS)],
                    "mean", "se", "n_folds_present"])
        for r in base:
            for cohort in ("lpcd", "biglunge"):
                pf = r.per_fold_metrics.get(cohort, {})
                vals = [pf.get(k, {}).get("macro_f1") for k in range(N_FOLDS)]
                finite = [v for v in vals if v is not None and np.isfinite(v)]
                mean = float(np.mean(finite)) if finite else None
                se = (float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                      if len(finite) >= 2 else None)
                w.writerow([r.cfg.pipeline, r.cfg.backbone_label, cohort,
                            *[v if v is not None else "" for v in vals],
                            mean if mean is not None else "",
                            se if se is not None else "",
                            len(finite)])

    lines: List[str] = [
        r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.",
        r"\begin{tabular}{@{} l l c c c c c c c @{}}",
        r"\toprule",
        r"\textbf{Pipeline} & \textbf{Backbone} & \textbf{Cohort} & "
        r"\textbf{F0} & \textbf{F1} & \textbf{F2} & \textbf{F3} & \textbf{F4} & "
        r"\textbf{Mean $\pm$ SE} \\",
        r"\midrule",
    ]
    for r in base:
        for cohort in ("lpcd", "biglunge"):
            pf = r.per_fold_metrics.get(cohort, {})
            vals = [pf.get(k, {}).get("macro_f1") for k in range(N_FOLDS)]
            finite = [v for v in vals if v is not None and np.isfinite(v)]
            fold_cells = [(f"${v:.3f}$" if v is not None and np.isfinite(v) else "--") for v in vals]
            if finite:
                mean = float(np.mean(finite))
                se = (float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                      if len(finite) >= 2 else 0.0)
                mean_cell = f"${mean:.3f} \\pm {se:.3f}$"
            else:
                mean_cell = "--"
            cohort_label = "LPCD" if cohort == "lpcd" else "BigLunge"
            lines.append(" & ".join([
                r.cfg.pipeline_label, r.cfg.backbone_label, cohort_label,
                *fold_cells, mean_cell,
            ]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_per_fold_variance.tex").write_text("\n".join(lines) + "\n")


# Literature anchor table — Honda and Dunn on LPCD. Hardcoded reference
# numbers from the SLR; our row comes from the baseline matrix on LPCD.
# Jacob & Menon (2024) was dropped from this list: their 10-fold
# slice-level CV places adjacent slices from one patient in both train
# and test folds, inflating accuracy by ~25--30 pp over patient-level
# evaluation. Their headline 0.990 is not a comparable patient-level
# baseline and is retired from the comparison rather than published
# alongside our numbers (see `Sections/Results.tex` §5.6 / Methodology
# §4.7.1 for the protocol-leakage argument).
LITERATURE_ANCHORS: List[Dict[str, Any]] = [
    {"study": r"Honda et al.\ \cite{Honda2024} (image-only)",
     "cohort": r"LPCD subset $n{=}77$", "split": "4-fold image-level",
     "accuracy": 0.699, "macro_f1": None,
     "note": "ResNet50+SE on bbox crops, 4-class incl LCC, image-level "
             "CV — adjacent slices from one patient can land in both "
             "train and held-out folds"},
    {"study": r"Honda et al.\ \cite{Honda2024} (+clinical)",
     "cohort": r"LPCD subset $n{=}77$", "split": "4-fold image-level",
     "accuracy": 0.759, "macro_f1": None,
     "note": "Same as above with sex+age+smoking; SCLC recall improves "
             "from 0.64 to 0.77 with metadata"},
    {"study": r"Dunn et al.\ \cite{Dunn2023}",
     "cohort": r"LPCD $n{=}324$", "split": "5-fold patient-level",
     "accuracy": 0.927, "macro_f1": None,
     "note": "Radiomic SVM on 3D whole-tumour features inside a "
             "manual ROI bounding box; SMOTE-balanced (without SMOTE "
             "SVM falls to 0.765); AUC 0.97 is ADC-vs-rest only"},
]


def write_literature_anchor_table(results: List[ConfigResult], out_dir: Path) -> None:
    base = _table_rows_for_arm(results, "base")

    csv_path = out_dir / "table_literature_anchors.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "label", "cohort", "split",
                    "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
                    "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
                    "note"])
        # Our baseline rows on LPCD
        for r in base:
            m = r.union_metrics.get("lpcd", {})
            w.writerow(["ours", f"{r.cfg.pipeline_label} {r.cfg.backbone_label}",
                        f"LPCD n={m.get('n_patients', 0)}",
                        "5-fold patient-level",
                        *_csv_pt_ci(m, "accuracy"),
                        *_csv_pt_ci(m, "macro_f1"),
                        "Pooled across 5 folds, stratified bootstrap CI n=1000"])
        for row in LITERATURE_ANCHORS:
            w.writerow(["anchor", row["study"], row["cohort"], row["split"],
                        row["accuracy"], None, None,
                        row["macro_f1"], None, None,
                        row["note"]])

    lines: List[str] = [
        r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.",
        r"\begin{tabular}{@{} l l l c c @{}}",
        r"\toprule",
        r"\textbf{Source} & \textbf{Cohort} & \textbf{Split} & "
        r"\textbf{Accuracy} & \textbf{Macro-F1} \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{This thesis (5-fold patient-level CV, patient-level metrics)}} \\",
    ]
    for r in base:
        m = r.union_metrics.get("lpcd", {})
        lines.append(" & ".join([
            f"{r.cfg.pipeline_label} {r.cfg.backbone_label} ({r.cfg.pretrain_label})",
            f"LPCD $n{{=}}{m.get('n_patients', 0)}$",
            "5-fold (pooled)",
            _fmt_ci_tex(m.get("accuracy"), *_ci(m, "accuracy")),
            _fmt_ci_tex(m.get("macro_f1"), *_ci(m, "macro_f1")),
        ]) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\textit{Literature anchors on Lung-PET-CT-Dx}} \\")
    for row in LITERATURE_ANCHORS:
        acc = f"${row['accuracy']:.3f}$" if row.get("accuracy") is not None else "--"
        f1 = f"${row['macro_f1']:.3f}$" if row.get("macro_f1") is not None else "--"
        lines.append(" & ".join([row["study"], row["cohort"], row["split"], acc, f1]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_literature_anchors.tex").write_text("\n".join(lines) + "\n")


# =============================================================================
# Training-diagnostics summary table
# =============================================================================

def _best_val_row_per_fold(
    rows: List[Dict[str, Any]], phase: str,
) -> List[Dict[str, Any]]:
    """For a phase ('dapt' or 'finetune'), return one summary dict per
    detected fold with the best-by-``val_macro_f1`` row's stats.

    Fold boundaries are detected by epoch reset, matching the existing
    ``_split_metrics_into_folds`` helper used by the training-curves
    figure. The summary dict carries the best ``val_macro_f1``, the
    matching ``val_accuracy``, and the epoch at which that best was
    observed. Folds where ``val_macro_f1`` never appears yield ``None``
    for the metric values but still contribute a fold-index slot, so
    the SE computation downstream can tell ``no fold`` from ``fold
    present but degenerate``.
    """
    runs = _split_metrics_into_folds(rows, phase)
    out: List[Dict[str, Any]] = []
    for run in runs[:N_FOLDS]:
        best_mf1: Optional[float] = None
        best_acc: Optional[float] = None
        best_epoch: Optional[int] = None
        for r in run:
            mf1 = r.get("val_macro_f1")
            if mf1 is None:
                continue
            try:
                mf1f = float(mf1)
            except (TypeError, ValueError):
                continue
            if best_mf1 is None or mf1f > best_mf1:
                best_mf1 = mf1f
                acc = r.get("val_accuracy")
                try:
                    best_acc = float(acc) if acc is not None else None
                except (TypeError, ValueError):
                    best_acc = None
                ep = r.get("epoch")
                try:
                    best_epoch = int(ep) if ep is not None else None
                except (TypeError, ValueError):
                    best_epoch = None
        out.append({
            "best_val_macro_f1": best_mf1,
            "best_val_accuracy": best_acc,
            "epoch_of_best": best_epoch,
            "n_rows": len(run),
        })
    return out


def _mean_se(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    finite = [float(v) for v in values
              if v is not None and np.isfinite(v)]
    if not finite:
        return None, None, 0
    mean = float(np.mean(finite))
    se = (float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
          if len(finite) >= 2 else 0.0)
    return mean, se, len(finite)


def write_training_summary_table(
    results: List[ConfigResult], out_dir: Path,
) -> None:
    """Best-val-macro-F1 / accuracy / epoch-of-best per fold, per phase.

    Reads ``metrics.jsonl`` from each baseline-arm model directory and
    emits a diagnostic adjunct to the headline test table: how high
    val rose during training and how early early-stopping (or the
    epoch budget) cut the run off. The table is *not* a substitute
    for the test-set numbers reported in §5.1 and is presented as
    such in the caption.
    """
    base = _table_rows_for_arm(results, "base")
    phases: List[Tuple[str, str]] = [("dapt", "DAPT"), ("finetune", "Fine-tune")]

    csv_path = out_dir / "table_training_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "backbone", "phase",
                    *[f"fold{k}_best_val_macro_f1" for k in range(N_FOLDS)],
                    "best_val_macro_f1_mean", "best_val_macro_f1_se",
                    *[f"fold{k}_best_val_accuracy" for k in range(N_FOLDS)],
                    "best_val_accuracy_mean", "best_val_accuracy_se",
                    *[f"fold{k}_epoch_of_best" for k in range(N_FOLDS)],
                    "epoch_of_best_mean", "epoch_of_best_se",
                    "n_folds_present"])
        for r in base:
            d = _model_dir(r.cfg)
            rows_jsonl = _load_metrics_jsonl(d / "metrics.jsonl")
            for phase_key, _ in phases:
                summaries = _best_val_row_per_fold(rows_jsonl, phase_key)
                # Pad to N_FOLDS so the CSV columns line up.
                while len(summaries) < N_FOLDS:
                    summaries.append({
                        "best_val_macro_f1": None,
                        "best_val_accuracy": None,
                        "epoch_of_best": None,
                        "n_rows": 0,
                    })
                mf1s = [s["best_val_macro_f1"] for s in summaries]
                accs = [s["best_val_accuracy"] for s in summaries]
                eps  = [s["epoch_of_best"] for s in summaries]
                mf1_mean, mf1_se, n_present = _mean_se(mf1s)
                acc_mean, acc_se, _ = _mean_se(accs)
                ep_mean, ep_se, _   = _mean_se(eps)
                w.writerow([
                    r.cfg.pipeline, r.cfg.backbone_label, phase_key,
                    *[(v if v is not None else "") for v in mf1s],
                    mf1_mean if mf1_mean is not None else "",
                    mf1_se if mf1_se is not None else "",
                    *[(v if v is not None else "") for v in accs],
                    acc_mean if acc_mean is not None else "",
                    acc_se if acc_se is not None else "",
                    *[(v if v is not None else "") for v in eps],
                    ep_mean if ep_mean is not None else "",
                    ep_se if ep_se is not None else "",
                    n_present,
                ])

    lines: List[str] = [
        r"% Auto-generated by scripts/build_final_results.py — do not edit by hand.",
        r"\begin{tabular}{@{} l l l c c c c @{}}",
        r"\toprule",
        r"\textbf{Pipeline} & \textbf{Backbone} & \textbf{Phase} & "
        r"\textbf{Best Val MF1} & \textbf{Best Val Acc.} & "
        r"\textbf{Epoch of Best} & \textbf{Folds} \\",
        r" & & & \textbf{(mean $\pm$ SE)} & \textbf{(mean $\pm$ SE)} & "
        r"\textbf{(mean $\pm$ SE)} & \textbf{(present)} \\",
        r"\midrule",
    ]
    for r in base:
        d = _model_dir(r.cfg)
        rows_jsonl = _load_metrics_jsonl(d / "metrics.jsonl")
        for phase_key, phase_label in phases:
            summaries = _best_val_row_per_fold(rows_jsonl, phase_key)
            mf1s = [s["best_val_macro_f1"] for s in summaries]
            accs = [s["best_val_accuracy"] for s in summaries]
            eps  = [s["epoch_of_best"] for s in summaries]
            mf1_mean, mf1_se, n_present = _mean_se(mf1s)
            acc_mean, acc_se, _ = _mean_se(accs)
            ep_mean, ep_se, _   = _mean_se(eps)

            def _fmt(mean: Optional[float], se: Optional[float],
                     decimals: int = 3) -> str:
                if mean is None or not np.isfinite(mean):
                    return "--"
                fmt = f"{{:.{decimals}f}}"
                if se is None or not np.isfinite(se):
                    return f"${fmt.format(mean)}$"
                return f"${fmt.format(mean)} \\pm {fmt.format(se)}$"

            def _fmt_ep(mean: Optional[float], se: Optional[float]) -> str:
                if mean is None or not np.isfinite(mean):
                    return "--"
                if se is None or not np.isfinite(se):
                    return f"${mean:.1f}$"
                return f"${mean:.1f} \\pm {se:.1f}$"

            lines.append(" & ".join([
                r.cfg.pipeline_label, r.cfg.backbone_label, phase_label,
                _fmt(mf1_mean, mf1_se, decimals=3),
                _fmt(acc_mean, acc_se, decimals=3),
                _fmt_ep(ep_mean, ep_se),
                f"${n_present}/{N_FOLDS}$",
            ]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "table_training_summary.tex").write_text("\n".join(lines) + "\n")


# =============================================================================
# Figure writers
# =============================================================================

def _setup_mpl() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _short_label(cfg: Config) -> str:
    """Short label for x-axis: e.g. ``2D EffNet-B0``."""
    bb = cfg.backbone_label.replace("\\,", " ")
    return f"{cfg.pipeline_label} {bb}"


def fig_overall_macro_f1(results: List[ConfigResult], out_dir: Path) -> None:
    """Side-by-side bars: LPCD-test vs BL-test macro-F1, baseline arm only."""
    import matplotlib.pyplot as plt
    base = _table_rows_for_arm(results, "base")
    if not base:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(base) + 2), 4.6))
    xs = np.arange(len(base))
    width = 0.38
    for k, (cohort, color, label) in enumerate([
            ("lpcd", "#4E79A7", "LPCD-test"),
            ("biglunge", "#E15759", "BigLunge-test")]):
        ys, lo, hi = [], [], []
        for r in base:
            m = r.union_metrics.get(cohort, {})
            v = m.get("macro_f1")
            ci_lo, ci_hi = _ci(m, "macro_f1")
            if v is None or not np.isfinite(v):
                ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(v)
            lo.append(v - ci_lo if (ci_lo is not None and np.isfinite(ci_lo)) else 0)
            hi.append(ci_hi - v if (ci_hi is not None and np.isfinite(ci_hi)) else 0)
        ax.bar(xs + (k - 0.5) * width, ys, width=width, color=color, label=label,
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.set_xticks(xs)
    ax.set_xticklabels([_short_label(r.cfg) for r in base], rotation=25, ha="right")
    ax.set_ylabel("Macro-F1 (95% bootstrap CI)"); ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 across the dimensionality ladder (baseline arm)")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_dir / "fig_overall_macro_f1.pdf"); plt.close(fig)


def fig_per_class_f1(results: List[ConfigResult], out_dir: Path) -> None:
    """Per-class F1 grouped bar chart on BigLunge-test, baseline arm."""
    import matplotlib.pyplot as plt
    base = _table_rows_for_arm(results, "base")
    if not base:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(base) + 2), 4.6))
    width = 0.26
    xs = np.arange(len(base))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ys, lo, hi = [], [], []
        for r in base:
            m = r.union_metrics.get("biglunge", {})
            f1s = m.get("per_class_f1") or []
            cis = m.get("per_class_f1_ci95") or []
            v = f1s[cls_idx] if cls_idx < len(f1s) else None
            ci = cis[cls_idx] if cls_idx < len(cis) else [None, None]
            if v is None or not np.isfinite(v):
                ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(v)
            ci_lo = ci[0] if ci else None
            ci_hi = ci[1] if ci else None
            lo.append(v - ci_lo if (ci_lo is not None and np.isfinite(ci_lo)) else 0)
            hi.append(ci_hi - v if (ci_hi is not None and np.isfinite(ci_hi)) else 0)
        ax.bar(xs + (cls_idx - 1) * width, ys, width=width, label=cls_name,
               color=CLASS_COLORS[cls_name], yerr=[lo, hi], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.set_xticks(xs)
    ax.set_xticklabels([_short_label(r.cfg) for r in base], rotation=25, ha="right")
    ax.set_ylabel("Per-class F1 (95% bootstrap CI)"); ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 — BigLunge test (baseline arm)")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_dir / "fig_per_class_f1.pdf"); plt.close(fig)


def fig_per_class_auc(results: List[ConfigResult], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    base = _table_rows_for_arm(results, "base")
    if not base:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(base) + 2), 4.6))
    width = 0.26
    xs = np.arange(len(base))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ys, lo, hi = [], [], []
        for r in base:
            m = r.union_metrics.get("biglunge", {})
            aucs = m.get("per_class_auc") or []
            cis = m.get("per_class_auc_ci95") or []
            v = aucs[cls_idx] if cls_idx < len(aucs) else None
            ci = cis[cls_idx] if cls_idx < len(cis) else [None, None]
            if v is None or not np.isfinite(v):
                ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(v)
            ci_lo = ci[0] if ci else None
            ci_hi = ci[1] if ci else None
            lo.append(v - ci_lo if (ci_lo is not None and np.isfinite(ci_lo)) else 0)
            hi.append(ci_hi - v if (ci_hi is not None and np.isfinite(ci_hi)) else 0)
        ax.bar(xs + (cls_idx - 1) * width, ys, width=width, label=cls_name,
               color=CLASS_COLORS[cls_name], yerr=[lo, hi], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(base) - 0.5, 0.51, "random", color="#888", fontsize=8, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([_short_label(r.cfg) for r in base], rotation=25, ha="right")
    ax.set_ylabel("Per-class AUC, one-vs-rest (95% CI)"); ax.set_ylim(0, 1)
    ax.set_title("Per-class AUC — BigLunge test (baseline arm)")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out_dir / "fig_per_class_auc.pdf"); plt.close(fig)


def fig_dapt_test_gap(results: List[ConfigResult], out_dir: Path) -> None:
    """LPCD-test → BigLunge-test macro-F1 drop, baseline arm."""
    import matplotlib.pyplot as plt
    base = _table_rows_for_arm(results, "base")
    if not base:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(base) + 2), 4.6))
    xs = np.arange(len(base))
    width = 0.38
    for k, (cohort, color, label) in enumerate([
            ("lpcd", "#4E79A7", "LPCD test"),
            ("biglunge", "#E15759", "BigLunge test")]):
        ys, lo, hi = [], [], []
        for r in base:
            m = r.union_metrics.get(cohort, {})
            v = m.get("macro_f1")
            ci_lo, ci_hi = _ci(m, "macro_f1")
            if v is None or not np.isfinite(v):
                ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(v)
            lo.append(v - ci_lo if (ci_lo is not None and np.isfinite(ci_lo)) else 0)
            hi.append(ci_hi - v if (ci_hi is not None and np.isfinite(ci_hi)) else 0)
        ax.bar(xs + (k - 0.5) * width, ys, width=width, color=color, label=label,
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})
    # Annotate the drop with arrows
    for i, r in enumerate(base):
        lpcd_v = r.union_metrics.get("lpcd", {}).get("macro_f1")
        bl_v = r.union_metrics.get("biglunge", {}).get("macro_f1")
        if (lpcd_v is not None and bl_v is not None
                and np.isfinite(lpcd_v) and np.isfinite(bl_v)):
            drop = bl_v - lpcd_v
            ax.text(i, max(lpcd_v, bl_v) + 0.04, f"$\\Delta={drop:+.2f}$",
                    ha="center", fontsize=8, color="#444")
    ax.set_xticks(xs)
    ax.set_xticklabels([_short_label(r.cfg) for r in base], rotation=25, ha="right")
    ax.set_ylabel("Macro-F1 (95% bootstrap CI)"); ax.set_ylim(0, 1)
    ax.set_title("Cross-cohort generalisation gap: LPCD-test $\\rightarrow$ BigLunge-test")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_dir / "fig_dapt_test_gap.pdf"); plt.close(fig)


def fig_confusion_matrices(results: List[ConfigResult], out_dir: Path) -> None:
    """Grid of patient-level CMs for baseline arm on BigLunge-test."""
    import matplotlib.pyplot as plt
    base = [r for r in results if r.cfg.arm == "base" and r.confusion.get("biglunge") is not None
            and r.confusion["biglunge"].sum() > 0]
    if not base:
        return
    n = len(base)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    for i, r in enumerate(base):
        cm = r.confusion["biglunge"]
        rr, cc = divmod(i, cols)
        ax = axes[rr][cc]
        row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
        norm = np.divide(cm, np.where(row_sum > 0, row_sum, 1.0))
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        for ii in range(3):
            for jj in range(3):
                color = "white" if norm[ii, jj] > 0.5 else "black"
                ax.text(jj, ii, str(int(cm[ii, jj])), ha="center", va="center",
                        fontsize=10, color=color)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_SHORT)
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_SHORT)
        ax.set_xlabel("Predicted")
        if cc == 0:
            ax.set_ylabel("True")
        ax.set_title(f"{_short_label(r.cfg)} (n={int(cm.sum())})", fontsize=10)
    for j in range(n, rows * cols):
        rr, cc = divmod(j, cols)
        axes[rr][cc].axis("off")
    fig.suptitle("Patient-level confusion matrices — BigLunge test", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_confusion_matrices.pdf"); plt.close(fig)


def _split_metrics_into_folds(rows: List[Dict[str, Any]], phase: str) -> List[List[Dict[str, Any]]]:
    """Detect run boundaries by epoch reset (epoch <= prev → new run)."""
    phase_rows = [r for r in rows if r.get("phase") == phase]
    runs: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_epoch = 0
    for r in phase_rows:
        epoch = int(r.get("epoch") or 0)
        if epoch <= last_epoch and current:
            runs.append(current); current = []
        current.append(r); last_epoch = epoch
    if current:
        runs.append(current)
    return runs


def fig_training_curves(results: List[ConfigResult], out_dir: Path) -> None:
    """Two-panel figure: DAPT (left) + FT (right). One subplot per config,
    each plotting val_macro_f1 over epochs with 5 folds overlaid."""
    import matplotlib.pyplot as plt
    base = _table_rows_for_arm(results, "base")
    rows = len(base)
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.3 * rows), squeeze=False)
    for ri, r in enumerate(base):
        d = _model_dir(r.cfg)
        rows_jsonl = _load_metrics_jsonl(d / "metrics.jsonl")
        for ci, (phase, title) in enumerate([("dapt", "DAPT (Lung-PET-CT-Dx)"),
                                              ("finetune", "Fine-tune (BigLunge)")]):
            ax = axes[ri][ci]
            runs = _split_metrics_into_folds(rows_jsonl, phase)
            for fk, run in enumerate(runs[:N_FOLDS]):
                ep = [int(rr.get("epoch") or 0) for rr in run]
                vf = [rr.get("val_macro_f1_rolling") or rr.get("val_macro_f1") for rr in run]
                if not ep or not vf:
                    continue
                xs = [e for e, v in zip(ep, vf) if v is not None]
                ys = [v for v in vf if v is not None]
                ax.plot(xs, ys, linewidth=1.1, alpha=0.65, label=f"fold {fk}")
            ax.set_title(f"{_short_label(r.cfg)} — {title}", fontsize=9)
            ax.set_xlabel("Epoch")
            if ci == 0:
                ax.set_ylabel("val macro-F1")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, linestyle=":")
            if ri == 0 and ci == 1:
                ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_training_curves.pdf"); plt.close(fig)


def fig_fpn_delta(results: List[ConfigResult], out_dir: Path) -> None:
    """Per-config baseline-vs-FPN bar comparison on BL macro-F1 and macro-AUC."""
    import matplotlib.pyplot as plt
    by_key: Dict[Tuple[str, str, str], Dict[str, ConfigResult]] = {}
    for r in results:
        key = (r.cfg.pipeline, r.cfg.model_type, r.cfg.backbone_label)
        by_key.setdefault(key, {})[r.cfg.arm] = r
    keys = list(by_key.keys())
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.4 * len(keys) + 4), 4.6))
    width = 0.38
    xs = np.arange(len(keys))
    for ai, metric in enumerate(["macro_f1", "macro_auc"]):
        ax = axes[ai]
        for k, (arm, color, label) in enumerate([("base", "#4E79A7", "Baseline"),
                                                  ("fpn",  "#E15759", "FPN")]):
            ys, lo, hi = [], [], []
            for key in keys:
                r = by_key[key].get(arm)
                if r is None:
                    ys.append(np.nan); lo.append(0); hi.append(0); continue
                m = r.union_metrics.get("biglunge", {})
                v = m.get(metric); ci_lo, ci_hi = _ci(m, metric)
                if v is None or not np.isfinite(v):
                    ys.append(np.nan); lo.append(0); hi.append(0); continue
                ys.append(v)
                lo.append(v - ci_lo if (ci_lo is not None and np.isfinite(ci_lo)) else 0)
                hi.append(ci_hi - v if (ci_hi is not None and np.isfinite(ci_hi)) else 0)
            ax.bar(xs + (k - 0.5) * width, ys, width=width, color=color, label=label,
                   yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{k[0].upper()} {k[2].replace('\\\\,', ' ')}" for k in keys],
            rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_title(f"BigLunge {metric} — baseline vs FPN")
        if ai == 0:
            ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_fpn_delta.pdf"); plt.close(fig)


# =============================================================================
# Coverage report + provenance
# =============================================================================

def write_coverage_report(results: List[ConfigResult], path: Path) -> None:
    lines: List[str] = [
        f"# Coverage report — {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| arm | pipeline | model | cohort | folds present | n total |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for cohort in ("lpcd", "biglunge"):
            folds = r.coverage.get(cohort, {})
            present = [str(k) for k in range(N_FOLDS) if folds.get(k)]
            n = r.union_metrics.get(cohort, {}).get("n_patients", 0)
            mark = ", ".join(present) if present else "(none)"
            lines.append(f"| {r.cfg.arm} | {r.cfg.pipeline} | {r.cfg.model_type} | {cohort} | {mark} | {n} |")
    path.write_text("\n".join(lines) + "\n")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_readme(results: List[ConfigResult], n_boot: int, out_dir: Path) -> None:
    sha = _git_sha()
    when = datetime.now().isoformat(timespec="seconds")
    total = len(results)
    complete = sum(1 for r in results
                   if all(r.coverage["lpcd"].get(k) for k in range(N_FOLDS))
                   and all(r.coverage["biglunge"].get(k) for k in range(N_FOLDS)))
    lines: List[str] = [
        "# Final thesis results — auto-generated",
        "",
        f"- Generated: `{when}`",
        f"- Repo commit: `{sha}`",
        f"- Bootstrap replicates: `n_boot={n_boot}`",
        f"- Configs fully covered (5/5 folds in both cohorts): `{complete}/{total}`",
        "",
        "## What's in here",
        "",
        "- `per_config/<arm>/<pipeline>/<model>/` — per-config artefacts: pooled",
        "  predictions, per-fold metrics, confusion matrices.",
        "- `tables/` — LaTeX-ready table fragments (`\\input{...}` from `Results.tex`)",
        "  alongside CSV companions.",
        "- `figures/` — PDF figures cited by `Results.tex`.",
        "- `coverage_report.md` — which (arm, model, cohort, fold) is present.",
        "",
        "## Aggregation",
        "",
        "Per-fold patient predictions are pooled into a single union-of-folds",
        "vector per (config, cohort), and metrics are computed once on that union",
        "with stratified non-parametric bootstrap CIs (resampling within each",
        "true-label class). This matches the methodology in Section 4.7 of the",
        "thesis: \"the union of the five disjoint per-fold test partitions covers",
        "the full cohort exactly once\".",
        "",
        "Per-fold variance (mean ± SE across the five fold point estimates) is",
        "reported separately in `tables/table_per_fold_variance.tex`.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "python scripts/build_final_results.py",
        "```",
        "",
        "Re-run after the gap-fill scripts finish to refresh the tree in place.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["base", "fpn"],
                    choices=["base", "fpn"],
                    help="Which arms to process. Default: both.")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap replicates per metric. Default: 1000.")
    ap.add_argument("--skip-figures", action="store_true",
                    help="Skip PDF figure generation (tables only).")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Output tree root. Default: results/thesis_final/")
    args = ap.parse_args()

    global RESULTS_ROOT
    if args.out_root is not None:
        RESULTS_ROOT = args.out_root.resolve()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "tables").mkdir(exist_ok=True)
    (RESULTS_ROOT / "figures").mkdir(exist_ok=True)
    (RESULTS_ROOT / "per_config").mkdir(exist_ok=True)

    matrix = [c for c in _matrix() if c.arm in args.arms]
    print(f"[final] processing {len(matrix)} configs across arms {args.arms} "
          f"(n_boot={args.n_boot})")

    results: List[ConfigResult] = []
    for cfg in matrix:
        print(f"[final] {cfg.arm}/{cfg.pipeline}/{cfg.model_type}")
        r = process_config(cfg, n_boot=args.n_boot)
        write_per_config(r)
        results.append(r)

    # Tables
    tables_dir = RESULTS_ROOT / "tables"
    write_overall_table(results, tables_dir)
    write_per_class_f1_table(results, tables_dir)
    write_per_class_auc_table(results, tables_dir)
    write_fpn_ablation_table(results, tables_dir)
    write_per_fold_variance_table(results, tables_dir)
    write_literature_anchor_table(results, tables_dir)
    write_training_summary_table(results, tables_dir)
    write_fpn_paired_table(results, tables_dir)

    # Figures
    if not args.skip_figures:
        _setup_mpl()
        figs_dir = RESULTS_ROOT / "figures"
        fig_overall_macro_f1(results, figs_dir)
        fig_per_class_f1(results, figs_dir)
        fig_per_class_auc(results, figs_dir)
        fig_dapt_test_gap(results, figs_dir)
        fig_confusion_matrices(results, figs_dir)
        fig_training_curves(results, figs_dir)
        fig_fpn_delta(results, figs_dir)

    # Coverage report + README
    write_coverage_report(results, RESULTS_ROOT / "coverage_report.md")
    write_readme(results, args.n_boot, RESULTS_ROOT)

    print(f"[final] done. wrote tree to {RESULTS_ROOT}")
    print(f"[final] check: {RESULTS_ROOT / 'coverage_report.md'}")


if __name__ == "__main__":
    main()
