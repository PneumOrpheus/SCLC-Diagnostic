"""Build the thesis-results tree from the raw training artifacts.

Reads:
- ``output/<pipeline>/<model>/metrics.jsonl``               (per-epoch + test rows)
- ``output/<pipeline>/<model>/<model>_*_inference_probabilities.json``
- ``/home/data/trained_models/<pipeline>/<model>/*_pbest_{raw,roll}.pth``
- ``~/.cache/monai_*_2d/.../dropped_patients.json``         (for the matched-set audit)

Writes a clean, deterministic ``thesis_results/`` tree. Idempotent: re-run
any time, output overwritten in place.

Layout (per pipeline, identical shape across 2D / MIL / 3D):

    thesis_results/<pipeline>/
    ├── per_model/<model>/
    │   ├── dapt_curve.csv            per-epoch DAPT training metrics
    │   ├── finetune_curve.csv        per-epoch FT training metrics
    │   ├── test_results.json         DAPT-test + BigLunge-test rows + 95% CIs
    │   ├── confusion_volume.csv      3x3 reconstructed from inference_probs
    │   ├── confusion_patient.csv     3x3 patient-level
    │   ├── inference_probs_test.json     verbatim copy of latest BigLunge-test JSON
    │   ├── inference_probs_dapt_test.json verbatim copy of latest DAPT-test JSON
    │   └── _provenance.json          which run, which checkpoints, dates
    ├── tables/
    │   ├── headline.csv              one row per (model, dataset)
    │   ├── headline.md               LaTeX-ready
    │   ├── training_summary.csv      one row per (model, phase): peak F1, epochs
    │   └── per_class_f1.csv          long-format: model, dataset, class, F1, CI
    └── figures/
        ├── fig_training_curves.pdf   DAPT + FT panels, one line per model
        ├── fig_per_class_f1.pdf      bar chart on BigLunge-test
        ├── fig_dapt_test_gap.pdf     DAPT-val peak / DAPT-test / BL-test
        └── fig_confusion_matrices.pdf grid, BL-test patient-level

Plus ``thesis_results/README.md`` documenting the canonical-run choice and
the reproduction command.

"Canonical run" identification: for each (model, phase) pair we walk the
metrics.jsonl rows for that phase in order, splitting them into runs
whenever the recorded ``epoch`` field decreases or stays equal (signature
of a new training run starting at epoch 1). The CANONICAL run is the LAST
run in chronological order. This is deterministic and matches "the most
recent training attempt for this model" — exactly what the thesis should
report.

Usage::

    python scripts/build_thesis_results.py                  # all pipelines
    python scripts/build_thesis_results.py --pipeline 2d
    python scripts/build_thesis_results.py --pipeline 2d --model efficientnet_b0_2d
    python scripts/build_thesis_results.py --skip-figures   # tables only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make repo importable when invoked directly.
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# =============================================================================
# Configuration
# =============================================================================

PIPELINES: Dict[str, List[str]] = {
    "2d":  [
        "efficientnet_b0_2d",
        "resnet50_2d", "resnet50_2d_rin",
        "densenet121_2d", "densenet121_2d_rin",
        "swin_tiny_2d",
    ],
    "mil": ["mil_resnet50", "mil_swin_tiny"],
    "3d":  ["swin_unetr"],
}

# Pretty labels for figures and the markdown headline table.
MODEL_LABEL: Dict[str, str] = {
    "efficientnet_b0_2d":  "EffNet-B0 (2D, ImageNet)",
    "densenet121_2d":      "DenseNet121 (2D, ImageNet)",
    "densenet121_2d_rin":  "DenseNet121 (2D, RadImageNet)",
    "resnet50_2d":         "ResNet-50 (2D, ImageNet)",
    "resnet50_2d_rin":     "ResNet-50 (2D, RadImageNet)",
    "swin_tiny_2d":        "Swin-Tiny (2D, RadImageNet)",
    "mil_resnet50":        "MIL ResNet-50",
    "mil_swin_tiny":       "MIL Swin-Tiny (RadImageNet)",
    "swin_unetr":          "SwinUNETR (3D)",
}

# Tableau-10 colorblind-friendly palette.
CLASS_NAMES: List[str] = ["Adenocarcinoma", "Small Cell", "Squamous"]
CLASS_COLORS: Dict[str, str] = {
    "Adenocarcinoma": "#4E79A7",
    "Small Cell":     "#F28E2B",
    "Squamous":       "#59A14F",
}
MODEL_COLORS: Dict[str, str] = {
    "efficientnet_b0_2d":  "#4E79A7",
    "resnet50_2d":         "#F28E2B",
    "resnet50_2d_rin":     "#FFB04A",
    "densenet121_2d":      "#59A14F",
    "densenet121_2d_rin":  "#86C77B",
    "swin_tiny_2d":        "#E15759",
    "mil_resnet50":        "#76B7B2",
    "mil_swin_tiny":       "#9C755F",
    "swin_unetr":          "#B07AA1",
}

REPO_ROOT = Path(_REPO)
OUTPUT_ROOT = REPO_ROOT / "results" / "output"
RESULTS_ROOT = REPO_ROOT / "results" / "thesis"
CHECKPOINT_ROOT = Path("/home/data/trained_models")


# =============================================================================
# Loading + canonical-run identification
# =============================================================================

def _metrics_path(pipeline: str, model_type: str) -> Path:
    return OUTPUT_ROOT / pipeline / model_type / "metrics.jsonl"


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[load] skipping malformed row in {path}: {exc}", file=sys.stderr)
    return rows


def split_into_runs(rows: List[Dict[str, Any]], phase: str) -> List[List[Dict[str, Any]]]:
    """Split phase-filtered rows into monotonic-epoch runs.

    A new run starts whenever the recorded ``epoch`` decreases or stays
    equal (epoch reset to 1 on the next attempt). Walking append-order rows
    forward, a new training-run boundary is detected this way.
    """
    phase_rows = [r for r in rows if r.get("phase") == phase]
    runs: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_epoch = 0
    for r in phase_rows:
        epoch = int(r.get("epoch") or 0)
        if epoch <= last_epoch and current:
            runs.append(current)
            current = []
        current.append(r)
        last_epoch = epoch
    if current:
        runs.append(current)
    return runs


def latest_test_row(rows: List[Dict[str, Any]], phase: str) -> Optional[Dict[str, Any]]:
    """Most recent test-row with the given phase tag (e.g. 'dapt_test', 'test')."""
    matching = [r for r in rows if r.get("phase") == phase]
    if not matching:
        return None
    matching.sort(key=lambda r: r.get("timestamp", ""))
    return matching[-1]


# =============================================================================
# Per-model writers
# =============================================================================

CURVE_FIELDS = [
    "epoch", "train_loss", "train_macro_f1",
    "val_macro_f1_raw", "val_macro_f1_rolling",
    "val_loss", "val_accuracy", "val_balanced_accuracy",
    "lr_backbone", "lr_head",
    "mixup_alpha", "mixup_active",
    "epochs_no_improve", "monitor_window", "monitor_level",
    "timestamp",
]


def write_curve_csv(run_rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Per-epoch curve. Columns are stable across pipelines; missing fields are blank."""
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in run_rows:
            row_out = {k: None for k in CURVE_FIELDS}
            row_out["epoch"] = r.get("epoch")
            row_out["train_loss"] = r.get("train_loss")
            row_out["train_macro_f1"] = r.get("train_macro_f1")
            # The metrics row stores the single-epoch peak as ``val_macro_f1``;
            # we rename to ``val_macro_f1_raw`` for clarity in the canonical CSV.
            row_out["val_macro_f1_raw"] = r.get("val_macro_f1")
            row_out["val_macro_f1_rolling"] = r.get("val_macro_f1_rolling")
            row_out["val_loss"] = r.get("val_loss")
            row_out["val_accuracy"] = r.get("val_accuracy")
            row_out["val_balanced_accuracy"] = r.get("val_balanced_accuracy")
            row_out["lr_backbone"] = r.get("lr_backbone")
            row_out["lr_head"] = r.get("lr_head")
            row_out["mixup_alpha"] = r.get("mixup_alpha")
            row_out["mixup_active"] = r.get("mixup_active")
            row_out["epochs_no_improve"] = r.get("epochs_no_improve")
            row_out["monitor_window"] = r.get("monitor_window")
            row_out["monitor_level"] = r.get("monitor_level")
            row_out["timestamp"] = r.get("timestamp")
            w.writerow(row_out)


def write_test_results(
    rows: List[Dict[str, Any]],
    out_path: Path,
    dapt_probs: Optional[Path] = None,
    bl_probs:   Optional[Path] = None,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Pull the latest dapt_test + test rows; augment with accuracy + AUC CIs
    computed from the corresponding inference-probabilities JSON. Returns
    the payload for downstream use.
    """
    payload = {
        "dapt_test":     latest_test_row(rows, "dapt_test"),
        "biglunge_test": latest_test_row(rows, "test"),
    }
    # Merge accuracy + AUC into each test row's ``test_patient`` block. If
    # a row has no test_patient block (legacy entries), create one.
    for key, probs_path in [("dapt_test", dapt_probs), ("biglunge_test", bl_probs)]:
        row = payload.get(key)
        if row is None:
            continue
        extra = compute_extra_metrics(probs_path, n_boot=n_boot)
        if not extra:
            continue
        block = row.get("test_patient")
        if not isinstance(block, dict):
            block = {}
            row["test_patient"] = block
        # Don't overwrite existing CIs already populated by the validator
        # (e.g. macro_f1_ci95) — only fill in the new fields.
        for k, v in extra.items():
            block.setdefault(k, v)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def find_latest_inference_probs(pipeline: str, model_type: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (dapt_test_file, biglunge_test_file) — most recent of each, by mtime."""
    out_dir = OUTPUT_ROOT / pipeline / model_type
    if not out_dir.is_dir():
        return None, None
    dapt = sorted(
        out_dir.glob(f"{model_type}_*_dapt_inference_probabilities.json"),
        key=lambda p: p.stat().st_mtime,
    )
    bl = [
        p for p in sorted(
            out_dir.glob(f"{model_type}_*_inference_probabilities.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if "_dapt_" not in p.name
    ]
    return (dapt[-1] if dapt else None, bl[-1] if bl else None)


def copy_inference_probs(model_dir: Path, dapt_src: Optional[Path], bl_src: Optional[Path]) -> None:
    if bl_src is not None:
        shutil.copy(bl_src, model_dir / "inference_probs_test.json")
    if dapt_src is not None:
        shutil.copy(dapt_src, model_dir / "inference_probs_dapt_test.json")


def build_confusion_from_probs(probs_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read an inference_probabilities JSON. Return (volume_cm, patient_cm) 3x3.

    ``volume_cm`` is built from the top-level ``samples`` (one entry per
    volume in the 2D pipeline; one entry per bag in MIL). ``patient_cm`` is
    built from ``patient_level.samples`` if present, else falls back to the
    volume CM (1 patient = 1 volume).
    """
    payload = json.loads(probs_path.read_text())

    def _cm(samples: List[Dict[str, Any]]) -> np.ndarray:
        cm = np.zeros((3, 3), dtype=np.int64)
        for s in samples:
            t = int(s.get("true_label", -1))
            p = int(s.get("pred_label", -1))
            if 0 <= t < 3 and 0 <= p < 3:
                cm[t, p] += 1
        return cm

    vol_samples = payload.get("samples") or []
    pat_samples = (payload.get("patient_level") or {}).get("samples") or []
    vol_cm = _cm(vol_samples)
    pat_cm = _cm(pat_samples) if pat_samples else vol_cm.copy()
    return vol_cm, pat_cm


def _samples_to_arrays(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert inference-probs samples into (y_true, y_pred, y_score) np arrays.

    ``y_score`` is shape (N, 3): per-sample softmax probabilities in the
    canonical class order (Adenocarcinoma=0, Small Cell=1, Squamous=2). If
    the per-sample ``probabilities`` dict is missing or malformed, the row
    is dropped.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[List[float]] = []
    for s in samples:
        t = s.get("true_label")
        p = s.get("pred_label")
        probs = s.get("probabilities") or {}
        if t is None or p is None:
            continue
        try:
            ti = int(t); pi = int(p)
        except (TypeError, ValueError):
            continue
        if not (0 <= ti < 3 and 0 <= pi < 3):
            continue
        # ``probabilities`` is keyed by class display names — fall back to
        # zeros if any class is missing so we don't silently lose samples.
        try:
            row = [
                float(probs.get(CLASS_NAMES[0], 0.0)),
                float(probs.get(CLASS_NAMES[1], 0.0)),
                float(probs.get(CLASS_NAMES[2], 0.0)),
            ]
        except (TypeError, ValueError):
            row = [0.0, 0.0, 0.0]
        y_true.append(ti)
        y_pred.append(pi)
        y_score.append(row)
    return (np.asarray(y_true, dtype=np.int64),
            np.asarray(y_pred, dtype=np.int64),
            np.asarray(y_score, dtype=np.float64))


def _bootstrap_metric(
    metric_fn,
    y_true: np.ndarray,
    *args,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Stratified bootstrap CI for any callable ``metric_fn(y_true, *args, idx=)``.

    The resampling indices are stratified within each true-label class so
    rare classes are guaranteed to be present in every replicate (without
    this, e.g. SC with n=12 sometimes drops out and the metric explodes).
    """
    rng = np.random.default_rng(rng_seed)
    classes = np.unique(y_true)
    idx_by_cls = {int(c): np.where(y_true == c)[0] for c in classes}
    point = float(metric_fn(y_true, *args))
    samples = np.empty(n_boot, dtype=np.float64)
    valid = 0
    for _ in range(n_boot):
        idx_parts = [
            rng.choice(idx_by_cls[c], size=len(idx_by_cls[c]), replace=True)
            for c in classes
            if len(idx_by_cls[c]) > 0
        ]
        idx = np.concatenate(idx_parts)
        boot_args = tuple(a[idx] if isinstance(a, np.ndarray) else a for a in args)
        try:
            v = float(metric_fn(y_true[idx], *boot_args))
            if np.isfinite(v):
                samples[valid] = v
                valid += 1
        except Exception:  # noqa: BLE001 — skip degenerate replicates
            continue
    samples = samples[:valid]
    if samples.size == 0:
        return point, point, point
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def compute_extra_metrics(
    probs_path: Optional[Path], n_boot: int = 1000,
) -> Dict[str, Any]:
    """Read an inference-probs JSON and compute (accuracy, macro-AUC, per-class AUC)
    with 95% stratified bootstrap CIs, on the **patient-level** samples.

    Returns a dict with the new fields ready to merge into the existing
    ``test_patient`` block. Skips silently and returns ``{}`` if the JSON
    isn't available or has no patient-level samples.
    """
    if not probs_path or not probs_path.is_file():
        return {}
    try:
        payload = json.loads(probs_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    pat_samples = (payload.get("patient_level") or {}).get("samples") or []
    if not pat_samples:
        # MIL stores samples flat (1 bag = 1 patient), not under patient_level.
        pat_samples = payload.get("samples") or []
    if not pat_samples:
        return {}
    y_true, y_pred, y_score = _samples_to_arrays(pat_samples)
    if y_true.size == 0:
        return {}

    out: Dict[str, Any] = {"num_patients_extra": int(y_true.size)}

    # --- Accuracy ---
    def _acc(yt, yp):
        return float((yt == yp).sum()) / max(1, len(yt))
    pt, lo, hi = _bootstrap_metric(_acc, y_true, y_pred, n_boot=n_boot, rng_seed=0)
    out["accuracy"] = pt
    out["accuracy_ci95"] = [lo, hi]

    # --- Macro AUC (one-vs-rest) ---
    # sklearn.roc_auc_score requires every present class to have at least
    # one positive AND one negative in the resample. Stratified bootstrap
    # by true label preserves positives; with 3 classes and every class
    # represented, both conditions hold.
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return out

    def _macro_auc(yt, ys):
        # ys is the score matrix indexed alongside yt.
        if ys.ndim != 2 or ys.shape[1] < 2:
            raise ValueError("score matrix expected (N, num_classes)")
        # Restrict to classes actually present in this resample.
        present = sorted(set(int(c) for c in yt))
        if len(present) < 2:
            raise ValueError("need ≥2 classes present")
        ys_sub = ys[:, present]
        return float(roc_auc_score(yt, ys_sub, labels=present,
                                   multi_class="ovr", average="macro"))

    try:
        pt, lo, hi = _bootstrap_metric(_macro_auc, y_true, y_score,
                                       n_boot=n_boot, rng_seed=0)
        out["macro_auc"] = pt
        out["macro_auc_ci95"] = [lo, hi]
    except Exception as exc:  # noqa: BLE001
        print(f"[auc] macro AUC failed: {exc}")

    # --- Per-class AUC (one-vs-rest), with bootstrap CI per class ---
    per_class_auc: List[float] = []
    per_class_auc_ci: List[List[float]] = []
    for c in range(3):
        y_bin = (y_true == c).astype(np.int64)
        scores_c = y_score[:, c]
        # Both positives and negatives need to exist in every replicate;
        # stratified-on-true_label bootstrap ensures positives. Negatives
        # are everything else, also preserved by stratified resampling.
        try:
            def _auc_c(yt, sc, _c=c):
                # Re-compute the binary indicator inside the bootstrap so
                # the resampled labels match the resampled scores exactly.
                yb = (yt == _c).astype(np.int64)
                if yb.sum() == 0 or yb.sum() == len(yb):
                    raise ValueError("degenerate binary labels")
                return float(roc_auc_score(yb, sc))
            pt, lo, hi = _bootstrap_metric(_auc_c, y_true, scores_c,
                                           n_boot=n_boot, rng_seed=0)
            per_class_auc.append(pt)
            per_class_auc_ci.append([lo, hi])
        except Exception as exc:  # noqa: BLE001
            print(f"[auc] per-class {c} ({CLASS_NAMES[c]}) failed: {exc}")
            per_class_auc.append(float("nan"))
            per_class_auc_ci.append([float("nan"), float("nan")])
    out["per_class_auc"] = per_class_auc
    out["per_class_auc_ci95"] = per_class_auc_ci
    out["ci_n_boot"] = int(n_boot)
    return out


def write_confusion_csv(cm: np.ndarray, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + CLASS_NAMES)
        for i in range(3):
            w.writerow([CLASS_NAMES[i]] + cm[i].tolist())


def find_latest_pbests(pipeline: str, model_type: str) -> Dict[str, Optional[str]]:
    """Latest dapt + finetune pbest checkpoints (raw + roll variants), as paths."""
    ckpt_dir = CHECKPOINT_ROOT / pipeline / model_type
    out: Dict[str, Optional[str]] = {
        "dapt_pbest_raw":     None,
        "dapt_pbest_roll":    None,
        "finetune_pbest_raw": None,
        "finetune_pbest_roll": None,
    }
    if not ckpt_dir.is_dir():
        return out
    for phase in ("dapt", "finetune"):
        for variant in ("raw", "roll"):
            files = sorted(
                ckpt_dir.glob(f"*_{phase}_pbest_{variant}.pth"),
                key=lambda p: p.stat().st_mtime,
            )
            if files:
                out[f"{phase}_pbest_{variant}"] = str(files[-1])
    return out


def _peak_in_run(run: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[int]]:
    """Return (peak_value, peak_epoch) for the given key across the run."""
    if not run:
        return None, None
    best = max(run, key=lambda r: float(r.get(key) or -1e9))
    val = best.get(key)
    return (float(val) if val is not None else None,
            int(best.get("epoch") or 0))


def write_provenance(
    out_path: Path, *,
    model_type: str, pipeline: str,
    metrics_jsonl: Path,
    canonical_dapt: List[Dict[str, Any]],
    canonical_ft:   List[Dict[str, Any]],
    test_payload: Dict[str, Any],
    pbests: Dict[str, Optional[str]],
    inference_sources: Tuple[Optional[Path], Optional[Path]],
) -> None:
    def _summarize_run(run: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not run:
            return {"n_epochs": 0}
        peak_raw, peak_raw_ep = _peak_in_run(run, "val_macro_f1")
        peak_roll, peak_roll_ep = _peak_in_run(run, "val_macro_f1_rolling")
        return {
            "n_epochs": len(run),
            "first_timestamp": run[0].get("timestamp"),
            "last_timestamp":  run[-1].get("timestamp"),
            "first_epoch": run[0].get("epoch"),
            "last_epoch":  run[-1].get("epoch"),
            "peak_val_macro_f1_raw":      peak_raw,
            "peak_val_macro_f1_raw_epoch": peak_raw_ep,
            "peak_val_macro_f1_rolling":  peak_roll,
            "peak_val_macro_f1_rolling_epoch": peak_roll_ep,
        }

    payload = {
        "model_type": model_type,
        "pipeline":   pipeline,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics_jsonl_source": str(metrics_jsonl.relative_to(REPO_ROOT))
            if metrics_jsonl.exists() else None,
        "canonical_dapt_run":     _summarize_run(canonical_dapt),
        "canonical_finetune_run": _summarize_run(canonical_ft),
        "dapt_test_timestamp":     (test_payload.get("dapt_test") or {}).get("timestamp"),
        "biglunge_test_timestamp": (test_payload.get("biglunge_test") or {}).get("timestamp"),
        "inference_probs_sources": {
            "dapt_test":     str(inference_sources[0]) if inference_sources[0] else None,
            "biglunge_test": str(inference_sources[1]) if inference_sources[1] else None,
        },
        "checkpoints": pbests,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))


# =============================================================================
# Cross-model tables
# =============================================================================

def _ci_lo(ci: Any) -> Optional[float]:
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return float(ci[0])
    return None


def _ci_hi(ci: Any) -> Optional[float]:
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return float(ci[1])
    return None


def _ci_str(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> str:
    if value is None:
        return ""
    if lo is None or hi is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def _per_class_f1(test_row: Dict[str, Any], idx: int) -> Optional[float]:
    p = test_row.get("test_patient") or {}
    arr = p.get("per_class_f1") or []
    return float(arr[idx]) if 0 <= idx < len(arr) else None


def _per_class_ci(test_row: Dict[str, Any], idx: int, lo_or_hi: int) -> Optional[float]:
    p = test_row.get("test_patient") or {}
    arr = p.get("per_class_f1_ci95") or []
    if not (0 <= idx < len(arr)):
        return None
    pair = arr[idx]
    return float(pair[lo_or_hi]) if isinstance(pair, (list, tuple)) and len(pair) == 2 else None


HEADLINE_FIELDS = [
    "model_type", "model_label", "dataset",
    "n_patients",
    "accuracy",      "accuracy_ci_lo",      "accuracy_ci_hi",
    "balanced_acc",  "balanced_acc_ci_lo",  "balanced_acc_ci_hi",
    "macro_f1",      "macro_f1_ci_lo",      "macro_f1_ci_hi",
    "macro_auc",     "macro_auc_ci_lo",     "macro_auc_ci_hi",
    "f1_adeno",      "f1_adeno_ci_lo",      "f1_adeno_ci_hi",
    "f1_smallcell",  "f1_smallcell_ci_lo",  "f1_smallcell_ci_hi",
    "f1_squamous",   "f1_squamous_ci_lo",   "f1_squamous_ci_hi",
    "auc_adeno",     "auc_adeno_ci_lo",     "auc_adeno_ci_hi",
    "auc_smallcell", "auc_smallcell_ci_lo", "auc_smallcell_ci_hi",
    "auc_squamous",  "auc_squamous_ci_lo",  "auc_squamous_ci_hi",
    "test_timestamp",
]

DATASETS = [
    ("dapt_test",     "Lung-PET-CT-Dx (test)"),
    ("biglunge_test", "BigLunge (test)"),
]


def _per_class_auc(test_row: Dict[str, Any], idx: int) -> Optional[float]:
    p = test_row.get("test_patient") or {}
    arr = p.get("per_class_auc") or []
    return float(arr[idx]) if 0 <= idx < len(arr) else None


def _per_class_auc_ci(test_row: Dict[str, Any], idx: int, lo_or_hi: int) -> Optional[float]:
    p = test_row.get("test_patient") or {}
    arr = p.get("per_class_auc_ci95") or []
    if not (0 <= idx < len(arr)):
        return None
    pair = arr[idx]
    return float(pair[lo_or_hi]) if isinstance(pair, (list, tuple)) and len(pair) == 2 else None


def build_headline_rows(model_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model_type, data in model_data.items():
        test_payload = data["test_payload"]
        for key, dataset_label in DATASETS:
            row = test_payload.get(key)
            if row is None:
                continue
            p = row.get("test_patient") or {}
            out.append({
                "model_type": model_type,
                "model_label": MODEL_LABEL.get(model_type, model_type),
                "dataset": dataset_label,
                "n_patients": p.get("num_patients"),
                "accuracy":            p.get("accuracy"),
                "accuracy_ci_lo":      _ci_lo(p.get("accuracy_ci95")),
                "accuracy_ci_hi":      _ci_hi(p.get("accuracy_ci95")),
                "balanced_acc":        p.get("balanced_accuracy"),
                "balanced_acc_ci_lo":  _ci_lo(p.get("balanced_accuracy_ci95")),
                "balanced_acc_ci_hi":  _ci_hi(p.get("balanced_accuracy_ci95")),
                "macro_f1":            p.get("macro_f1"),
                "macro_f1_ci_lo":      _ci_lo(p.get("macro_f1_ci95")),
                "macro_f1_ci_hi":      _ci_hi(p.get("macro_f1_ci95")),
                "macro_auc":           p.get("macro_auc"),
                "macro_auc_ci_lo":     _ci_lo(p.get("macro_auc_ci95")),
                "macro_auc_ci_hi":     _ci_hi(p.get("macro_auc_ci95")),
                "f1_adeno":            _per_class_f1(row, 0),
                "f1_adeno_ci_lo":      _per_class_ci(row, 0, 0),
                "f1_adeno_ci_hi":      _per_class_ci(row, 0, 1),
                "f1_smallcell":        _per_class_f1(row, 1),
                "f1_smallcell_ci_lo":  _per_class_ci(row, 1, 0),
                "f1_smallcell_ci_hi":  _per_class_ci(row, 1, 1),
                "f1_squamous":         _per_class_f1(row, 2),
                "f1_squamous_ci_lo":   _per_class_ci(row, 2, 0),
                "f1_squamous_ci_hi":   _per_class_ci(row, 2, 1),
                "auc_adeno":           _per_class_auc(row, 0),
                "auc_adeno_ci_lo":     _per_class_auc_ci(row, 0, 0),
                "auc_adeno_ci_hi":     _per_class_auc_ci(row, 0, 1),
                "auc_smallcell":       _per_class_auc(row, 1),
                "auc_smallcell_ci_lo": _per_class_auc_ci(row, 1, 0),
                "auc_smallcell_ci_hi": _per_class_auc_ci(row, 1, 1),
                "auc_squamous":        _per_class_auc(row, 2),
                "auc_squamous_ci_lo":  _per_class_auc_ci(row, 2, 0),
                "auc_squamous_ci_hi":  _per_class_auc_ci(row, 2, 1),
                "test_timestamp":      row.get("timestamp"),
            })
    return out


def write_headline_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADLINE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_headline_md(rows: List[Dict[str, Any]], out_path: Path, pipeline: str) -> None:
    lines: List[str] = []
    lines.append(f"# {pipeline.upper()} pipeline — headline test results\n")
    lines.append("Patient-level metrics on each held-out test split. CIs are stratified bootstrap "
                 "(n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.\n")
    lines.append("")
    # --- Overall metrics table ---
    lines.append("## Overall metrics\n")
    lines.append("| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model_label']} | {r['dataset']} | {r['n_patients']} | "
            f"{_ci_str(r['accuracy'], r['accuracy_ci_lo'], r['accuracy_ci_hi'])} | "
            f"{_ci_str(r['balanced_acc'], r['balanced_acc_ci_lo'], r['balanced_acc_ci_hi'])} | "
            f"{_ci_str(r['macro_f1'], r['macro_f1_ci_lo'], r['macro_f1_ci_hi'])} | "
            f"{_ci_str(r['macro_auc'], r['macro_auc_ci_lo'], r['macro_auc_ci_hi'])} |"
        )
    lines.append("")
    # --- Per-class F1 table ---
    lines.append("## Per-class F1\n")
    lines.append("| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model_label']} | {r['dataset']} | "
            f"{_ci_str(r['f1_adeno'], r['f1_adeno_ci_lo'], r['f1_adeno_ci_hi'])} | "
            f"{_ci_str(r['f1_smallcell'], r['f1_smallcell_ci_lo'], r['f1_smallcell_ci_hi'])} | "
            f"{_ci_str(r['f1_squamous'], r['f1_squamous_ci_lo'], r['f1_squamous_ci_hi'])} |"
        )
    lines.append("")
    # --- Per-class AUC table ---
    lines.append("## Per-class AUC (one-vs-rest)\n")
    lines.append("| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model_label']} | {r['dataset']} | "
            f"{_ci_str(r['auc_adeno'], r['auc_adeno_ci_lo'], r['auc_adeno_ci_hi'])} | "
            f"{_ci_str(r['auc_smallcell'], r['auc_smallcell_ci_lo'], r['auc_smallcell_ci_hi'])} | "
            f"{_ci_str(r['auc_squamous'], r['auc_squamous_ci_lo'], r['auc_squamous_ci_hi'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


TRAINING_SUMMARY_FIELDS = [
    "model_type", "phase",
    "n_epochs", "first_epoch", "last_epoch",
    "peak_val_macro_f1_raw", "peak_val_macro_f1_raw_epoch",
    "peak_val_macro_f1_rolling", "peak_val_macro_f1_rolling_epoch",
    "first_timestamp", "last_timestamp",
]


def write_training_summary(model_data: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for model_type, data in model_data.items():
        for phase_label, run_key in [("dapt", "canonical_dapt"), ("finetune", "canonical_ft")]:
            run = data.get(run_key) or []
            if not run:
                continue
            peak_raw, peak_raw_ep = _peak_in_run(run, "val_macro_f1")
            peak_roll, peak_roll_ep = _peak_in_run(run, "val_macro_f1_rolling")
            rows.append({
                "model_type": model_type,
                "phase":      phase_label,
                "n_epochs":   len(run),
                "first_epoch": run[0].get("epoch"),
                "last_epoch":  run[-1].get("epoch"),
                "peak_val_macro_f1_raw":          peak_raw,
                "peak_val_macro_f1_raw_epoch":    peak_raw_ep,
                "peak_val_macro_f1_rolling":      peak_roll,
                "peak_val_macro_f1_rolling_epoch": peak_roll_ep,
                "first_timestamp": run[0].get("timestamp"),
                "last_timestamp":  run[-1].get("timestamp"),
            })
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAINING_SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


PER_CLASS_FIELDS = [
    "model_type", "model_label", "dataset", "class",
    "f1",  "f1_ci_lo",  "f1_ci_hi",
    "auc", "auc_ci_lo", "auc_ci_hi",
]


def write_per_class_table(headline_rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Long-format per-class table — F1 + AUC. Easy for plotting / pivot."""
    rows: List[Dict[str, Any]] = []
    for hr in headline_rows:
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            slug = ["adeno", "smallcell", "squamous"][cls_idx]
            rows.append({
                "model_type":  hr["model_type"],
                "model_label": hr["model_label"],
                "dataset":     hr["dataset"],
                "class":       cls_name,
                "f1":          hr.get(f"f1_{slug}"),
                "f1_ci_lo":    hr.get(f"f1_{slug}_ci_lo"),
                "f1_ci_hi":    hr.get(f"f1_{slug}_ci_hi"),
                "auc":         hr.get(f"auc_{slug}"),
                "auc_ci_lo":   hr.get(f"auc_{slug}_ci_lo"),
                "auc_ci_hi":   hr.get(f"auc_{slug}_ci_hi"),
            })
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_CLASS_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =============================================================================
# Figures (read from the canonical tables we just wrote)
# =============================================================================

def _setup_mpl_style() -> None:
    import matplotlib.pyplot as plt  # local import — figure step is optional
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
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _read_curve(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fig_training_curves(pipeline: str, models: List[str]) -> Optional[Path]:
    """Two panels: DAPT (left), FT (right). Raw thin + rolling-3 thick per model."""
    import matplotlib.pyplot as plt
    pmd = RESULTS_ROOT / pipeline / "per_model"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    ax_d, ax_f = axes
    plotted = False
    for m in models:
        model_dir = pmd / m
        for ax, fname, panel in [(ax_d, "dapt_curve.csv", "dapt"),
                                  (ax_f, "finetune_curve.csv", "finetune")]:
            curve = _read_curve(model_dir / fname)
            if not curve:
                continue
            xs = [int(r["epoch"]) for r in curve if r.get("epoch")]
            ys_raw = [float(r["val_macro_f1_raw"])     for r in curve if r.get("val_macro_f1_raw") not in (None, "")]
            ys_roll = [float(r["val_macro_f1_rolling"]) for r in curve if r.get("val_macro_f1_rolling") not in (None, "")]
            color = MODEL_COLORS.get(m, "#444")
            label = MODEL_LABEL.get(m, m)
            if xs and ys_raw:
                ax.plot(xs[:len(ys_raw)], ys_raw, color=color, alpha=0.35, linewidth=1.0)
            if xs and ys_roll:
                ax.plot(xs[:len(ys_roll)], ys_roll, color=color, linewidth=1.7, label=label)
                plotted = True
    if not plotted:
        plt.close(fig)
        print(f"[fig_training_curves:{pipeline}] no data, skipping")
        return None
    # Annotate LP-FT freeze boundary on the FT panel (matches default
    # finetune_freeze_backbone_epochs=5).
    ax_f.axvline(5.5, color="#999", linestyle=":", linewidth=1)
    yl = ax_f.get_ylim()
    ax_f.text(5.5, yl[0] + 0.02 * (yl[1] - yl[0]), "  LP→FT", color="#666",
              fontsize=8, va="bottom", ha="left")
    ax_d.set_xlabel("Epoch"); ax_d.set_ylabel("Patient-level MacroF1 (val)")
    ax_d.set_title("DAPT on Lung-PET-CT-Dx"); ax_d.grid(alpha=0.3, linestyle=":")
    ax_f.set_xlabel("Epoch"); ax_f.set_title("Fine-tune on BigLunge")
    ax_f.grid(alpha=0.3, linestyle=":")
    ax_d.legend(loc="lower right", frameon=False)
    fig.suptitle("Validation MacroF1 (raw thin / rolling-3 thick)", fontsize=11, y=1.02)
    out = RESULTS_ROOT / pipeline / "figures" / "fig_training_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_per_class_f1(pipeline: str, models: List[str]) -> Optional[Path]:
    """Bar chart on BigLunge test: 3 bars per model (one per class) with CIs."""
    import matplotlib.pyplot as plt
    pmd = RESULTS_ROOT / pipeline / "per_model"
    data: Dict[str, Tuple[List[float], List[Tuple[float, float]]]] = {}
    for m in models:
        tp = pmd / m / "test_results.json"
        if not tp.is_file():
            continue
        payload = json.loads(tp.read_text())
        bl = payload.get("biglunge_test") or {}
        p = bl.get("test_patient") or {}
        f1s = p.get("per_class_f1") or []
        cis = p.get("per_class_f1_ci95") or []
        if len(f1s) < 3:
            continue
        data[m] = ([float(x) for x in f1s[:3]],
                   [(float(cis[i][0]), float(cis[i][1])) if i < len(cis) else (float(f1s[i]), float(f1s[i])) for i in range(3)])
    if not data:
        print(f"[fig_per_class_f1:{pipeline}] no data, skipping")
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(data) + 2), 4.5))
    width = 0.26
    xs = np.arange(len(data))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ys, lo, hi = [], [], []
        for m in data:
            f1s, cis = data[m]
            ys.append(f1s[cls_idx])
            lo.append(f1s[cls_idx] - cis[cls_idx][0])
            hi.append(cis[cls_idx][1] - f1s[cls_idx])
        ax.bar(xs + (cls_idx - 1) * width, ys, width=width, label=cls_name,
               color=CLASS_COLORS[cls_name], yerr=[lo, hi], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in data], rotation=25, ha="right")
    ax.set_ylabel("Per-class F1 (95% CI)"); ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 — BigLunge test")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    out = RESULTS_ROOT / pipeline / "figures" / "fig_per_class_f1.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig_dapt_test_gap(pipeline: str, models: List[str]) -> Optional[Path]:
    """Three bars per model: DAPT-val (peak roll-3) / DAPT-test / BL-test."""
    import matplotlib.pyplot as plt
    pmd = RESULTS_ROOT / pipeline / "per_model"
    series_data: Dict[str, Dict[str, Tuple[Optional[float], Optional[Tuple[float, float]]]]] = {}
    for m in models:
        mdir = pmd / m
        # DAPT-val peak (rolling) — read from the curve and find the max.
        curve = _read_curve(mdir / "dapt_curve.csv")
        dapt_val_peak = None
        if curve:
            roll_vals = [float(r["val_macro_f1_rolling"]) for r in curve if r.get("val_macro_f1_rolling") not in (None, "")]
            dapt_val_peak = max(roll_vals) if roll_vals else None
        # DAPT-test + BL-test from test_results.json.
        tp = mdir / "test_results.json"
        if not tp.is_file():
            continue
        tr = json.loads(tp.read_text())
        def _get(key: str) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
            row = tr.get(key) or {}
            p = row.get("test_patient") or {}
            v = p.get("macro_f1")
            ci = p.get("macro_f1_ci95") or [None, None]
            if v is None:
                return None, None
            ci_t = (float(ci[0]), float(ci[1])) if all(c is not None for c in ci) else None
            return float(v), ci_t
        dapt_test = _get("dapt_test")
        bl_test = _get("biglunge_test")
        series_data[m] = {
            "DAPT val (peak roll-3)": (dapt_val_peak, None),
            "DAPT test":              dapt_test,
            "BigLunge test":          bl_test,
        }
    if not series_data:
        print(f"[fig_dapt_test_gap:{pipeline}] no data, skipping")
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(series_data) + 2), 4.5))
    width = 0.26
    xs = np.arange(len(series_data))
    series = ["DAPT val (peak roll-3)", "DAPT test", "BigLunge test"]
    colors = ["#9DB7C8", "#4E79A7", "#E15759"]
    for k, (label, color) in enumerate(zip(series, colors)):
        ys, lo, hi = [], [], []
        for m in series_data:
            v, ci = series_data[m][label]
            if v is None:
                ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(v)
            if ci is None:
                lo.append(0); hi.append(0)
            else:
                lo.append(v - ci[0]); hi.append(ci[1] - v)
        ax.bar(xs + (k - 1) * width, ys, width=width, label=label, color=color,
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in series_data], rotation=25, ha="right")
    ax.set_ylabel("Patient-level MacroF1 (95% CI)"); ax.set_ylim(0, 1)
    ax.set_title("Generalization gap: DAPT-val → DAPT-test → BigLunge-test")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    out = RESULTS_ROOT / pipeline / "figures" / "fig_dapt_test_gap.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def _read_test_payload(pipeline: str, model: str) -> Dict[str, Any]:
    p = RESULTS_ROOT / pipeline / "per_model" / model / "test_results.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def fig_per_class_auc(pipeline: str, models: List[str]) -> Optional[Path]:
    """Per-class AUC (one-vs-rest) bar chart on BigLunge test. Mirrors
    ``fig_per_class_f1`` but for AUC instead of F1.
    """
    import matplotlib.pyplot as plt
    data: Dict[str, Tuple[List[float], List[Tuple[float, float]]]] = {}
    for m in models:
        payload = _read_test_payload(pipeline, m)
        bl = payload.get("biglunge_test") or {}
        p = bl.get("test_patient") or {}
        aucs = p.get("per_class_auc") or []
        cis = p.get("per_class_auc_ci95") or []
        if len(aucs) < 3:
            continue
        data[m] = ([float(x) for x in aucs[:3]],
                   [(float(cis[i][0]), float(cis[i][1])) if i < len(cis)
                    else (float(aucs[i]), float(aucs[i]))
                    for i in range(3)])
    if not data:
        print(f"[fig_per_class_auc:{pipeline}] no BigLunge-test AUC data, skipping")
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(data) + 2), 4.5))
    width = 0.26
    xs = np.arange(len(data))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ys, lo, hi = [], [], []
        for m in data:
            aucs, cis = data[m]
            ys.append(aucs[cls_idx])
            lo.append(aucs[cls_idx] - cis[cls_idx][0])
            hi.append(cis[cls_idx][1] - aucs[cls_idx])
        ax.bar(xs + (cls_idx - 1) * width, ys, width=width, label=cls_name,
               color=CLASS_COLORS[cls_name], yerr=[lo, hi], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})
    # Random-classifier reference line at AUC=0.5.
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(data) - 0.5, 0.51, "random", color="#888", fontsize=8, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in data], rotation=25, ha="right")
    ax.set_ylabel("Per-class AUC, one-vs-rest (95% CI)"); ax.set_ylim(0, 1)
    ax.set_title("Per-class AUC — BigLunge test")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", frameon=False)
    out = RESULTS_ROOT / pipeline / "figures" / "fig_per_class_auc.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig_accuracy_gap(pipeline: str, models: List[str]) -> Optional[Path]:
    """Accuracy across DAPT-val (peak rolling), DAPT-test, BigLunge-test.
    Mirrors ``fig_dapt_test_gap`` but for plain accuracy. CIs from bootstrap
    on the test sets; DAPT-val is the peak rolling value (no CI tracked).
    """
    import matplotlib.pyplot as plt
    series_data: Dict[str, Dict[str, Tuple[Optional[float], Optional[Tuple[float, float]]]]] = {}
    for m in models:
        mdir = RESULTS_ROOT / pipeline / "per_model" / m
        # DAPT-val peak (rolling) accuracy from the dapt_curve.csv.
        dv = None
        curve = _read_curve(mdir / "dapt_curve.csv")
        if curve:
            # Use balanced_accuracy / accuracy from val rows (val_balanced_accuracy
            # in the row maps to the rolling-mean monitor; we want raw best).
            accs = [float(r["val_accuracy"]) for r in curve if r.get("val_accuracy") not in (None, "")]
            if accs:
                dv = max(accs)
        payload = _read_test_payload(pipeline, m)
        def _get(key: str) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
            row = payload.get(key) or {}
            p = row.get("test_patient") or {}
            v = p.get("accuracy")
            ci = p.get("accuracy_ci95") or [None, None]
            if v is None:
                return None, None
            ci_t = (float(ci[0]), float(ci[1])) if all(c is not None for c in ci) else None
            return float(v), ci_t
        if curve is None and not payload:
            continue
        series_data[m] = {
            "DAPT val (peak)": (dv, None),
            "DAPT test":       _get("dapt_test"),
            "BigLunge test":   _get("biglunge_test"),
        }
    series_data = {m: d for m, d in series_data.items() if any(v[0] is not None for v in d.values())}
    if not series_data:
        print(f"[fig_accuracy_gap:{pipeline}] no accuracy data, skipping")
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(series_data) + 2), 4.5))
    width = 0.26
    xs = np.arange(len(series_data))
    series = ["DAPT val (peak)", "DAPT test", "BigLunge test"]
    colors = ["#9DB7C8", "#4E79A7", "#E15759"]
    for k, (label, color) in enumerate(zip(series, colors)):
        ys, lo, hi = [], [], []
        for m in series_data:
            v, ci = series_data[m][label]
            if v is None:
                ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(v)
            if ci is None:
                lo.append(0); hi.append(0)
            else:
                lo.append(v - ci[0]); hi.append(ci[1] - v)
        ax.bar(xs + (k - 1) * width, ys, width=width, label=label, color=color,
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in series_data], rotation=25, ha="right")
    ax.set_ylabel("Patient-level accuracy (95% CI)"); ax.set_ylim(0, 1)
    ax.set_title("Accuracy: DAPT-val → DAPT-test → BigLunge-test")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    out = RESULTS_ROOT / pipeline / "figures" / "fig_accuracy_gap.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig_macro_auc_gap(pipeline: str, models: List[str]) -> Optional[Path]:
    """Macro AUC on DAPT-test vs BigLunge-test. (No DAPT-val series — AUC
    is not tracked per epoch.)
    """
    import matplotlib.pyplot as plt
    series_data: Dict[str, Dict[str, Tuple[Optional[float], Optional[Tuple[float, float]]]]] = {}
    for m in models:
        payload = _read_test_payload(pipeline, m)
        if not payload:
            continue
        def _get(key: str) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
            row = payload.get(key) or {}
            p = row.get("test_patient") or {}
            v = p.get("macro_auc")
            ci = p.get("macro_auc_ci95") or [None, None]
            if v is None:
                return None, None
            ci_t = (float(ci[0]), float(ci[1])) if all(c is not None for c in ci) else None
            return float(v), ci_t
        series_data[m] = {
            "DAPT test":     _get("dapt_test"),
            "BigLunge test": _get("biglunge_test"),
        }
    series_data = {m: d for m, d in series_data.items() if any(v[0] is not None for v in d.values())}
    if not series_data:
        print(f"[fig_macro_auc_gap:{pipeline}] no AUC data, skipping")
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(series_data) + 2), 4.5))
    width = 0.35
    xs = np.arange(len(series_data))
    series = ["DAPT test", "BigLunge test"]
    colors = ["#4E79A7", "#E15759"]
    for k, (label, color) in enumerate(zip(series, colors)):
        ys, lo, hi = [], [], []
        for m in series_data:
            v, ci = series_data[m][label]
            if v is None:
                ys.append(np.nan); lo.append(0); hi.append(0); continue
            ys.append(v)
            if ci is None:
                lo.append(0); hi.append(0)
            else:
                lo.append(v - ci[0]); hi.append(ci[1] - v)
        ax.bar(xs + (k - 0.5) * width, ys, width=width, label=label, color=color,
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(series_data) - 0.5, 0.51, "random", color="#888", fontsize=8, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in series_data], rotation=25, ha="right")
    ax.set_ylabel("Macro AUC (one-vs-rest, 95% CI)"); ax.set_ylim(0, 1)
    ax.set_title("Macro AUC: DAPT-test vs BigLunge-test")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    out = RESULTS_ROOT / pipeline / "figures" / "fig_macro_auc_gap.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig_roc_curves(pipeline: str, models: List[str]) -> Optional[Path]:
    """3 panels (one per class). Per-panel: ROC curve overlay, one line per
    model, computed from BigLunge-test inference probabilities. Each line
    annotated with its bootstrap-AUC point estimate.
    """
    import matplotlib.pyplot as plt
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        print("[fig_roc_curves] scikit-learn not available, skipping")
        return None
    pmd = RESULTS_ROOT / pipeline / "per_model"
    # Gather (y_true, y_score, auc_per_class) for each model with BL-test data.
    models_with_data: Dict[str, Tuple[np.ndarray, np.ndarray, List[Optional[float]]]] = {}
    for m in models:
        probs_path = pmd / m / "inference_probs_test.json"
        if not probs_path.is_file():
            continue
        try:
            payload = json.loads(probs_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pat_samples = (payload.get("patient_level") or {}).get("samples") or []
        if not pat_samples:
            pat_samples = payload.get("samples") or []
        if not pat_samples:
            continue
        y_true, _, y_score = _samples_to_arrays(pat_samples)
        if y_true.size == 0:
            continue
        # Per-class AUC point estimates from the test_results.json (already CI-bootstrapped).
        tp = _read_test_payload(pipeline, m).get("biglunge_test") or {}
        per_cls_auc = (tp.get("test_patient") or {}).get("per_class_auc") or [None, None, None]
        models_with_data[m] = (y_true, y_score, per_cls_auc)
    if not models_with_data:
        print(f"[fig_roc_curves:{pipeline}] no inference probs for BigLunge-test, skipping")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[cls_idx]
        ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, alpha=0.6)
        for m, (y_true, y_score, per_cls_auc) in models_with_data.items():
            y_bin = (y_true == cls_idx).astype(np.int64)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            scores = y_score[:, cls_idx]
            try:
                fpr, tpr, _ = roc_curve(y_bin, scores)
            except ValueError:
                continue
            auc = per_cls_auc[cls_idx] if cls_idx < len(per_cls_auc) else None
            label = MODEL_LABEL.get(m, m)
            if auc is not None:
                label = f"{label} (AUC={auc:.3f})"
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(m, "#444"), linewidth=1.7, label=label)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate")
        if cls_idx == 0:
            ax.set_ylabel("True positive rate")
        ax.set_title(f"{cls_name} (vs rest)")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Per-class ROC — BigLunge test (patient-level)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = RESULTS_ROOT / pipeline / "figures" / "fig_roc_curves.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig_confusion_matrices(pipeline: str, models: List[str]) -> Optional[Path]:
    """Grid of 3x3 patient-level CMs from inference_probs_test.json."""
    import matplotlib.pyplot as plt
    pmd = RESULTS_ROOT / pipeline / "per_model"
    cms: Dict[str, Tuple[np.ndarray, int]] = {}
    for m in models:
        # Use the patient-level CM CSV we already wrote.
        cpath = pmd / m / "confusion_patient.csv"
        if not cpath.is_file():
            continue
        cm = np.zeros((3, 3), dtype=np.int64)
        with cpath.open() as f:
            r = csv.reader(f)
            next(r)  # header
            for i, row in enumerate(r):
                cm[i] = [int(x) for x in row[1:1 + 3]]
        if cm.sum() > 0:
            cms[m] = (cm, int(cm.sum()))
    if not cms:
        print(f"[fig_confusion_matrices:{pipeline}] no data, skipping")
        return None
    n = len(cms)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)
    for i, (m, (cm, n_p)) in enumerate(cms.items()):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
        norm = np.divide(cm, np.where(row_sum > 0, row_sum, 1.0))
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        for ii in range(3):
            for jj in range(3):
                color = "white" if norm[ii, jj] > 0.5 else "black"
                ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center",
                        fontsize=10, color=color)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Adeno", "SC", "Sq"])
        ax.set_yticks(range(3)); ax.set_yticklabels(["Adeno", "SC", "Sq"])
        ax.set_xlabel("Predicted")
        if c == 0:
            ax.set_ylabel("True")
        ax.set_title(f"{MODEL_LABEL.get(m, m)} (n={n_p})", fontsize=10)
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    fig.suptitle("Patient-level confusion matrices — BigLunge test", fontsize=12, y=1.02)
    fig.tight_layout()
    out = RESULTS_ROOT / pipeline / "figures" / "fig_confusion_matrices.pdf"
    fig.savefig(out); plt.close(fig)
    return out


# =============================================================================
# README
# =============================================================================

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def write_readme(out_path: Path, model_data_per_pipeline: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    sha = _git_sha()
    now = datetime.now().isoformat(timespec="seconds")

    lines: List[str] = []
    lines.append("# Thesis results")
    lines.append("")
    lines.append(f"Generated: `{now}`  ·  Git: `{sha}`")
    lines.append("")
    lines.append("All numbers reported in the thesis are derived from the files in this tree. "
                 "Run `python scripts/build_thesis_results.py` to regenerate from "
                 "`output/<pipeline>/<model>/metrics.jsonl` plus the inference-probability "
                 "JSONs in the same directories.")
    lines.append("")
    lines.append("## Layout")
    lines.append("")
    lines.append("```")
    lines.append("thesis_results/<pipeline>/")
    lines.append("├── per_model/<model>/         per-run training curves + test rows + CMs")
    lines.append("├── tables/                    cross-model headline + per-class + summary")
    lines.append("└── figures/                   PDF figures, sourced from tables/")
    lines.append("```")
    lines.append("")
    lines.append("## Canonical-run identification")
    lines.append("")
    lines.append("`metrics.jsonl` is append-only across attempts. For each (model, phase) "
                 "we identify the **canonical run** as the LAST monotonic-epoch block in "
                 "chronological order — i.e., the most recent training attempt. Earlier "
                 "rows from failed attempts are kept in the source files for audit but are "
                 "not reflected here.")
    lines.append("")
    lines.append("## Per-model canonical runs (this generation)")
    lines.append("")
    for pipeline in sorted(model_data_per_pipeline):
        pipeline_data = model_data_per_pipeline[pipeline]
        if not pipeline_data:
            continue
        lines.append(f"### {pipeline.upper()} pipeline")
        lines.append("")
        lines.append("| Model | DAPT epochs | DAPT range | FT epochs | FT range |")
        lines.append("|---|---|---|---|---|")
        for model_type, data in pipeline_data.items():
            d = data.get("canonical_dapt") or []
            f = data.get("canonical_ft") or []
            d_ep = len(d)
            f_ep = len(f)
            d_range = (
                f"{d[0].get('timestamp', '?')} → {d[-1].get('timestamp', '?')}"
                if d else "—"
            )
            f_range = (
                f"{f[0].get('timestamp', '?')} → {f[-1].get('timestamp', '?')}"
                if f else "—"
            )
            lines.append(
                f"| {MODEL_LABEL.get(model_type, model_type)} | {d_ep} | {d_range} | "
                f"{f_ep} | {f_range} |"
            )
        lines.append("")
    lines.append("## Headline tables")
    lines.append("")
    for pipeline in sorted(model_data_per_pipeline):
        if not model_data_per_pipeline[pipeline]:
            continue
        md = RESULTS_ROOT / pipeline / "tables" / "headline.md"
        if md.is_file():
            lines.append(md.read_text())
    out_path.write_text("\n".join(lines).rstrip() + "\n")


# =============================================================================
# Top-level orchestration
# =============================================================================

def build_pipeline(pipeline: str, only_model: Optional[str], skip_figures: bool) -> Dict[str, Dict[str, Any]]:
    """Build per_model + tables (+ figures unless skipped) for one pipeline.

    Returns ``model_data`` keyed by model_type, for downstream README rendering.
    """
    models = PIPELINES[pipeline]
    if only_model is not None:
        if only_model not in models:
            print(f"[skip] {only_model} not in pipeline {pipeline}")
            return {}
        models = [only_model]

    pipeline_root = RESULTS_ROOT / pipeline
    (pipeline_root / "per_model").mkdir(parents=True, exist_ok=True)
    (pipeline_root / "tables").mkdir(parents=True, exist_ok=True)
    if not skip_figures:
        (pipeline_root / "figures").mkdir(parents=True, exist_ok=True)

    model_data: Dict[str, Dict[str, Any]] = {}

    for model_type in models:
        mpath = _metrics_path(pipeline, model_type)
        if not mpath.is_file():
            print(f"[skip] {pipeline}/{model_type}: no metrics.jsonl at {mpath}")
            continue
        rows = load_metrics(mpath)
        if not rows:
            print(f"[skip] {pipeline}/{model_type}: empty metrics.jsonl")
            continue

        # Identify canonical runs.
        dapt_runs = split_into_runs(rows, "dapt")
        ft_runs   = split_into_runs(rows, "finetune")
        canonical_dapt = dapt_runs[-1] if dapt_runs else []
        canonical_ft   = ft_runs[-1] if ft_runs else []

        # Per-model output dir
        mdir = pipeline_root / "per_model" / model_type
        mdir.mkdir(parents=True, exist_ok=True)

        # Curves
        if canonical_dapt:
            write_curve_csv(canonical_dapt, mdir / "dapt_curve.csv")
        if canonical_ft:
            write_curve_csv(canonical_ft, mdir / "finetune_curve.csv")

        # Inference probs (copy + CMs). We need these BEFORE writing
        # test_results.json so we can attach the bootstrap-CI accuracy and
        # AUC numbers, which require the per-sample probabilities.
        dapt_src, bl_src = find_latest_inference_probs(pipeline, model_type)
        copy_inference_probs(mdir, dapt_src, bl_src)
        if bl_src is not None:
            vol_cm, pat_cm = build_confusion_from_probs(mdir / "inference_probs_test.json")
            write_confusion_csv(vol_cm, mdir / "confusion_volume.csv")
            write_confusion_csv(pat_cm, mdir / "confusion_patient.csv")

        # Test results JSON, augmented with accuracy + macro/per-class AUC + CIs
        test_payload = write_test_results(
            rows, mdir / "test_results.json",
            dapt_probs=(mdir / "inference_probs_dapt_test.json") if dapt_src else None,
            bl_probs=(mdir / "inference_probs_test.json") if bl_src else None,
        )

        # Pbest checkpoint paths
        pbests = find_latest_pbests(pipeline, model_type)

        # Provenance
        write_provenance(
            mdir / "_provenance.json",
            model_type=model_type, pipeline=pipeline,
            metrics_jsonl=mpath,
            canonical_dapt=canonical_dapt, canonical_ft=canonical_ft,
            test_payload=test_payload, pbests=pbests,
            inference_sources=(dapt_src, bl_src),
        )

        model_data[model_type] = {
            "canonical_dapt": canonical_dapt,
            "canonical_ft":   canonical_ft,
            "test_payload":   test_payload,
            "pbests":         pbests,
        }
        print(f"[ok]  {pipeline}/{model_type}: "
              f"DAPT {len(canonical_dapt)} ep, FT {len(canonical_ft)} ep, "
              f"DAPT-test={'y' if test_payload['dapt_test'] else 'n'}, "
              f"BL-test={'y' if test_payload['biglunge_test'] else 'n'}")

    # Cross-model tables
    headline_rows = build_headline_rows(model_data)
    if headline_rows:
        write_headline_csv(headline_rows, pipeline_root / "tables" / "headline.csv")
        write_headline_md(headline_rows, pipeline_root / "tables" / "headline.md", pipeline)
        write_per_class_table(headline_rows, pipeline_root / "tables" / "per_class_metrics.csv")
        # Backward-compat: keep the old per_class_f1.csv name as a copy for
        # any thesis prose that might already cite it.
        shutil.copy(
            pipeline_root / "tables" / "per_class_metrics.csv",
            pipeline_root / "tables" / "per_class_f1.csv",
        )
    if model_data:
        write_training_summary(model_data, pipeline_root / "tables" / "training_summary.csv")

    # Figures
    if not skip_figures and model_data:
        try:
            _setup_mpl_style()
            kept = list(model_data.keys())
            fig_training_curves(pipeline, kept)
            fig_per_class_f1(pipeline, kept)
            fig_per_class_auc(pipeline, kept)
            fig_dapt_test_gap(pipeline, kept)
            fig_accuracy_gap(pipeline, kept)
            fig_macro_auc_gap(pipeline, kept)
            fig_roc_curves(pipeline, kept)
            fig_confusion_matrices(pipeline, kept)
        except ImportError as exc:
            print(f"[fig] matplotlib not available: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[fig:{pipeline}] failed: {exc}")

    return model_data


def snapshot_existing(out_root: Path, pipelines: List[str]) -> List[Path]:
    """Copy each pipeline's current ``thesis_results/<pipeline>/`` to
    ``thesis_results/_archive/<pipeline>_<UTC-timestamp>/`` before we
    overwrite. Returns the list of archive paths created.

    Rationale: a single training run is noisy. If a re-run produces worse
    numbers per-model than the prior run, you want to be able to revert
    that model to the earlier state without losing the new numbers for
    the other models. Snapshotting decouples the "latest-run-wins"
    determinism (good for reproducibility) from data loss risk.
    """
    archives: List[Path] = []
    if not out_root.is_dir():
        return archives
    archive_root = out_root / "_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    for pipeline in pipelines:
        src = out_root / pipeline
        if not src.is_dir():
            continue
        dst = archive_root / f"{pipeline}_{ts}"
        # copytree fails if dst exists; the timestamp guarantees uniqueness
        # at second granularity, but defend against clock-rewound edge cases.
        if dst.exists():
            dst = archive_root / f"{pipeline}_{ts}_{os.getpid()}"
        shutil.copytree(src, dst)
        archives.append(dst)
        print(f"[snapshot] {src.relative_to(out_root.parent)} -> {dst.relative_to(out_root.parent)}")
    return archives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=list(PIPELINES.keys()) + ["all"],
                        default="all")
    parser.add_argument("--model", default=None,
                        help="Restrict to a single model_type (must belong to --pipeline).")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Build per_model/ + tables/ but no figures.")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="Skip snapshotting the existing thesis_results/<pipeline>/ "
                             "to thesis_results/_archive/ before overwriting. Default is "
                             "to always snapshot.")
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    pipelines = list(PIPELINES.keys()) if args.pipeline == "all" else [args.pipeline]

    if not args.no_snapshot:
        snapshot_existing(RESULTS_ROOT, pipelines)

    model_data_per_pipeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for pipeline in pipelines:
        model_data_per_pipeline[pipeline] = build_pipeline(pipeline, args.model, args.skip_figures)

    write_readme(RESULTS_ROOT / "README.md", model_data_per_pipeline)
    print(f"\nthesis_results/ rebuilt — see {RESULTS_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
