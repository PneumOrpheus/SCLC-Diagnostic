"""Write 1-row dapt_curve.csv + finetune_curve.csv for each radiomics model.

The deep-pipeline figure code (``scripts/build_thesis_results.py``) reads
``dapt_curve.csv`` for the "DAPT val (peak)" bar in ``fig_accuracy_gap`` /
``fig_macro_auc_gap`` / ``fig_dapt_test_gap``. Radiomics has no per-epoch
training metrics (one-shot LASSO + fit), so we synthesise a single-row curve
using the val-set scores from the saved val inference-probabilities JSONs.

Conventions (mirrors the deep-pipeline framing):
  * ``dapt_curve.csv``     → val on **Lung-PET-CT-Dx** (in-sample for
    LPCT-trained models, cross-dataset transfer for BL-trained models).
  * ``finetune_curve.csv`` → val on **BigLunge** (in-sample for BL-trained,
    cross-dataset transfer for LPCT-trained).

The val probs JSONs are written by ``train_eval.train_and_eval_algo`` into
``results/output/2d/<model>/val/`` (see the ``test_dfs`` routing).

Run after ``train_eval`` and before ``build_thesis_results.py``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_2D = REPO_ROOT / "results" / "output" / "2d"
PER_MODEL_DIR = REPO_ROOT / "results" / "thesis" / "2d" / "per_model"

CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]
RADIOMICS_MODELS = (
    "radiomics_svm",    "radiomics_svm_bl",
    "radiomics_rf",     "radiomics_rf_bl",
    "radiomics_gb",     "radiomics_gb_bl",
)

CURVE_FIELDS = [
    "epoch", "train_loss", "train_macro_f1",
    "val_macro_f1_raw", "val_macro_f1_rolling",
    "val_loss", "val_accuracy", "val_balanced_accuracy",
    "lr_backbone", "lr_head",
    "mixup_alpha", "mixup_active",
    "epochs_no_improve", "monitor_window", "monitor_level",
    "timestamp",
]


def _latest_val_probs(model_type: str, kind: str) -> Optional[Path]:
    """``kind`` ∈ {'dapt_val', 'val'} — same routing as plots.py."""
    val_dir = OUT_2D / model_type / "val"
    if not val_dir.is_dir():
        return None
    if kind == "dapt_val":
        cands = sorted(val_dir.glob(f"{model_type}_*_dapt_val_inference_probabilities.json"),
                       key=lambda p: p.stat().st_mtime)
    else:
        cands = sorted(val_dir.glob(f"{model_type}_*_val_inference_probabilities.json"),
                       key=lambda p: p.stat().st_mtime)
        cands = [p for p in cands if "_dapt_val_" not in p.name]
    return cands[-1] if cands else None


def _val_metrics_from_probs(p: Path) -> Optional[Dict[str, float]]:
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    samples = payload.get("samples") or []
    if not samples:
        return None
    y_true, y_pred = [], []
    for s in samples:
        y_true.append(int(s["true_label"]))
        probs = s.get("probabilities") or {}
        order = [float(probs.get(c, 0.0)) for c in CLASS_NAMES]
        y_pred.append(int(np.argmax(order)))
    y_true = np.array(y_true, dtype=int); y_pred = np.array(y_pred, dtype=int)
    return {
        "val_accuracy":          float(accuracy_score(y_true, y_pred)),
        "val_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "val_macro_f1":          float(f1_score(y_true, y_pred, average="macro",
                                                labels=[0, 1, 2], zero_division=0)),
    }


def _write_curve(out_path: Path, m: Dict[str, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS, extrasaction="ignore")
        w.writeheader()
        row = {k: None for k in CURVE_FIELDS}
        row["epoch"] = 1
        row["val_macro_f1_raw"] = m["val_macro_f1"]
        row["val_macro_f1_rolling"] = m["val_macro_f1"]
        row["val_accuracy"] = m["val_accuracy"]
        row["val_balanced_accuracy"] = m["val_balanced_accuracy"]
        w.writerow(row)


def main() -> None:
    written = 0
    for mt in RADIOMICS_MODELS:
        per_model = PER_MODEL_DIR / mt
        per_model.mkdir(parents=True, exist_ok=True)

        dapt_p = _latest_val_probs(mt, "dapt_val")
        if dapt_p is not None:
            m = _val_metrics_from_probs(dapt_p)
            if m is not None:
                _write_curve(per_model / "dapt_curve.csv", m)
                print(f"[curves] {mt}/dapt_curve.csv ← {dapt_p.name} (val_acc={m['val_accuracy']:.3f})")
                written += 1

        ft_p = _latest_val_probs(mt, "val")
        if ft_p is not None:
            m = _val_metrics_from_probs(ft_p)
            if m is not None:
                _write_curve(per_model / "finetune_curve.csv", m)
                print(f"[curves] {mt}/finetune_curve.csv ← {ft_p.name} (val_acc={m['val_accuracy']:.3f})")
                written += 1
    print(f"\n[curves] wrote {written} curve CSVs total")


if __name__ == "__main__":
    main()
