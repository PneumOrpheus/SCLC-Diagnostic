"""Phase 4: SHAP feature attributions for the radiomics models.

For each fitted model_type produced by ``train_eval.py``, compute SHAP values
on the held-out test set and write:

  * ``results/thesis/2d/per_model/<model_type>/shap_top10.csv`` — top features
    by mean(|SHAP|) globally and per class.
  * ``results/thesis/2d/figures/radiomics/<model_type>_shap_summary.png`` —
    standard SHAP summary plot.

We refit the model with the same recipe used in ``train_eval.fit_final`` so
the SHAP analysis sees the same z-scored, LASSO-selected, SMOTE-balanced
feature space the actual classifier was trained on. (We don't pickle the
classifier; this stays in-memory.)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sclc.radiomics.train_eval import (
    META_COLS,
    OUTPUT_ROOT,
    RADIOMICS_DIR,
    fit_final,
    load_stable,
    make_models,
    split_xy,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_2D = REPO_ROOT / "results" / "thesis" / "2d"
PER_MODEL_DIR = THESIS_2D / "per_model"
FIG_DIR = THESIS_2D / "figures" / "radiomics"


def _explainer_for(algo: str, clf, background: np.ndarray):
    import shap
    if algo == "svm":
        # Kernel explainer with k-means background to keep runtime sane.
        bg = shap.sample(background, min(50, len(background)), random_state=42)
        return shap.KernelExplainer(clf.predict_proba, bg)
    if algo in ("rf", "gb"):
        return shap.TreeExplainer(clf)
    raise ValueError(f"No SHAP explainer wired for algo={algo}")


def _shap_values_array(values: Any, n_samples: int, n_feats: int, n_classes: int = 3) -> np.ndarray:
    """Normalize SHAP output into shape (n_classes, n_samples, n_feats).

    SHAP returns either a list of per-class arrays (older API), or a single
    ndarray with class as the trailing axis (newer API). Normalize so we can
    index by class consistently.
    """
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=0)
    arr = np.asarray(values)
    if arr.ndim == 3:
        # Either (n_classes, n_samples, n_feats) or (n_samples, n_feats, n_classes).
        if arr.shape[0] == n_classes and arr.shape[1] == n_samples:
            return arr
        if arr.shape[-1] == n_classes and arr.shape[0] == n_samples:
            return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 2:
        # Binary case shouldn't apply here, but keep robust.
        return np.stack([arr] * n_classes, axis=0)
    raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")


def _save_summary_plot(
    shap_values: np.ndarray,  # (n_samples, n_feats)
    X: np.ndarray,
    feat_names: List[str],
    out_path: Path,
    title: str,
) -> None:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    shap.summary_plot(shap_values, X, feature_names=feat_names, show=False, max_display=15)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def interpret_one(
    algo: str, model_type: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
    inner_summary: Dict[str, Any], feat_names: List[str],
) -> Dict[str, Any]:
    print(f"[interpret] {model_type} (algo={algo})")
    X_tr, y_tr, _ = split_xy(train_df, ("train", "val"))
    hp = inner_summary[algo]["best_params"]
    clf, mu, sigma, sel_names = fit_final(algo, hp, X_tr, y_tr, feat_names)

    test_sub = test_df[test_df["split"] == "test"].reset_index(drop=True)
    if test_sub.empty:
        print(f"  [interpret] no test rows for {model_type}; skipping")
        return {}
    # Align columns to train feature space.
    for c in feat_names:
        if c not in test_sub.columns:
            test_sub[c] = 0.0
    X_te = test_sub[feat_names].to_numpy(dtype=float)
    X_te_z = (X_te - mu) / sigma
    sel_idx = [feat_names.index(n) for n in sel_names]
    X_te_sel = X_te_z[:, sel_idx]

    expl = _explainer_for(algo, clf, X_te_sel)
    raw = expl.shap_values(X_te_sel)
    shap_arr = _shap_values_array(raw, n_samples=X_te_sel.shape[0], n_feats=len(sel_names))

    # Top features by mean(|SHAP|) — globally and per class.
    abs_per_class = np.abs(shap_arr).mean(axis=1)  # (n_classes, n_feats)
    abs_global = abs_per_class.mean(axis=0)         # (n_feats,)

    rows: List[Dict[str, Any]] = []
    order_global = np.argsort(abs_global)[::-1]
    for rank, fi in enumerate(order_global[:10]):
        rows.append({
            "rank": rank + 1,
            "scope": "global",
            "feature": sel_names[int(fi)],
            "mean_abs_shap": float(abs_global[fi]),
            "abs_shap_adeno": float(abs_per_class[0, fi]),
            "abs_shap_smallcell": float(abs_per_class[1, fi]),
            "abs_shap_squamous": float(abs_per_class[2, fi]),
        })
    for c, cname in enumerate(["Adenocarcinoma", "Small Cell", "Squamous"]):
        order_c = np.argsort(abs_per_class[c])[::-1]
        for rank, fi in enumerate(order_c[:10]):
            rows.append({
                "rank": rank + 1,
                "scope": cname,
                "feature": sel_names[int(fi)],
                "mean_abs_shap": float(abs_per_class[c, fi]),
                "abs_shap_adeno": float(abs_per_class[0, fi]),
                "abs_shap_smallcell": float(abs_per_class[1, fi]),
                "abs_shap_squamous": float(abs_per_class[2, fi]),
            })

    out_csv = PER_MODEL_DIR / model_type / "shap_top10.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [interpret] {out_csv}")

    # Summary plot — pool magnitudes across classes for a single PNG.
    pooled = np.mean(np.abs(shap_arr), axis=0)  # (n_samples, n_feats)
    fig_path = FIG_DIR / f"{model_type}_shap_summary.png"
    _save_summary_plot(pooled, X_te_sel, sel_names, fig_path, title=f"{model_type} SHAP (mean |value| across classes)")
    print(f"  [interpret] {fig_path}")

    return {
        "shap_top_csv": str(out_csv),
        "summary_plot": str(fig_path),
        "selected_features": sel_names,
    }


def run() -> Dict[str, Any]:
    summary_path = RADIOMICS_DIR / "train_eval_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} not found; run sclc.radiomics.train_eval first.")
    with open(summary_path) as f:
        summary = json.load(f)
    inner_lpcd = summary["inner_cv_lpcd"]
    inner_bl = summary["inner_cv_biglunge"]

    lpcd = load_stable("lpcd")
    bl = load_stable("biglunge")
    feat_lpcd = [c for c in lpcd.columns if c not in META_COLS]
    feat_bl = [c for c in bl.columns if c not in META_COLS]
    common = sorted(set(feat_lpcd) & set(feat_bl))
    keep_lpcd = [c for c in lpcd.columns if c in META_COLS or c in common]
    keep_bl = [c for c in bl.columns if c in META_COLS or c in common]
    lpcd = lpcd[keep_lpcd]
    bl = bl[keep_bl]

    out: Dict[str, Any] = {}
    # Run SHAP for each LPCT-trained algo on LPCT-test (the most-cited in-sample setup).
    for algo in ("svm", "rf", "gb"):
        model_type = f"radiomics_{algo}"
        out[model_type] = interpret_one(algo, model_type, lpcd, lpcd, inner_lpcd, common)

    # Run SHAP for each BL-trained algo on BL-test.
    for algo in ("svm", "rf", "gb"):
        model_type = f"radiomics_{algo}_bl"
        out[model_type] = interpret_one(algo, model_type, bl, bl, inner_bl, common)

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    args = p.parse_args()
    run()


if __name__ == "__main__":
    main()
