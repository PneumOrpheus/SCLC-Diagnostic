"""Radiomics-pipeline visualization.

Produces:
  * PCA 2D scatter per dataset (PC1 vs PC2, colored by class).
  * Feature-importance bar charts (RF/GB built-ins, SVM via SHAP CSV).
  * LASSO-selected feature lists per model_type with coefficient magnitudes.
  * Confusion matrices per model_type / test set.
  * Per-class one-vs-rest ROC curves per dataset.

Outputs land in ``results/thesis/2d/figures/radiomics/``. Class colors match
the project palette in ``scripts/build_thesis_results.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
RAD_DIR = REPO_ROOT / "results" / "radiomics"
FIG_DIR = REPO_ROOT / "results" / "thesis" / "2d" / "figures" / "radiomics"
PROV_DIR = REPO_ROOT / "results" / "thesis" / "2d" / "per_model"
OUT_DIR = REPO_ROOT / "results" / "output" / "2d"

CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]
CLASS_COLORS = {
    "Adenocarcinoma": "#4E79A7",
    "Small Cell":     "#F28E2B",
    "Squamous":       "#59A14F",
}
SPLIT_MARKERS = {"train": "o", "val": "s", "test": "^"}
META_COLS = {"patient_id", "class_idx", "class_name", "split"}


# ---------- helpers ----------------------------------------------------------

def _load_stable(dataset: str) -> pd.DataFrame:
    p = RAD_DIR / f"stable_features_{dataset}.csv"
    return pd.read_csv(p)


def _feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def _provenance(model_type: str) -> Optional[Dict[str, Any]]:
    """Pull the radiomics-side metadata for ``model_type``.

    ``build_thesis_results.py``'s rebuild overwrites our per-model
    ``_provenance.json`` with the deep-model schema (zeroing out
    ``selected_features``/``hyperparams``). Source the real values from
    ``results/radiomics/train_eval_summary.json`` instead.
    """
    summary_path = RAD_DIR / "train_eval_summary.json"
    if not summary_path.is_file():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    for r in summary.get("model_reports", []):
        if r.get("model_type") == model_type:
            return r
    return None


def _latest_probs(model_type: str, phase: str) -> Optional[Path]:
    """Return the most recent probs JSON for ``model_type`` + ``phase``.

    Phase taxonomy (matches train_eval routing):
      * ``dapt_test`` — LPCT-Dx test (top-level ``<model>_*_dapt_inference_probabilities.json``)
      * ``test``      — BigLunge test (top-level, plain ``_inference_probabilities.json``)
      * ``dapt_val``  — LPCT-Dx val (under ``val/`` sub-dir)
      * ``val``       — BigLunge val   (under ``val/`` sub-dir)
    """
    out_dir = OUT_DIR / model_type
    if not out_dir.is_dir():
        return None
    if phase == "dapt_test":
        cands = sorted(out_dir.glob(f"{model_type}_*_dapt_inference_probabilities.json"),
                       key=lambda p: p.stat().st_mtime)
    elif phase == "test":
        cands = [p for p in sorted(out_dir.glob(f"{model_type}_*_inference_probabilities.json"),
                                   key=lambda p: p.stat().st_mtime)
                 if "_dapt_" not in p.name]
    elif phase == "dapt_val":
        cands = sorted((out_dir / "val").glob(f"{model_type}_*_dapt_val_inference_probabilities.json"),
                       key=lambda p: p.stat().st_mtime) if (out_dir / "val").is_dir() else []
    elif phase == "val":
        cands = sorted((out_dir / "val").glob(f"{model_type}_*_val_inference_probabilities.json"),
                       key=lambda p: p.stat().st_mtime) if (out_dir / "val").is_dir() else []
        # Exclude the dapt_val files (their name also matches "_val_").
        cands = [p for p in cands if "_dapt_val_" not in p.name]
    else:
        return None
    return cands[-1] if cands else None


def _load_probs(p: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (y_true, proba (N,3), patient_ids) from an inference_probabilities JSON.

    Returns empty arrays if the file is corrupt (e.g., the truncated
    swin_unetr DAPT probs JSON from the non-atomic-write bug). Callers
    must check the returned size before plotting.
    """
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[plots] WARN: skipping corrupt probs JSON {p}: {e}")
        return np.array([], dtype=int), np.zeros((0, 3), dtype=float), []
    y_true: List[int] = []
    proba: List[List[float]] = []
    pids: List[str] = []
    for s in payload.get("samples", []):
        y_true.append(int(s["true_label"]))
        probs = s.get("probabilities") or {}
        proba.append([float(probs.get(c, 0.0)) for c in CLASS_NAMES])
        pids.append(s.get("patient_id", ""))
    return np.array(y_true, dtype=int), np.array(proba, dtype=float), pids


# ---------- PCA --------------------------------------------------------------

def plot_pca(dataset: str) -> Path:
    df = _load_stable(dataset)
    feats = _feat_cols(df)
    X = df[feats].to_numpy(dtype=float)
    # z-score before PCA on the whole pool — purely for visualization.
    mu, sd = X.mean(axis=0), X.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    Xz = (X - mu) / sd
    pca = PCA(n_components=2, random_state=42).fit(Xz)
    Z = pca.transform(Xz)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    for cls_idx, cls in enumerate(CLASS_NAMES):
        mask = df["class_idx"].to_numpy() == cls_idx
        ax.scatter(Z[mask, 0], Z[mask, 1], s=22, alpha=0.7,
                   c=CLASS_COLORS[cls], label=cls,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)")
    label = "Lung-PET-CT-Dx" if dataset == "lpcd" else "BigLunge"
    ax.set_title(f"PCA — {label}  (n={len(df)}, {len(feats)} stable features)")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / f"pca_{dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


def plot_pca_loadings(dataset: str, top_k: int = 8) -> Path:
    """Bar chart of the top-|loading| features on PC1 and PC2."""
    df = _load_stable(dataset)
    feats = _feat_cols(df)
    X = df[feats].to_numpy(dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    pca = PCA(n_components=2, random_state=42).fit((X - mu) / sd)
    loadings = pca.components_  # (2, n_feats)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pc, vals in zip(axes, ("PC1", "PC2"), loadings):
        order = np.argsort(np.abs(vals))[::-1][:top_k][::-1]
        names = [_short(feats[i]) for i in order]
        ax.barh(names, vals[order],
                color=["#d9514e" if v >= 0 else "#5b8eda" for v in vals[order]])
        ax.set_title(f"{pc} loadings — top {top_k} by |loading|", fontsize=10)
        ax.set_xlabel("loading")
        ax.axvline(0, color="black", linewidth=0.5)
    fig.suptitle(f"PCA loadings — {'Lung-PET-CT-Dx' if dataset == 'lpcd' else 'BigLunge'}",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"pca_loadings_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


def _short(name: str, max_len: int = 38) -> str:
    """Shorten a PyRadiomics feature name like 'original_glcm_Idmn' -> 'glcm_Idmn'."""
    s = name.replace("original_", "").replace("Imc", "Imc")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


# ---------- Feature importance ----------------------------------------------

def plot_feature_importance_rf_gb(model_type: str, top_k: int = 15) -> Optional[Path]:
    """Refit the model and plot built-in feature_importances_ (RF, GB)."""
    if not (model_type.startswith("radiomics_rf") or model_type.startswith("radiomics_gb")):
        return None
    prov = _provenance(model_type)
    if prov is None:
        return None
    sel = prov.get("selected_features", [])
    if not sel:
        return None

    # Refit on the same train+val rows so importances reflect what's deployed.
    trained_on = prov.get("trained_on", "lpcd")
    df = _load_stable("lpcd" if trained_on == "lpcd" else "biglunge")
    train = df[df["split"].isin(("train", "val"))]
    feats_all = _feat_cols(df)
    # Stable features written for both datasets may differ; use intersection
    # via prov['selected_features'].
    keep = [c for c in sel if c in feats_all]
    if not keep:
        return None
    X = train[keep].to_numpy(dtype=float)
    y = train["class_idx"].to_numpy(dtype=int)
    # Use the same hyperparams the provenance recorded.
    hp = prov.get("hyperparams", {}) or {}
    if model_type.startswith("radiomics_rf"):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(class_weight="balanced", random_state=42, **hp).fit(X, y)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(random_state=42, **hp).fit(X, y)
    importance = clf.feature_importances_
    order = np.argsort(importance)[::-1][:top_k][::-1]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(order))))
    names = [_short(keep[i]) for i in order]
    ax.barh(names, importance[order], color="#4E79A7")
    ax.set_xlabel("Built-in feature importance")
    ax.set_title(f"{model_type}  feature importance  (top {len(order)} of {len(keep)} selected)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / f"feature_importance_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


def plot_feature_importance_svm_via_shap(model_type: str, top_k: int = 15) -> Optional[Path]:
    """SVM has no built-in importance; use SHAP top-10 CSV from interpret.py."""
    if not model_type.startswith("radiomics_svm"):
        return None
    csv_path = PROV_DIR / model_type / "shap_top10.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    glb = df[df["scope"] == "global"].sort_values("mean_abs_shap", ascending=False).head(top_k)
    if glb.empty:
        return None
    glb = glb.iloc[::-1]  # ascending for barh
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(glb))))
    names = [_short(s) for s in glb["feature"]]
    ax.barh(names, glb["mean_abs_shap"], color="#76B7B2")
    ax.set_xlabel("Mean |SHAP value| (test set)")
    ax.set_title(f"{model_type}  feature importance (SHAP)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / f"feature_importance_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


# ---------- LASSO selected features per model -------------------------------

def plot_selected_features() -> Path:
    """Heatmap-style table: which features are picked across the 6 radiomics
    model_types? Highlights overlap and divergence.
    """
    rows: List[Tuple[str, List[str]]] = []
    for mt in (
        "radiomics_svm", "radiomics_svm_bl",
        "radiomics_rf",  "radiomics_rf_bl",
        "radiomics_gb",  "radiomics_gb_bl",
    ):
        prov = _provenance(mt)
        if prov is None:
            continue
        rows.append((mt, prov.get("selected_features", []) or []))

    all_feats = sorted({f for _, fs in rows for f in fs})
    if not all_feats:
        return FIG_DIR / "selected_features.png"
    mat = np.zeros((len(rows), len(all_feats)), dtype=int)
    for i, (_, fs) in enumerate(rows):
        for j, f in enumerate(all_feats):
            mat[i, j] = 1 if f in fs else 0

    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(all_feats)),
                                    max(2.5, 0.45 * len(rows))))
    ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_feats)))
    ax.set_xticklabels([_short(f) for f in all_feats], rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([m for m, _ in rows], fontsize=8)
    ax.set_title("LASSO-selected features per radiomics model", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "selected_features.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


# ---------- Confusion matrices ----------------------------------------------

def _plot_cm(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["A", "S", "Q"], fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(["A", "S", "Q"], fontsize=9)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("true", fontsize=9)
    ax.set_title(title, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8)


def plot_confusion_matrices() -> Path:
    """Grid of confusion matrices: rows = (cohort, model-family), cols = algo.

    Test-only for now (BigLunge val intentionally not shown — the val files
    exist on disk under ``output/2d/<model>/val/`` but the user wants test
    figures first).
    """
    rows = [
        ("LPCT-Dx test (LPCT-trained)",               "dapt_test", ("radiomics_svm",    "radiomics_rf",    "radiomics_gb")),
        ("BigLunge test (LPCT-trained, transfer)",    "test",      ("radiomics_svm",    "radiomics_rf",    "radiomics_gb")),
        ("BigLunge test (BL-trained, in-sample)",     "test",      ("radiomics_svm_bl", "radiomics_rf_bl", "radiomics_gb_bl")),
        ("LPCT-Dx test (BL-trained, transfer)",       "dapt_test", ("radiomics_svm_bl", "radiomics_rf_bl", "radiomics_gb_bl")),
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 2.7 * len(rows)))
    for r, (row_label, phase, models) in enumerate(rows):
        for c, mt in enumerate(models):
            probs_path = _latest_probs(mt, phase)
            ax = axes[r][c]
            if probs_path is None:
                ax.set_axis_off()
                if c == 0:
                    ax.text(0.5, 0.5, "(no probs JSON)", ha="center", va="center", fontsize=9)
                continue
            y_true, proba, _ = _load_probs(probs_path)
            if y_true.size == 0:
                ax.set_axis_off()
                continue
            y_pred = np.argmax(proba, axis=1)
            _plot_cm(ax, y_true, y_pred, title=mt)
        axes[r][0].set_ylabel(row_label + "\ntrue", fontsize=8)
    fig.suptitle("Confusion matrices — radiomics pipeline (A=Adeno, S=SmallCell, Q=Squamous)",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


# ---------- ROC curves -------------------------------------------------------

def plot_roc_per_dataset(
    phase: str,
    models: Tuple[str, ...],
    dataset_label: str, fname: str,
) -> Path:
    """Per-class one-vs-rest ROC for the given phase ('dapt_test' or 'test')."""
    palette = ["#4E79A7", "#F28E2B", "#76B7B2", "#59A14F", "#E15759", "#B07AA1"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    for c, cname in enumerate(CLASS_NAMES):
        ax = axes[c]
        for m_i, mt in enumerate(models):
            probs_path = _latest_probs(mt, phase)
            if probs_path is None:
                continue
            y_true, proba, _ = _load_probs(probs_path)
            y_bin = (y_true == c).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            fpr, tpr, _ = roc_curve(y_bin, proba[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=palette[m_i % len(palette)], linewidth=1.7,
                    label=f"{mt}  AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR", fontsize=9)
        if c == 0:
            ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(cname, fontsize=10, color=CLASS_COLORS[cname])
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"ROC curves (one-vs-rest) — {dataset_label}", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


# ---------- top-level --------------------------------------------------------

# =============================================================================
# DL-equivalent figures: same grid layouts driven by saved DL test probs.
# (Val probs for DL are not currently written during training; would require
# a re-run of `--mode inference` against the val split for each saved
# checkpoint. Test rows fill the figure for now; val will be added when the
# DL val-inference collector lands.)
# =============================================================================

DL_MODELS_2D = (
    "efficientnet_b0_2d", "densenet121_2d", "resnet50_2d",
    "swin_tiny_2d", "resnet50_2d_rin", "densenet121_2d_rin",
)
DL_MODELS_MIL = ("mil_resnet50", "mil_swin_tiny")
DL_MODELS_3D = ("swin_unetr",)
DL_FIG_DIR = REPO_ROOT / "results" / "thesis" / "_unified_dl_radiomics"


def _dl_latest_probs(pipeline: str, model_type: str, phase: str) -> Optional[Path]:
    out_dir = REPO_ROOT / "results" / "output" / pipeline / model_type
    if not out_dir.is_dir():
        return None
    if phase == "dapt_test":
        cands = sorted(out_dir.glob(f"{model_type}_*_dapt_inference_probabilities.json"),
                       key=lambda p: p.stat().st_mtime)
    else:
        cands = [p for p in sorted(out_dir.glob(f"{model_type}_*_inference_probabilities.json"),
                                   key=lambda p: p.stat().st_mtime)
                 if "_dapt_" not in p.name]
    return cands[-1] if cands else None


def plot_dl_confusion_matrices() -> Path:
    """Confusion matrices for all DL models on both LPCT-Dx test and BL test."""
    DL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    families = [
        ("2d/dapt_test",  "LPCT-Dx test (2D)",        "2d",  "dapt_test", DL_MODELS_2D),
        ("2d/test",       "BigLunge test (2D)",        "2d",  "test",      DL_MODELS_2D),
        ("mil/dapt_test", "LPCT-Dx test (MIL DAPT)",  "mil", "dapt_test", DL_MODELS_MIL),
        ("mil/test",      "BigLunge test (MIL FT)",    "mil", "test",      DL_MODELS_MIL),
        ("3d/dapt_test",  "LPCT-Dx test (3D)",        "3d",  "dapt_test", DL_MODELS_3D),
        ("3d/test",       "BigLunge test (3D)",        "3d",  "test",      DL_MODELS_3D),
    ]
    n_rows = len(families)
    n_cols = max(len(m) for *_, m in families)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols + 1, 2.6 * n_rows))
    if n_rows == 1: axes = [axes]
    for r, (_, row_label, pipeline, phase, models) in enumerate(families):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(models):
                ax.set_axis_off(); continue
            mt = models[c]
            probs_path = _dl_latest_probs(pipeline, mt, phase)
            if probs_path is None:
                ax.set_axis_off(); continue
            y_true, proba, _ = _load_probs(probs_path)
            if y_true.size == 0:
                ax.set_axis_off()
                continue
            y_pred = np.argmax(proba, axis=1)
            _plot_cm(ax, y_true, y_pred, title=mt)
        axes[r][0].set_ylabel(row_label + "\ntrue", fontsize=8)
    fig.suptitle("Confusion matrices — deep models (test only)", fontsize=11)
    fig.tight_layout()
    out = DL_FIG_DIR / "dl_confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


def plot_dl_roc(phase: str, dataset_label: str, fname: str) -> Path:
    DL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    palette = ["#4E79A7", "#F28E2B", "#76B7B2", "#59A14F", "#E15759", "#B07AA1", "#FFB04A", "#9C755F", "#BAB0AC"]
    triples = [("2d", m) for m in DL_MODELS_2D] + [("mil", m) for m in DL_MODELS_MIL] + [("3d", m) for m in DL_MODELS_3D]
    for c, cname in enumerate(CLASS_NAMES):
        ax = axes[c]
        for i, (pipeline, mt) in enumerate(triples):
            probs_path = _dl_latest_probs(pipeline, mt, phase)
            if probs_path is None:
                continue
            y_true, proba, _ = _load_probs(probs_path)
            y_bin = (y_true == c).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            fpr, tpr, _ = roc_curve(y_bin, proba[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=palette[i % len(palette)], linewidth=1.4,
                    label=f"{mt}  AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR", fontsize=9)
        if c == 0: ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(cname, fontsize=10, color=CLASS_COLORS[cname])
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle(f"ROC curves (one-vs-rest) — {dataset_label} (deep models)", fontsize=11)
    fig.tight_layout()
    out = DL_FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


def plot_dl_pca(pipeline: str, models: Tuple[str, ...], dataset: str, fname: str) -> Optional[Path]:
    """PCA on the per-patient softmax outputs of all DL models in a pipeline.

    This is a pragmatic stand-in for "PCA on penultimate-layer features"
    without re-running the models — the softmax distributions still capture
    each model's class-separation signal, and PCA on them visualises whether
    the deep models cluster the cohort similarly to the radiomics features.
    Re-running val/test forwards to grab penultimate features is a follow-up
    worth doing, but is a separate compute job.
    """
    phase = "dapt_test" if dataset == "lpcd" else "test"
    rows: List[List[float]] = []
    pids: List[str] = []
    labels: List[int] = []
    for mt in models:
        probs_path = _dl_latest_probs(pipeline, mt, phase)
        if probs_path is None:
            continue
        y, proba, ids = _load_probs(probs_path)
        for i, pid in enumerate(ids):
            rows.append(proba[i].tolist())
            pids.append(pid)
            labels.append(int(y[i]))
    if not rows:
        return None
    # Average per-patient probs across models so each patient = 1 row.
    df = pd.DataFrame(rows, columns=CLASS_NAMES)
    df["patient_id"] = pids
    df["class_idx"] = labels
    agg = df.groupby("patient_id").agg({c: "mean" for c in CLASS_NAMES} | {"class_idx": "first"})
    X = agg[CLASS_NAMES].to_numpy()
    pca = PCA(n_components=2, random_state=42).fit(X)
    Z = pca.transform(X)
    var = pca.explained_variance_ratio_

    DL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cls_idx, cls in enumerate(CLASS_NAMES):
        m = agg["class_idx"].to_numpy() == cls_idx
        ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.7, c=CLASS_COLORS[cls], label=cls,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ds_label = "Lung-PET-CT-Dx (test)" if dataset == "lpcd" else "BigLunge (test)"
    ax.set_title(f"PCA of mean DL softmax outputs ({pipeline}) — {ds_label}\n"
                 f"(n={len(agg)}, {len(models)} models averaged)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = DL_FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}")
    return out


def run() -> List[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    out.append(plot_pca("lpcd"))
    out.append(plot_pca("biglunge"))
    out.append(plot_pca_loadings("lpcd"))
    out.append(plot_pca_loadings("biglunge"))

    for mt in ("radiomics_svm", "radiomics_svm_bl",
               "radiomics_rf", "radiomics_rf_bl",
               "radiomics_gb", "radiomics_gb_bl"):
        if mt.startswith("radiomics_svm"):
            r = plot_feature_importance_svm_via_shap(mt)
        else:
            r = plot_feature_importance_rf_gb(mt)
        if r:
            out.append(r)

    out.append(plot_selected_features())
    out.append(plot_confusion_matrices())
    out.append(plot_roc_per_dataset(
        "dapt_test",
        ("radiomics_svm", "radiomics_rf", "radiomics_gb"),
        "Lung-PET-CT-Dx (test, LPCT-trained, in-sample)",
        "roc_lpcd_insample.png",
    ))
    out.append(plot_roc_per_dataset(
        "test",
        ("radiomics_svm", "radiomics_rf", "radiomics_gb"),
        "BigLunge (test, LPCT-trained, cross-dataset transfer)",
        "roc_biglunge_xfer.png",
    ))
    out.append(plot_roc_per_dataset(
        "test",
        ("radiomics_svm_bl", "radiomics_rf_bl", "radiomics_gb_bl"),
        "BigLunge (test, BL-trained, in-sample)",
        "roc_biglunge_insample.png",
    ))
    out.append(plot_roc_per_dataset(
        "dapt_test",
        ("radiomics_svm_bl", "radiomics_rf_bl", "radiomics_gb_bl"),
        "Lung-PET-CT-Dx (test, BL-trained, cross-dataset transfer)",
        "roc_lpcd_xfer.png",
    ))
    return out


def main() -> None:
    paths = run()
    print()
    print(f"=== Wrote {len(paths)} figure(s) to {FIG_DIR} ===")


if __name__ == "__main__":
    main()
