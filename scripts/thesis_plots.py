"""Generate every figure used in the thesis from `metrics.jsonl` files.

Each `make_*` function reads from ``output/{pipeline}/{model_type}/metrics.jsonl``
and writes a PDF to ``figures/``. Skips gracefully when a model's metrics
aren't on disk yet, so this script is safe to run while the pipeline is
still training.

Run all plots::

    python scripts/thesis_plots.py

Or a subset::

    python scripts/thesis_plots.py --only per_class_f1 confusion_matrices

Order of figures (matches the thesis ordering):

1. fig_class_distribution.pdf   — patients-per-class per dataset (context)
2. fig_dropped_patients.pdf     — silent-drop audit (limitations)
3. fig_training_curves.pdf      — val MacroF1 vs epoch, DAPT + FT panels
4. fig_dapt_test_gap.pdf        — DAPT-val / DAPT-test / BigLunge-test bars
5. fig_per_class_f1.pdf         — per-class F1 + 95% CI per model
6. fig_confusion_matrices.pdf   — 3×3 heatmaps per model
7. fig_mil_attention.pdf        — attention vs slice z-index (weak-supervision contribution)

The MIL attention figure is computed at draw-time by running the MIL
checkpoint on the BigLunge val bag-loader; it requires the trained MIL
checkpoint on disk and a valid CUDA device.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Make the repo importable when running this file as a script.
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

MODELS_2D: List[str] = [
    "efficientnet_b0_2d",
    "resnet50_2d",
    "densenet121_2d",
    "swin_tiny_2d",
]
MODEL_MIL = "mil_resnet50"
MODEL_3D = "swin_unetr"

# Display order in panels (left-to-right). Architecturally-grouped: 2D
# tumor-crop CNNs, then transformer (Swin-Tiny 2D), then MIL, then 3D.
DEFAULT_MODEL_ORDER: List[str] = MODELS_2D + [MODEL_MIL, MODEL_3D]

# Pretty labels for axis/legend.
MODEL_LABELS: Dict[str, str] = {
    "efficientnet_b0_2d": "EffNet-B0 (2D)",
    "densenet121_2d": "DenseNet121 (2D)",
    "resnet50_2d": "ResNet-50 (2D)",
    "swin_tiny_2d": "Swin-Tiny (2D)",
    "mil_resnet50": "MIL ResNet-50",
    "swin_unetr": "SwinUNETR (3D)",
}

# Pipeline directory under output/. Used to find metrics.jsonl.
PIPELINE_OF: Dict[str, str] = {
    **{m: "2d" for m in MODELS_2D},
    "mil_resnet50": "mil",
    "swin_unetr": "3d",
}

CLASS_NAMES: List[str] = ["Adenocarcinoma", "Small Cell", "Squamous"]
# Colorblind-friendly palette (Tableau 10, reordered).
CLASS_COLORS: Dict[str, str] = {
    "Adenocarcinoma": "#4E79A7",  # blue
    "Small Cell": "#F28E2B",      # orange
    "Squamous": "#59A14F",        # green
}

# Pretty per-model colors for training curves. Uses the Tableau 10 palette.
MODEL_COLORS: Dict[str, str] = {
    "efficientnet_b0_2d": "#4E79A7",
    "resnet50_2d": "#F28E2B",
    "densenet121_2d": "#59A14F",
    "swin_tiny_2d": "#E15759",
    "mil_resnet50": "#76B7B2",
    "swin_unetr": "#B07AA1",
}


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

def setup_style() -> None:
    """Thesis-grade matplotlib defaults. Vector output, readable fonts."""
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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # editable text in the resulting PDF
    })


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------

def metrics_path(model_type: str, output_root: str = "output") -> Path:
    return Path(output_root) / PIPELINE_OF[model_type] / model_type / "metrics.jsonl"


def load_metrics_rows(model_type: str, output_root: str = "output") -> List[Dict[str, Any]]:
    p = metrics_path(model_type, output_root)
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_row(rows: List[Dict[str, Any]], phase: str) -> Optional[Dict[str, Any]]:
    """Most recent row tagged with the given phase (timestamp-sorted)."""
    matching = [r for r in rows if r.get("phase") == phase]
    if not matching:
        return None
    matching.sort(key=lambda r: r.get("timestamp", ""))
    return matching[-1]


def per_epoch_phase(rows: List[Dict[str, Any]], phase: str) -> List[Dict[str, Any]]:
    """All rows of a given phase, sorted by epoch ascending. Drops rows with no epoch."""
    matching = [r for r in rows if r.get("phase") == phase and r.get("epoch") is not None]
    matching.sort(key=lambda r: int(r.get("epoch") or 0))
    return matching


def patient_block(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the row's patient-level sub-dict (training rows use ``val_patient``,
    test-inference rows use ``test_patient``). Empty dict if neither.
    """
    if "test_patient" in row:
        return row.get("test_patient") or {}
    if "val_patient" in row:
        return row.get("val_patient") or {}
    return {}


# ----------------------------------------------------------------------------
# Figure: training curves (val MacroF1 vs epoch, DAPT + FT panels)
# ----------------------------------------------------------------------------

def make_training_curves(
    output_root: str = "output",
    out_path: str = "figures/fig_training_curves.pdf",
    models: Optional[List[str]] = None,
) -> Optional[str]:
    models = models or DEFAULT_MODEL_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    ax_dapt, ax_ft = axes

    plotted_any = False
    for m in models:
        rows = load_metrics_rows(m, output_root)
        if not rows:
            continue
        # DAPT panel: phase ∈ {"dapt", "dapt_lp"}.
        for phase in ("dapt", "dapt_lp"):
            ep_rows = per_epoch_phase(rows, phase)
            if not ep_rows:
                continue
            xs = [int(r["epoch"]) for r in ep_rows]
            ys_raw = [float(r.get("val_macro_f1", float("nan"))) for r in ep_rows]
            ys_roll = [float(r.get("val_macro_f1_rolling", float("nan"))) for r in ep_rows]
            color = MODEL_COLORS.get(m, "#444")
            ls = "-" if phase == "dapt" else "--"
            ax_dapt.plot(xs, ys_raw, color=color, alpha=0.35, linewidth=1.0, linestyle=ls)
            ax_dapt.plot(xs, ys_roll, color=color, linewidth=1.7,
                         linestyle=ls, label=MODEL_LABELS.get(m, m))
            plotted_any = True

        # Fine-tune panel.
        ep_rows = per_epoch_phase(rows, "finetune")
        if ep_rows:
            xs = [int(r["epoch"]) for r in ep_rows]
            ys_raw = [float(r.get("val_macro_f1", float("nan"))) for r in ep_rows]
            ys_roll = [float(r.get("val_macro_f1_rolling", float("nan"))) for r in ep_rows]
            color = MODEL_COLORS.get(m, "#444")
            ax_ft.plot(xs, ys_raw, color=color, alpha=0.35, linewidth=1.0)
            ax_ft.plot(xs, ys_roll, color=color, linewidth=1.7,
                       label=MODEL_LABELS.get(m, m))
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[training_curves] no metrics found, skipping.")
        return None

    # LP-FT freeze boundary annotation (if any FT row had freeze).
    ax_ft.axvline(5.5, color="#999", linestyle=":", linewidth=1)
    ax_ft.text(5.5, ax_ft.get_ylim()[1] * 0.05, "  LP→FT", color="#666",
               fontsize=8, va="bottom", ha="left")

    ax_dapt.set_xlabel("Epoch")
    ax_dapt.set_ylabel("Patient-level MacroF1 (val)")
    ax_dapt.set_title("DAPT on Lung-PET-CT-Dx")
    ax_dapt.grid(alpha=0.3, linestyle=":")

    ax_ft.set_xlabel("Epoch")
    ax_ft.set_title("Fine-tune on BigLunge")
    ax_ft.grid(alpha=0.3, linestyle=":")

    ax_dapt.legend(loc="lower right", frameon=False)
    fig.suptitle("Validation MacroF1 over training epochs (raw thin / rolling-3 thick)",
                 fontsize=11, y=1.02)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[training_curves] -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# Figure: DAPT-val / DAPT-test / BigLunge-test bars per model
# ----------------------------------------------------------------------------

def make_dapt_test_gap(
    output_root: str = "output",
    out_path: str = "figures/fig_dapt_test_gap.pdf",
    models: Optional[List[str]] = None,
) -> Optional[str]:
    models = models or DEFAULT_MODEL_ORDER
    rows_per_model: Dict[str, Tuple[Optional[float], Optional[List[float]],
                                    Optional[float], Optional[List[float]],
                                    Optional[float], Optional[List[float]]]] = {}

    for m in models:
        rows = load_metrics_rows(m, output_root)
        if not rows:
            continue
        # DAPT val: take the best-by-rolling-mean epoch from the DAPT phase.
        dapt_rows = per_epoch_phase(rows, "dapt") or per_epoch_phase(rows, "dapt_lp")
        dapt_val_mf1 = None
        dapt_val_ci = None
        if dapt_rows:
            best = max(dapt_rows, key=lambda r: float(r.get("val_macro_f1_rolling") or 0))
            p = best.get("val_patient") or {}
            dapt_val_mf1 = p.get("macro_f1", best.get("val_macro_f1"))
            dapt_val_ci = p.get("macro_f1_ci95")

        # DAPT test
        dt = latest_row(rows, "dapt_test")
        dapt_test_mf1 = None
        dapt_test_ci = None
        if dt is not None:
            p = dt.get("test_patient") or {}
            dapt_test_mf1 = p.get("macro_f1", dt.get("test_macro_f1"))
            dapt_test_ci = p.get("macro_f1_ci95")

        # BigLunge test
        bt = latest_row(rows, "test")
        bl_test_mf1 = None
        bl_test_ci = None
        if bt is not None:
            p = bt.get("test_patient") or {}
            bl_test_mf1 = p.get("macro_f1", bt.get("test_macro_f1"))
            bl_test_ci = p.get("macro_f1_ci95")

        rows_per_model[m] = (
            dapt_val_mf1, dapt_val_ci,
            dapt_test_mf1, dapt_test_ci,
            bl_test_mf1, bl_test_ci,
        )

    if not rows_per_model:
        print("[dapt_test_gap] no rows found, skipping.")
        return None

    n = len(rows_per_model)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * n + 2), 4.5))
    width = 0.26
    xs = np.arange(n)
    series = ["DAPT val (peak roll-3)", "DAPT test (n=53)", "BigLunge test (n≈44)"]
    colors = ["#9DB7C8", "#4E79A7", "#E15759"]

    for k, (label, color) in enumerate(zip(series, colors)):
        ys = []
        yerr_lo = []
        yerr_hi = []
        for m in rows_per_model:
            tup = rows_per_model[m]
            mf1 = tup[2 * k]
            ci = tup[2 * k + 1]
            if mf1 is None:
                ys.append(np.nan)
                yerr_lo.append(0)
                yerr_hi.append(0)
                continue
            ys.append(float(mf1))
            if ci is not None and len(ci) == 2:
                yerr_lo.append(float(mf1) - float(ci[0]))
                yerr_hi.append(float(ci[1]) - float(mf1))
            else:
                yerr_lo.append(0)
                yerr_hi.append(0)
        ax.bar(xs + (k - 1) * width, ys, width=width, label=label, color=color,
               yerr=[yerr_lo, yerr_hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in rows_per_model.keys()],
                       rotation=25, ha="right")
    ax.set_ylabel("Patient-level MacroF1 (95% CI)")
    ax.set_ylim(0, 1)
    ax.set_title("Generalization gap: DAPT-val → DAPT-test → BigLunge-test")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[dapt_test_gap] -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# Figure: per-class F1 with 95% CI per model on BigLunge test
# ----------------------------------------------------------------------------

def make_per_class_f1(
    output_root: str = "output",
    out_path: str = "figures/fig_per_class_f1.pdf",
    phase: str = "test",
    models: Optional[List[str]] = None,
    title_suffix: str = " (BigLunge test)",
) -> Optional[str]:
    """Bar chart: for each model, three bars (one per class) with 95% CI.

    Pass ``phase="dapt_test"`` for the Lung-PET-CT-Dx test split version.
    """
    models = models or DEFAULT_MODEL_ORDER
    data: Dict[str, Tuple[List[float], List[Tuple[float, float]]]] = {}
    for m in models:
        rows = load_metrics_rows(m, output_root)
        r = latest_row(rows, phase) if rows else None
        if r is None:
            continue
        p = r.get("test_patient") or {}
        per_cls = p.get("per_class_f1") or []
        per_cls_ci = p.get("per_class_f1_ci95") or []
        if len(per_cls) < 3:
            continue
        cis = []
        for c in range(3):
            ci = per_cls_ci[c] if c < len(per_cls_ci) else (per_cls[c], per_cls[c])
            cis.append((float(ci[0]), float(ci[1])))
        data[m] = ([float(x) for x in per_cls[:3]], cis)

    if not data:
        print(f"[per_class_f1:{phase}] no rows found, skipping.")
        return None

    n_models = len(data)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * n_models + 2), 4.5))
    width = 0.26
    xs = np.arange(n_models)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ys = []
        err_lo = []
        err_hi = []
        for m in data:
            f1s, cis = data[m]
            ys.append(f1s[cls_idx])
            err_lo.append(f1s[cls_idx] - cis[cls_idx][0])
            err_hi.append(cis[cls_idx][1] - f1s[cls_idx])
        offset = (cls_idx - 1) * width
        ax.bar(xs + offset, ys, width=width, label=cls_name,
               color=CLASS_COLORS[cls_name],
               yerr=[err_lo, err_hi], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in data.keys()],
                       rotation=25, ha="right")
    ax.set_ylabel("Per-class F1 (95% CI)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class F1{title_suffix}")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[per_class_f1:{phase}] -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# Figure: confusion matrices (3×3 heatmap per model)
# ----------------------------------------------------------------------------

def make_confusion_matrices(
    output_root: str = "output",
    out_path: str = "figures/fig_confusion_matrices.pdf",
    phase: str = "test",
    models: Optional[List[str]] = None,
    title_suffix: str = " (BigLunge test)",
) -> Optional[str]:
    """Per-model confusion matrices reconstructed from per-class precision/recall.

    The metrics.jsonl rows store ``per_class_precision`` and ``per_class_recall``
    plus support — that's enough to invert back to the confusion matrix:
        TP_c = round(recall_c * support_c)
        FP_c = round(TP_c / max(precision_c, eps) - TP_c)
    Off-diagonals can't be recovered uniquely from class-level metrics, so we
    pull the explicit confusion matrix from the inference probability JSON if
    available, falling back to a TP-only diagonal-plus-uncertainty render.
    """
    models = models or DEFAULT_MODEL_ORDER

    # Try the standalone inference_probabilities JSON files first — they
    # contain per-sample (true_label, pred_label) so we can rebuild the CM.
    cms: Dict[str, np.ndarray] = {}
    n_patients: Dict[str, int] = {}
    for m in models:
        pipeline = PIPELINE_OF[m]
        out_dir = Path(output_root) / pipeline / m
        if not out_dir.is_dir():
            continue
        # Pick the most recent inference probabilities JSON.
        suffix = "_inference_probabilities.json" if phase == "test" else "_dapt_inference_probabilities.json"
        candidates = sorted(out_dir.glob(f"*{suffix}"))
        if not candidates:
            # Try any matching json.
            candidates = sorted(out_dir.glob("*inference_probabilities*.json"))
        if not candidates:
            continue
        try:
            payload = json.loads(candidates[-1].read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Patient-level samples take priority for the BigLunge test phase.
        samples_block = payload.get("patient_level", {}).get("samples") or payload.get("samples") or []
        if not samples_block:
            continue
        cm = np.zeros((3, 3), dtype=np.int64)
        for s in samples_block:
            t = int(s.get("true_label", -1))
            p = int(s.get("pred_label", -1))
            if 0 <= t < 3 and 0 <= p < 3:
                cm[t, p] += 1
        if cm.sum() == 0:
            continue
        cms[m] = cm
        n_patients[m] = int(cm.sum())

    if not cms:
        print(f"[confusion_matrices:{phase}] no inference_probabilities found, skipping.")
        return None

    n = len(cms)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)
    for i, (m, cm) in enumerate(cms.items()):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        # Row-normalize for the heatmap colors (recall per row).
        row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
        norm = np.divide(cm, np.where(row_sum > 0, row_sum, 1.0))
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        for ii in range(3):
            for jj in range(3):
                txt = f"{cm[ii, jj]}"
                color = "white" if norm[ii, jj] > 0.5 else "black"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=10, color=color)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Adeno", "SC", "Sq"])
        ax.set_yticks(range(3)); ax.set_yticklabels(["Adeno", "SC", "Sq"])
        ax.set_xlabel("Predicted")
        if c == 0:
            ax.set_ylabel("True")
        ax.set_title(f"{MODEL_LABELS.get(m, m)} (n={n_patients[m]})", fontsize=10)
    # Hide unused axes
    for j in range(len(cms), rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    fig.suptitle(f"Confusion matrices{title_suffix}", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[confusion_matrices:{phase}] -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# Figure: class distribution per dataset
# ----------------------------------------------------------------------------

def make_class_distribution(
    out_path: str = "figures/fig_class_distribution.pdf",
    biglunge_csv: str = "/home/data/TrainingData/patients_parameters.csv",
    lung_pet_dir: str = "/home/data/Lung-PET-CT-Dx-Clean",
) -> Optional[str]:
    """Patient-count per class per dataset. Reads source paths directly so it
    captures the *unfiltered* distribution before any pipeline-side dropping.
    """
    counts = {"BigLunge": [0, 0, 0], "Lung-PET-CT-Dx": [0, 0, 0]}

    # BigLunge: read CSV, count patients per MorphologicalGroup.
    if os.path.isfile(biglunge_csv):
        try:
            import pandas as pd  # local import — pandas isn't required elsewhere
            df = pd.read_csv(biglunge_csv)
            from sclc.data.loaders import BIGLUNGE_CLASS_MAP
            for _, row in df.iterrows():
                grp = str(row.get("MorphologicalGroup", "")).strip()
                if grp in BIGLUNGE_CLASS_MAP:
                    counts["BigLunge"][BIGLUNGE_CLASS_MAP[grp]] += 1
        except Exception as e:  # noqa: BLE001
            print(f"[class_distribution] BigLunge read failed: {e}")

    # Lung-PET-CT-Dx: count folders by -A / -B / -G letter.
    if os.path.isdir(lung_pet_dir):
        from sclc.data.loaders import CLASS_MAP
        for entry in os.listdir(lung_pet_dir):
            if not os.path.isdir(os.path.join(lung_pet_dir, entry)):
                continue
            matched = [v for k, v in CLASS_MAP.items() if f"-{k}" in entry]
            if len(matched) == 1:
                counts["Lung-PET-CT-Dx"][matched[0]] += 1

    if all(sum(c) == 0 for c in counts.values()):
        print("[class_distribution] no source data accessible, skipping.")
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    datasets = list(counts.keys())
    width = 0.26
    xs = np.arange(len(datasets))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ys = [counts[d][cls_idx] for d in datasets]
        ax.bar(xs + (cls_idx - 1) * width, ys, width=width,
               label=cls_name, color=CLASS_COLORS[cls_name])
        for i, y in enumerate(ys):
            ax.text(xs[i] + (cls_idx - 1) * width, y + max(ys) * 0.01, str(y),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(datasets)
    ax.set_ylabel("Patient count")
    ax.set_title("Class distribution per dataset (all patients, pre-pipeline filtering)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[class_distribution] -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# Figure: silent-drop audit per pipeline (which patients each pipeline lost)
# ----------------------------------------------------------------------------

def make_dropped_patients(
    out_path: str = "figures/fig_dropped_patients.pdf",
    cache_root: str = "~/.cache",
) -> Optional[str]:
    """Read each pipeline's ``dropped_patients.json`` and visualize the
    matched/dropped patient sets per split."""
    cache_root = os.path.expanduser(cache_root)
    sources: Dict[str, str] = {
        "BigLunge 2D": os.path.join(cache_root, "monai_biglunge_2d", "img224_crop96_mp1", "dropped_patients.json"),
        "Lung-PET-CT-Dx 2D": os.path.join(cache_root, "monai_lung_pet_ct_clean_2d", "img224_crop96_mp1", "dropped_patients.json"),
    }
    data_per_pipeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, path in sources.items():
        if not os.path.isfile(path):
            continue
        try:
            data_per_pipeline[name] = json.loads(open(path).read())
        except (json.JSONDecodeError, OSError):
            continue

    if not data_per_pipeline:
        print("[dropped_patients] no inventory JSON found, skipping.")
        return None

    splits = ["train", "val", "test"]
    n_pipelines = len(data_per_pipeline)
    fig, axes = plt.subplots(n_pipelines, 1, figsize=(8, 2.4 * n_pipelines + 0.5),
                             squeeze=False, sharex=False)
    for r_idx, (name, drops) in enumerate(data_per_pipeline.items()):
        ax = axes[r_idx][0]
        kept = [len((drops.get(s) or {}).get("kept_patients") or []) for s in splits]
        no_mask = [len((drops.get(s) or {}).get("dropped_no_mask") or []) for s in splits]
        no_slices = [len((drops.get(s) or {}).get("dropped_no_tumor_slices") or []) for s in splits]
        xs = np.arange(len(splits))
        ax.bar(xs, kept, color="#59A14F", label="Kept")
        ax.bar(xs, no_mask, bottom=kept, color="#F28E2B", label="No tumor mask")
        ax.bar(xs, no_slices, bottom=[k + nm for k, nm in zip(kept, no_mask)],
               color="#E15759", label="No tumor slices ≥ min_pixels")
        ax.set_xticks(xs); ax.set_xticklabels([s.title() for s in splits])
        ax.set_ylabel("Patients")
        ax.set_title(name)
        if r_idx == 0:
            ax.legend(loc="upper right", frameon=False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("Silent-drop audit: patients each pipeline lost during entry expansion",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[dropped_patients] -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# Figure: MIL attention overlay (weak-supervision contribution)
# ----------------------------------------------------------------------------

def make_mil_attention(
    out_path: str = "figures/fig_mil_attention.pdf",
    checkpoint_path: str = "",
    biglunge_data_path: str = "/home/data/TrainingData",
    biglunge_csv: str = "/home/data/TrainingData/patients_parameters.csv",
    img_size: int = 384,
    bag_size: int = 16,
    n_patients_to_plot: int = 10,
) -> Optional[str]:
    """Run the MIL model on BigLunge val bags and plot per-instance attention
    weights overlaid against tumor-mask z-extent (where a tumor mask exists).

    This is the weak-supervision contribution figure: "MIL attention
    concentrates on tumor-containing slices X% of the time without explicit
    supervision." Only runs if checkpoint exists.
    """
    if not checkpoint_path:
        # Auto-locate the most recent MIL fine-tune raw checkpoint.
        candidates = sorted(Path("/home/data/trained_models/mil/mil_resnet50").glob("*finetune_pbest_raw.pth"))
        if not candidates:
            print("[mil_attention] no MIL checkpoint found, skipping.")
            return None
        checkpoint_path = str(candidates[-1])

    if not os.path.isfile(checkpoint_path):
        print(f"[mil_attention] checkpoint not found: {checkpoint_path}")
        return None

    try:
        import torch  # local import — heavy
        from torch.utils.data import DataLoader
        from sclc.data.dataset_mil import create_dataset_mil_bag
        from sclc.training.train_mil import simple_collate_fn_mil
        from sclc.models import MILResNet50Classifier
    except ImportError as e:
        print(f"[mil_attention] missing dependency: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MILResNet50Classifier(num_classes=3, mil_mode="att").to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    _, val_ds, _ = create_dataset_mil_bag(
        data_path=biglunge_data_path, csv_path=biglunge_csv,
        dataset_type="big_lunge", img_size=img_size, bag_size=bag_size,
        testing=False, cache_workers=4,
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=simple_collate_fn_mil, num_workers=2)

    samples = []
    for i, batch in enumerate(loader):
        if i >= n_patients_to_plot:
            break
        x, y, meta = batch
        x = x.to(device)
        attn = model.attention_weights(x).cpu().numpy()[0]  # (N,)
        samples.append({
            "patient_id": meta[0].get("patient_id"),
            "label": int(y.item()),
            "attn": attn,
        })

    if not samples:
        print("[mil_attention] no samples produced.")
        return None

    n = len(samples)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 1.6 * rows + 0.5),
                             squeeze=False, sharex=True)
    for i, s in enumerate(samples):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        N = len(s["attn"])
        xs = np.arange(N)
        ax.bar(xs, s["attn"], color=CLASS_COLORS[CLASS_NAMES[s["label"]]],
               width=0.8, alpha=0.85)
        ax.set_xlabel("Slice index in bag (z↑)")
        ax.set_ylabel("Attention")
        ax.set_title(f"{s['patient_id']} ({CLASS_NAMES[s['label']]})", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    fig.suptitle("MIL attention weights per BigLunge val patient "
                 "(no per-slice supervision)", fontsize=11, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[mil_attention] -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

ALL_PLOTS: Dict[str, Any] = {
    "class_distribution": make_class_distribution,
    "dropped_patients": make_dropped_patients,
    "training_curves": make_training_curves,
    "dapt_test_gap": make_dapt_test_gap,
    "per_class_f1": make_per_class_f1,
    "per_class_f1_dapt": lambda **k: make_per_class_f1(
        phase="dapt_test",
        out_path=k.pop("out_path", "figures/fig_per_class_f1_dapt.pdf"),
        title_suffix=" (Lung-PET-CT-Dx test)",
        **k,
    ),
    "confusion_matrices": make_confusion_matrices,
    "confusion_matrices_dapt": lambda **k: make_confusion_matrices(
        phase="dapt_test",
        out_path=k.pop("out_path", "figures/fig_confusion_matrices_dapt.pdf"),
        title_suffix=" (Lung-PET-CT-Dx test)",
        **k,
    ),
    "mil_attention": make_mil_attention,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--only", nargs="*", default=None,
                        help=f"Subset of plots to run; defaults to all. "
                             f"Available: {sorted(ALL_PLOTS.keys())}")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.figures_dir, exist_ok=True)
    targets = args.only or list(ALL_PLOTS.keys())

    for name in targets:
        fn = ALL_PLOTS.get(name)
        if fn is None:
            print(f"[main] unknown plot: {name}; skip.")
            continue
        try:
            fn(output_root=args.output_root) if "output_root" in fn.__code__.co_varnames else fn()
        except TypeError:
            # Fallback for the lambda-wrapped variants that don't take output_root.
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"[main] {name} failed: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[main] {name} failed: {e}")


if __name__ == "__main__":
    main()
