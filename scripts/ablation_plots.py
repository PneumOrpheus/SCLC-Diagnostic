"""Ablation plots: baseline (no FPN) vs FPN+det/seg for the non-RadImageNet models.

Reads ``metrics.jsonl`` files from two output trees (baseline and FPN) and
generates paired-comparison figures suitable for the thesis ablation section.

Figures produced
----------------
fig_ablation_macro_f1.pdf
    Two-panel figure (BigLunge test / DAPT test). Each model has a gray bar
    (baseline) and a coloured bar (FPN), with 95 % CI error bars.

fig_ablation_delta.pdf
    Horizontal delta bars: FPN MacroF1 − baseline MacroF1 per model,
    BigLunge test.  Green = improvement, red = regression.

fig_ablation_per_class.pdf
    3-column grid (one column per class). Each column is a paired bar chart
    over models showing baseline vs FPN per-class F1 (BigLunge test).

Usage::

    python scripts/ablation_plots.py \\
        --baseline-root results/output_ablation_base \\
        --fpn-root      results/output_ablation_fpn  \\
        --figures-dir   results/thesis_ablation/figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_MODELS: List[str] = [
    "efficientnet_b0_2d",
    "resnet50_2d",
    "densenet121_2d",
    "swin_tiny_2d",
    "mil_resnet50",
    "mil_swin_tiny",
    "swin_unetr",
]

PIPELINE_OF: Dict[str, str] = {
    "efficientnet_b0_2d": "2d",
    "resnet50_2d":        "2d",
    "densenet121_2d":     "2d",
    "swin_tiny_2d":       "2d",
    "mil_resnet50":       "mil",
    "mil_swin_tiny":      "mil",
    "swin_unetr":         "3d",
}

MODEL_LABELS: Dict[str, str] = {
    "efficientnet_b0_2d": "EffNet-B0\n(2D)",
    "resnet50_2d":        "ResNet-50\n(2D)",
    "densenet121_2d":     "DenseNet121\n(2D)",
    "swin_tiny_2d":       "Swin-Tiny\n(2D)",
    "mil_resnet50":       "MIL\nResNet-50",
    "mil_swin_tiny":      "MIL\nSwin-Tiny",
    "swin_unetr":         "SwinUNETR\n(3D)",
}

CLASS_NAMES: List[str] = ["Adenocarcinoma", "Small Cell", "Squamous"]
CLASS_COLORS: Dict[str, str] = {
    "Adenocarcinoma": "#4E79A7",
    "Small Cell":     "#F28E2B",
    "Squamous":       "#59A14F",
}

BASELINE_COLOR = "#AAAAAA"
FPN_COLOR      = "#2196F3"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.format":    "pdf",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
    })


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _metrics_path(model_type: str, output_root: str) -> Path:
    pipeline = PIPELINE_OF.get(model_type, "2d")
    return Path(output_root) / pipeline / model_type / "metrics.jsonl"


def _load_rows(model_type: str, output_root: str) -> List[Dict[str, Any]]:
    p = _metrics_path(model_type, output_root)
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_row(rows: List[Dict[str, Any]], phase: str) -> Optional[Dict[str, Any]]:
    matching = [r for r in rows if r.get("phase") == phase]
    if not matching:
        return None
    matching.sort(key=lambda r: r.get("timestamp", ""))
    return matching[-1]


def _patient_block(row: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """Return the patient-level sub-dict for inference rows (test / dapt_test)."""
    key = "test_patient" if phase in ("test", "dapt_test") else "val_patient"
    return row.get(key) or {}


def _extract_macro_f1(
    model_type: str, output_root: str, phase: str
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    rows = _load_rows(model_type, output_root)
    row  = _latest_row(rows, phase)
    if row is None:
        return None, None
    pb = _patient_block(row, phase)
    f1 = pb.get("macro_f1")
    ci = pb.get("macro_f1_ci95")
    if f1 is None:
        return None, None
    ci_tuple = (float(ci[0]), float(ci[1])) if ci and len(ci) == 2 else (float(f1), float(f1))
    return float(f1), ci_tuple


def _extract_per_class_f1(
    model_type: str, output_root: str, phase: str
) -> Tuple[Optional[List[float]], Optional[List[Tuple[float, float]]]]:
    rows = _load_rows(model_type, output_root)
    row  = _latest_row(rows, phase)
    if row is None:
        return None, None
    pb     = _patient_block(row, phase)
    vals   = pb.get("per_class_f1") or []
    cis    = pb.get("per_class_f1_ci95") or []
    if len(vals) < 3:
        return None, None
    ci_list = []
    for c in range(3):
        ci = cis[c] if c < len(cis) else (vals[c], vals[c])
        ci_list.append((float(ci[0]), float(ci[1])))
    return [float(v) for v in vals[:3]], ci_list


# ---------------------------------------------------------------------------
# Figure 1 — paired MacroF1 comparison (baseline vs FPN)
# ---------------------------------------------------------------------------

def make_ablation_macro_f1(
    baseline_root: str,
    fpn_root: str,
    figures_dir: str,
    models: Optional[List[str]] = None,
) -> Optional[str]:
    """Two-panel paired bar chart: baseline vs FPN MacroF1 on BL-test and DAPT-test."""
    models = models or ABLATION_MODELS
    out_path = os.path.join(figures_dir, "fig_ablation_macro_f1.pdf")

    phases      = [("test",      "BigLunge test"),
                   ("dapt_test", "DAPT test (Lung-PET-CT-Dx)")]
    n_phases    = len(phases)
    fig, axes   = plt.subplots(1, n_phases, figsize=(5 * n_phases + 1, 4.5), sharey=True)
    if n_phases == 1:
        axes = [axes]

    bar_w  = 0.35
    plotted_any = False

    for ax, (phase, phase_label) in zip(axes, phases):
        xs       = []
        base_ys  = []
        base_lo  = []
        base_hi  = []
        fpn_ys   = []
        fpn_lo   = []
        fpn_hi   = []
        labels   = []

        for i, m in enumerate(models):
            b_f1, b_ci = _extract_macro_f1(m, baseline_root, phase)
            f_f1, f_ci = _extract_macro_f1(m, fpn_root, phase)
            if b_f1 is None and f_f1 is None:
                continue
            xs.append(len(labels))
            labels.append(MODEL_LABELS.get(m, m))
            b_f1 = b_f1 or 0.0
            f_f1 = f_f1 or 0.0
            b_ci = b_ci or (b_f1, b_f1)
            f_ci = f_ci or (f_f1, f_f1)
            base_ys.append(b_f1)
            base_lo.append(b_f1 - b_ci[0])
            base_hi.append(b_ci[1] - b_f1)
            fpn_ys.append(f_f1)
            fpn_lo.append(f_f1 - f_ci[0])
            fpn_hi.append(f_ci[1] - f_f1)

        if not xs:
            ax.set_visible(False)
            continue

        plotted_any = True
        xs = np.array(xs, dtype=float)
        ax.bar(xs - bar_w / 2, base_ys, bar_w, label="Baseline",
               color=BASELINE_COLOR, edgecolor="white",
               yerr=[base_lo, base_hi], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})
        ax.bar(xs + bar_w / 2, fpn_ys, bar_w, label="FPN + det/seg",
               color=FPN_COLOR, edgecolor="white",
               yerr=[fpn_lo, fpn_hi], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, ha="center")
        ax.set_ylabel("Patient MacroF1 (95 % CI)")
        ax.set_ylim(0, 1)
        ax.set_title(f"MacroF1 — {phase_label}")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(loc="lower right", frameon=False)

    if not plotted_any:
        print(f"[ablation_macro_f1] no data found in either root, skipping.")
        plt.close(fig)
        return None

    os.makedirs(figures_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ablation_macro_f1] -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — delta chart (FPN − baseline)
# ---------------------------------------------------------------------------

def make_ablation_delta(
    baseline_root: str,
    fpn_root: str,
    figures_dir: str,
    models: Optional[List[str]] = None,
    phase: str = "test",
    phase_label: str = "BigLunge test",
) -> Optional[str]:
    """Horizontal bar chart: FPN MacroF1 − baseline MacroF1 per model."""
    models = models or ABLATION_MODELS
    out_path = os.path.join(figures_dir, "fig_ablation_delta.pdf")

    deltas: List[Tuple[str, float]] = []
    for m in models:
        b_f1, _ = _extract_macro_f1(m, baseline_root, phase)
        f_f1, _ = _extract_macro_f1(m, fpn_root, phase)
        if b_f1 is None or f_f1 is None:
            continue
        deltas.append((MODEL_LABELS.get(m, m), f_f1 - b_f1))

    if not deltas:
        print("[ablation_delta] no data found, skipping.")
        return None

    deltas.sort(key=lambda x: x[1])
    labels = [d[0] for d in deltas]
    values = [d[1] for d in deltas]
    colors = [("#2e7d32" if v >= 0 else "#c62828") for v in values]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.55 * len(deltas) + 1.5)))
    ys = np.arange(len(deltas))
    ax.barh(ys, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ΔMacroF1  (FPN − Baseline)")
    ax.set_title(f"FPN improvement over baseline — {phase_label}")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    green_patch = mpatches.Patch(color="#2e7d32", label="Improvement")
    red_patch   = mpatches.Patch(color="#c62828", label="Regression")
    ax.legend(handles=[green_patch, red_patch], loc="lower right", frameon=False)

    os.makedirs(figures_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ablation_delta] -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 — per-class F1 comparison
# ---------------------------------------------------------------------------

def make_ablation_per_class(
    baseline_root: str,
    fpn_root: str,
    figures_dir: str,
    models: Optional[List[str]] = None,
    phase: str = "test",
    phase_label: str = "BigLunge test",
) -> Optional[str]:
    """3-column grid: per-class F1, baseline vs FPN, one column per class."""
    models = models or ABLATION_MODELS
    out_path = os.path.join(figures_dir, "fig_ablation_per_class.pdf")

    # Collect data per model
    model_labels: List[str] = []
    base_vals:  List[List[float]] = [[], [], []]
    base_lo:    List[List[float]] = [[], [], []]
    base_hi:    List[List[float]] = [[], [], []]
    fpn_vals:   List[List[float]] = [[], [], []]
    fpn_lo:     List[List[float]] = [[], [], []]
    fpn_hi:     List[List[float]] = [[], [], []]

    for m in models:
        b_pcf1, b_pcf1_ci = _extract_per_class_f1(m, baseline_root, phase)
        f_pcf1, f_pcf1_ci = _extract_per_class_f1(m, fpn_root, phase)
        if b_pcf1 is None and f_pcf1 is None:
            continue
        model_labels.append(MODEL_LABELS.get(m, m))
        b_pcf1    = b_pcf1    or [0.0, 0.0, 0.0]
        f_pcf1    = f_pcf1    or [0.0, 0.0, 0.0]
        b_pcf1_ci = b_pcf1_ci or [(v, v) for v in b_pcf1]
        f_pcf1_ci = f_pcf1_ci or [(v, v) for v in f_pcf1]
        for c in range(3):
            base_vals[c].append(b_pcf1[c])
            base_lo[c].append(b_pcf1[c]   - b_pcf1_ci[c][0])
            base_hi[c].append(b_pcf1_ci[c][1] - b_pcf1[c])
            fpn_vals[c].append(f_pcf1[c])
            fpn_lo[c].append(f_pcf1[c]    - f_pcf1_ci[c][0])
            fpn_hi[c].append(f_pcf1_ci[c][1] - f_pcf1[c])

    if not model_labels:
        print("[ablation_per_class] no data found, skipping.")
        return None

    n = len(model_labels)
    bar_w = 0.35
    xs    = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 4.5), sharey=True)
    for c, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        cls_color = CLASS_COLORS[cls_name]
        ax.bar(xs - bar_w / 2, base_vals[c], bar_w, label="Baseline",
               color=BASELINE_COLOR, edgecolor="white",
               yerr=[base_lo[c], base_hi[c]], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})
        ax.bar(xs + bar_w / 2, fpn_vals[c], bar_w, label="FPN + det/seg",
               color=cls_color, edgecolor="white",
               yerr=[fpn_lo[c], fpn_hi[c]], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.7})
        ax.set_xticks(xs)
        ax.set_xticklabels(model_labels, ha="center")
        ax.set_ylabel("Per-class F1 (95 % CI)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{cls_name} — {phase_label}")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(loc="lower right", frameon=False)

    os.makedirs(figures_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ablation_per_class] -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CSV summary table
# ---------------------------------------------------------------------------

def make_ablation_csv(
    baseline_root: str,
    fpn_root: str,
    figures_dir: str,
    models: Optional[List[str]] = None,
    phase: str = "test",
) -> Optional[str]:
    """Long-format CSV: model, variant, macro_f1, ci_lo, ci_hi, per_class_f1_*."""
    import csv
    models   = models or ABLATION_MODELS
    out_path = os.path.join(figures_dir, "ablation_summary.csv")
    rows_out = []
    for m in models:
        for variant, root in [("baseline", baseline_root), ("fpn", fpn_root)]:
            f1, ci   = _extract_macro_f1(m, root, phase)
            pcf1, _  = _extract_per_class_f1(m, root, phase)
            if f1 is None:
                continue
            row: Dict[str, Any] = {
                "model":        m,
                "variant":      variant,
                "phase":        phase,
                "macro_f1":     round(f1, 4),
                "ci_lo":        round(ci[0], 4) if ci else "",
                "ci_hi":        round(ci[1], 4) if ci else "",
            }
            if pcf1:
                for c, cls in enumerate(CLASS_NAMES):
                    row[f"f1_{cls.lower().replace(' ', '_')}"] = round(pcf1[c], 4)
            rows_out.append(row)

    if not rows_out:
        print("[ablation_csv] no data found, skipping.")
        return None

    os.makedirs(figures_dir, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[ablation_csv] -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline-root", required=True,
                        help="output-root from the baseline (no-FPN) run")
    parser.add_argument("--fpn-root", required=True,
                        help="output-root from the FPN+det/seg run")
    parser.add_argument("--figures-dir", default="results/thesis_ablation/figures",
                        help="directory to write PDFs and CSV into")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of models to include (default: all non-RIN models)")
    parser.add_argument("--phase", default="test",
                        choices=["test", "dapt_test"],
                        help="phase for delta and per-class figures (default: test)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these figure names: macro_f1 delta per_class csv")
    args = parser.parse_args()

    _setup_style()
    models = args.models

    phase_label = "BigLunge test" if args.phase == "test" else "DAPT test"

    figs = {
        "macro_f1":  lambda: make_ablation_macro_f1(
            args.baseline_root, args.fpn_root, args.figures_dir, models),
        "delta":     lambda: make_ablation_delta(
            args.baseline_root, args.fpn_root, args.figures_dir, models,
            args.phase, phase_label),
        "per_class": lambda: make_ablation_per_class(
            args.baseline_root, args.fpn_root, args.figures_dir, models,
            args.phase, phase_label),
        "csv":       lambda: make_ablation_csv(
            args.baseline_root, args.fpn_root, args.figures_dir, models,
            args.phase),
    }

    to_run = args.only if args.only else list(figs.keys())
    for name in to_run:
        if name not in figs:
            print(f"[ablation_plots] unknown figure '{name}', skipping.")
            continue
        figs[name]()


if __name__ == "__main__":
    main()
