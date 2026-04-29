"""Consolidate test-set metrics from all model `metrics.jsonl` files into one
results table.

Reads the most recent `phase="dapt_test"` and `phase="test"` rows from each
output dir, pulls point estimates + 95% bootstrap CIs (already computed by
validate_epoch_*), and emits a single CSV that can be dropped into the thesis
table verbatim.

Usage::

    python scripts/report_test_metrics.py \
        --output-root output \
        --csv-out runs/results_table.csv

The script does NOT run the model — it only consolidates already-emitted
metrics. To produce test rows, run training with `--mode full` (which calls
_run_test_inference at end of DAPT and end of FT) or `--mode inference`
against an existing checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[report] WARNING: skipping malformed row in {path}: {e}", file=sys.stderr)
    return rows


def _latest_test_row(rows: List[Dict[str, Any]], phase: str) -> Optional[Dict[str, Any]]:
    """Return the most recent row with the given phase tag, or None."""
    matching = [r for r in rows if r.get("phase") == phase]
    if not matching:
        return None
    matching.sort(key=lambda r: r.get("timestamp", ""))
    return matching[-1]


def _row_summary(row: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Flatten one test row into the headline columns."""
    p = row.get("test_patient") or {}
    ci_mf1 = p.get("macro_f1_ci95") or [None, None]
    ci_bacc = p.get("balanced_accuracy_ci95") or [None, None]
    per_cls_f1 = p.get("per_class_f1") or [None, None, None]
    per_cls_f1_ci = p.get("per_class_f1_ci95") or [[None, None]] * 3
    return {
        "model_type": row.get("model_type"),
        "dataset": dataset,
        "phase": row.get("phase"),
        "n_patients": p.get("num_patients"),
        "macro_f1": p.get("macro_f1", row.get("test_macro_f1")),
        "macro_f1_ci_lo": ci_mf1[0],
        "macro_f1_ci_hi": ci_mf1[1],
        "balanced_acc": p.get("balanced_accuracy", row.get("test_balanced_accuracy")),
        "balanced_acc_ci_lo": ci_bacc[0],
        "balanced_acc_ci_hi": ci_bacc[1],
        "f1_adeno": per_cls_f1[0] if len(per_cls_f1) > 0 else None,
        "f1_smallcell": per_cls_f1[1] if len(per_cls_f1) > 1 else None,
        "f1_squamous": per_cls_f1[2] if len(per_cls_f1) > 2 else None,
        "f1_adeno_ci": per_cls_f1_ci[0] if len(per_cls_f1_ci) > 0 else None,
        "f1_smallcell_ci": per_cls_f1_ci[1] if len(per_cls_f1_ci) > 1 else None,
        "f1_squamous_ci": per_cls_f1_ci[2] if len(per_cls_f1_ci) > 2 else None,
        "ci_n_boot": p.get("ci_n_boot"),
        "timestamp": row.get("timestamp"),
    }


def _format_with_ci(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> str:
    if value is None:
        return ""
    if lo is None or hi is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", default="results/output",
        help="Root of the per-model output tree ({pipeline}/{model_type}/metrics.jsonl).",
    )
    parser.add_argument(
        "--csv-out", default="runs/results_table.csv",
        help="Destination CSV for the consolidated results table.",
    )
    parser.add_argument(
        "--md-out", default="runs/results_table.md",
        help="Destination markdown file with a thesis-ready table.",
    )
    args = parser.parse_args()

    rows_out: List[Dict[str, Any]] = []
    pipeline_dirs = ["2d", "3d", "mil"]
    for pipeline in pipeline_dirs:
        pipeline_root = os.path.join(args.output_root, pipeline)
        if not os.path.isdir(pipeline_root):
            continue
        for model_type in sorted(os.listdir(pipeline_root)):
            metrics_path = os.path.join(pipeline_root, model_type, "metrics.jsonl")
            if not os.path.isfile(metrics_path):
                continue
            rows = _load_jsonl(metrics_path)
            for phase, dataset_label in [("dapt_test", "Lung-PET-CT-Dx (test)"),
                                         ("test", "BigLunge (test)")]:
                latest = _latest_test_row(rows, phase)
                if latest is None:
                    continue
                rows_out.append(_row_summary(latest, dataset=dataset_label))

    if not rows_out:
        print("[report] No test rows found. Run --mode full or --mode inference first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    print(f"[report] wrote CSV -> {args.csv_out}  ({len(rows_out)} rows)")

    md_lines: List[str] = []
    md_lines.append("# Test-set results (point estimate, 95% bootstrap CI)\n")
    md_lines.append("")
    md_lines.append("| Model | Dataset | n | MacroF1 | Balanced Acc | F1 (Adeno / SmallCell / Squamous) |")
    md_lines.append("|---|---|---|---|---|---|")
    for r in rows_out:
        per_cls = " / ".join(
            _format_with_ci(r.get(k), (r.get(f"{k}_ci") or [None, None])[0], (r.get(f"{k}_ci") or [None, None])[1])
            for k in ("f1_adeno", "f1_smallcell", "f1_squamous")
        )
        md_lines.append(
            f"| {r['model_type']} | {r['dataset']} | {r['n_patients']} | "
            f"{_format_with_ci(r['macro_f1'], r['macro_f1_ci_lo'], r['macro_f1_ci_hi'])} | "
            f"{_format_with_ci(r['balanced_acc'], r['balanced_acc_ci_lo'], r['balanced_acc_ci_hi'])} | "
            f"{per_cls} |"
        )
    md_lines.append("")
    md_lines.append("CIs are stratified bootstrap (n_boot=1000) on the patient-level "
                    "predictions, computed per-epoch by validate_epoch_*.")
    os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
    with open(args.md_out, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[report] wrote MD  -> {args.md_out}")


if __name__ == "__main__":
    main()
