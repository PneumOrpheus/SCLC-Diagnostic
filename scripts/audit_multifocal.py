"""Count connected components in BigLunge tumor masks per patient.

Produces:
- ``output/multifocal_audit.csv``: one row per patient with class, n_components,
  largest-component-voxels, total tumor voxels, voxel size of each CC.
- Console summary: counts per class, fraction multifocal (>1 large component),
  fraction missing-mask, fraction empty-mask.
- ``figures/fig_multifocal_distribution.pdf``: histogram of components per class.

Run::

    python scripts/audit_multifocal.py \
        --data-root /home/data/TrainingData \
        --csv /home/data/TrainingData/patients_parameters.csv \
        --out-csv output/multifocal_audit.csv \
        --out-fig figures/fig_multifocal_distribution.pdf

A "large" component is one with at least ``--min-component-voxels`` voxels
(default 50). This filters specks/false positives so the multifocal count
reflects actual lesions.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_labels(csv_path: str) -> Dict[str, str]:
    """Patient ID -> MorphologicalGroup, lower-cased and trimmed."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row["Patient"]).strip()
        grp = str(row.get("MorphologicalGroup", "")).strip()
        out[pid] = grp
    return out



def _connected_components(mask: np.ndarray, min_voxels: int) -> Tuple[int, List[int], int]:
    """Return (n_large_cc, sorted_sizes, total_nonzero_voxels)."""
    from scipy.ndimage import label as cc_label
    binary = mask > 0.5
    total = int(binary.sum())
    if total == 0:
        return 0, [], 0
    # 6-connectivity (faces) is the conservative choice for medical masks;
    # 26-connectivity (default) over-merges nearby blobs.
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    labeled, n = cc_label(binary, structure=structure)
    if n == 0:
        return 0, [], total
    sizes = np.bincount(labeled.ravel())[1:]  # drop background bucket
    large = [int(s) for s in sizes if s >= min_voxels]
    large.sort(reverse=True)
    return len(large), large, total


def audit(
    data_root: str,
    csv_path: str,
    min_component_voxels: int = 50,
    tumor_mask_suffix: str = "_label_tc.nii.gz",
) -> List[Dict[str, Any]]:
    import nibabel as nib
    labels = _load_labels(csv_path)
    rows: List[Dict[str, Any]] = []
    root = Path(data_root)
    patient_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for d in patient_dirs:
        pid = d.name
        if pid not in labels:
            continue
        mask_path = d / f"{pid}{tumor_mask_suffix}"
        cls_str = labels[pid]
        if not mask_path.exists():
            rows.append({
                "patient_id": pid,
                "class": labels[pid],
                "mask_present": False,
                "n_components_large": 0,
                "component_sizes": "",
                "largest_component_voxels": 0,
                "total_tumor_voxels": 0,
            })
            continue
        try:
            arr = nib.load(str(mask_path), mmap=False).get_fdata(dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[audit] {pid}: failed to load mask ({e})")
            continue
        n, sizes, total = _connected_components(arr, min_component_voxels)
        rows.append({
            "patient_id": pid,
            "class": labels[pid],
            "mask_present": True,
            "n_components_large": n,
            "component_sizes": ";".join(str(s) for s in sizes),
            "largest_component_voxels": sizes[0] if sizes else 0,
            "total_tumor_voxels": total,
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[audit] wrote CSV -> {out_path} ({len(rows)} rows)")


def print_summary(rows: List[Dict[str, Any]]) -> None:
    classes = ["Adenocarcinoma", "Small Cell", "Squamous", "Other"]
    print("\n=== Multifocal audit summary ===")
    print(f"{'Class':<16} {'N':>4} {'NoMask':>7} {'Empty':>6} "
          f"{'Mono':>5} {'Multi':>6} {'%Multi':>7} "
          f"{'MedComp':>8} {'P95Comp':>8} {'MedSize':>9} {'P95Size':>9}")
    for cls in classes:
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        n = len(sub)
        no_mask = sum(1 for r in sub if not r["mask_present"])
        present = [r for r in sub if r["mask_present"]]
        empty = sum(1 for r in present if r["n_components_large"] == 0
                    and r["total_tumor_voxels"] == 0)
        mono = sum(1 for r in present if r["n_components_large"] == 1)
        multi = sum(1 for r in present if r["n_components_large"] >= 2)
        comp_counts = np.array([r["n_components_large"] for r in present
                                if r["n_components_large"] > 0])
        sizes = np.array([r["largest_component_voxels"] for r in present
                          if r["largest_component_voxels"] > 0])
        med_comp = float(np.median(comp_counts)) if comp_counts.size else 0.0
        p95_comp = float(np.percentile(comp_counts, 95)) if comp_counts.size else 0.0
        med_size = float(np.median(sizes)) if sizes.size else 0.0
        p95_size = float(np.percentile(sizes, 95)) if sizes.size else 0.0
        pct_multi = (100.0 * multi / max(1, mono + multi))
        print(f"{cls:<16} {n:>4} {no_mask:>7} {empty:>6} "
              f"{mono:>5} {multi:>6} {pct_multi:>6.1f}% "
              f"{med_comp:>8.1f} {p95_comp:>8.1f} {med_size:>9.0f} {p95_size:>9.0f}")
    print("\nColumns: NoMask=missing _label_tc.nii.gz; Empty=mask exists but all-zero;")
    print("         Mono=1 large component; Multi>=2; sizes are voxel counts.")


def write_figure(rows: List[Dict[str, Any]], out_path: str) -> None:
    import matplotlib.pyplot as plt
    classes = ["Adenocarcinoma", "Small Cell", "Squamous"]
    colors = {"Adenocarcinoma": "#4E79A7", "Small Cell": "#F28E2B", "Squamous": "#59A14F"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_n, ax_size = axes

    # Histogram: number of components, capped at 8+ for readability.
    bins = np.arange(0.5, 9.5, 1.0)
    for cls in classes:
        cs = [r["n_components_large"] for r in rows
              if r["class"] == cls and r["mask_present"]]
        cs_capped = [min(int(c), 8) for c in cs]
        ax_n.hist(cs_capped, bins=bins, alpha=0.6, label=cls,
                  color=colors[cls], edgecolor="white")
    ax_n.set_xlabel("Connected components (large; size ≥ min_voxels)")
    ax_n.set_ylabel("Patient count")
    ax_n.set_title("Components per patient by class")
    ax_n.set_xticks(np.arange(1, 9))
    ax_n.set_xticklabels([str(i) if i < 8 else "8+" for i in range(1, 9)])
    ax_n.grid(axis="y", alpha=0.3, linestyle=":")
    ax_n.legend(loc="upper right", frameon=False)

    # Distribution: largest-component size in voxels, log scale.
    for cls in classes:
        sizes = [r["largest_component_voxels"] for r in rows
                 if r["class"] == cls and r["largest_component_voxels"] > 0]
        if not sizes:
            continue
        ax_size.hist(np.log10(np.asarray(sizes, dtype=np.float64)),
                     bins=30, alpha=0.6, label=cls, color=colors[cls],
                     edgecolor="white")
    ax_size.set_xlabel("log10(largest component voxels)")
    ax_size.set_ylabel("Patient count")
    ax_size.set_title("Dominant-lesion size by class")
    ax_size.grid(axis="y", alpha=0.3, linestyle=":")
    ax_size.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[audit] wrote figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/data/TrainingData")
    parser.add_argument("--csv", default="/home/data/TrainingData/patients_parameters.csv")
    parser.add_argument("--out-csv", default="output/multifocal_audit.csv")
    parser.add_argument("--out-fig", default="figures/fig_multifocal_distribution.pdf")
    parser.add_argument("--min-component-voxels", type=int, default=50,
                        help="Components smaller than this are treated as noise / not counted "
                             "toward 'multifocal' classification.")
    parser.add_argument("--tumor-mask-suffix", default="_label_tc.nii.gz")
    args = parser.parse_args()

    rows = audit(
        data_root=args.data_root, csv_path=args.csv,
        min_component_voxels=args.min_component_voxels,
        tumor_mask_suffix=args.tumor_mask_suffix,
    )
    if not rows:
        print("[audit] no rows produced.")
        sys.exit(1)
    write_csv(rows, args.out_csv)
    print_summary(rows)
    write_figure(rows, args.out_fig)


if __name__ == "__main__":
    main()
