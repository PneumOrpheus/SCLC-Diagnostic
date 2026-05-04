"""Phase 2: stability filter for radiomics features.

Drops features that are unstable under ±1-voxel mask perturbation, then drops
near-zero-variance features and one of every highly-correlated pair. Mitigates
the auto-seg-only mask problem (no radiologist correction available).

Pipeline:
  1. Read three feature CSVs per dataset (baseline / dilated / eroded).
  2. Per-feature ICC(3,1) across the 3 versions over patients in the train+val
     split (test is held out for the final fit; can leak test ROI shapes
     otherwise via the ICC stability decision).
  3. Drop features with ICC < 0.75.
  4. Drop near-zero-variance (std < 1e-6 post z-score).
  5. Drop one of every Pearson |r| > 0.9 pair, prefer keeping the higher-ICC.
  6. Write stable_features_<dataset>.csv + feature_audit_<dataset>.json.

ICC(3,1) reference: Shrout & Fleiss 1979, two-way mixed-effects, single rater,
absolute agreement. Standard radiomics test-retest convention; cited in Shah
et al. 2021 \\cite{Shah2021}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RADIOMICS_DIR = REPO_ROOT / "results" / "radiomics"

ICC_CUTOFF = 0.75
NZV_STD_CUTOFF = 1e-6
CORR_CUTOFF = 0.9


def _icc_3_1(matrix: np.ndarray) -> float:
    """ICC(3,1): two-way mixed-effects, single rater, absolute agreement.

    ``matrix``: (n_subjects, n_raters). Returns NaN if degenerate.
    """
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand_mean = matrix.mean()
    ss_total = ((matrix - grand_mean) ** 2).sum()
    ss_between_subjects = k * ((matrix.mean(axis=1) - grand_mean) ** 2).sum()
    ss_between_raters = n * ((matrix.mean(axis=0) - grand_mean) ** 2).sum()
    ss_residual = ss_total - ss_between_subjects - ss_between_raters
    if n - 1 == 0 or (n - 1) * (k - 1) == 0:
        return float("nan")
    ms_between_subjects = ss_between_subjects / (n - 1)
    ms_residual = ss_residual / ((n - 1) * (k - 1))
    if ms_between_subjects + (k - 1) * ms_residual == 0:
        return float("nan")
    icc = (ms_between_subjects - ms_residual) / (
        ms_between_subjects + (k - 1) * ms_residual
    )
    return float(icc)


def _aligned(
    base_df: pd.DataFrame,
    dil_df: pd.DataFrame,
    ero_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Return base/dil/ero aligned to common patients + common feature columns."""
    keys = sorted(
        set(base_df["patient_id"]) & set(dil_df["patient_id"]) & set(ero_df["patient_id"])
    )
    base = base_df[base_df["patient_id"].isin(keys)].sort_values("patient_id").reset_index(drop=True)
    dil = dil_df[dil_df["patient_id"].isin(keys)].sort_values("patient_id").reset_index(drop=True)
    ero = ero_df[ero_df["patient_id"].isin(keys)].sort_values("patient_id").reset_index(drop=True)
    meta_cols = {"patient_id", "class_idx", "class_name", "split"}
    feats = sorted(
        (set(base.columns) & set(dil.columns) & set(ero.columns)) - meta_cols
    )
    return base, dil, ero, feats


def filter_features(
    base_csv: Path,
    dil_csv: Path,
    ero_csv: Path,
    out_csv: Path,
    audit_path: Path,
    use_splits_for_icc: Tuple[str, ...] = ("train", "val"),
) -> Path:
    base = pd.read_csv(base_csv)
    dil = pd.read_csv(dil_csv)
    ero = pd.read_csv(ero_csv)
    base, dil, ero, feats = _aligned(base, dil, ero, )

    # ICC computed on train+val to keep test ROIs out of the stability decision.
    icc_mask = base["split"].isin(use_splits_for_icc).to_numpy()
    if icc_mask.sum() < 2:
        raise RuntimeError(
            f"Too few train/val patients for ICC ({int(icc_mask.sum())}); "
            f"check splits.json + extraction outputs."
        )

    iccs: Dict[str, float] = {}
    for f in feats:
        m = np.column_stack([
            base.loc[icc_mask, f].to_numpy(dtype=float),
            dil.loc[icc_mask, f].to_numpy(dtype=float),
            ero.loc[icc_mask, f].to_numpy(dtype=float),
        ])
        # Replace NaN/inf with column means so a single problem patient
        # doesn't kill the whole feature.
        for c in range(m.shape[1]):
            col = m[:, c]
            finite = np.isfinite(col)
            if finite.sum() == 0:
                m[:, c] = 0.0
                continue
            mean = float(col[finite].mean())
            col[~finite] = mean
            m[:, c] = col
        iccs[f] = _icc_3_1(m)

    # Step 1: ICC filter.
    stable = [f for f, v in iccs.items() if np.isfinite(v) and v >= ICC_CUTOFF]
    dropped_icc = [f for f in feats if f not in stable]

    # Step 2: near-zero-variance on train+val.
    train_val = base[base["split"].isin(use_splits_for_icc)].reset_index(drop=True)
    nzv: List[str] = []
    for f in list(stable):
        x = train_val[f].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0 or x.std() < NZV_STD_CUTOFF:
            nzv.append(f)
            stable.remove(f)

    # Step 3: correlation cluster reduction. Keep the higher-ICC feature in each pair.
    if len(stable) > 1:
        sub = train_val[stable].to_numpy(dtype=float)
        # Pairwise Pearson correlations; mask diagonal and lower triangle.
        # Standardize first; |r|=|covariance| of standardized.
        with np.errstate(invalid="ignore"):
            std = sub.std(axis=0, ddof=1)
            std = np.where(std == 0, 1.0, std)
            z = (sub - sub.mean(axis=0)) / std
            r = np.corrcoef(z, rowvar=False)
        n_f = len(stable)
        dropped_corr: List[Tuple[str, str, float]] = []
        keep = np.ones(n_f, dtype=bool)
        for i in range(n_f):
            if not keep[i]:
                continue
            for j in range(i + 1, n_f):
                if not keep[j]:
                    continue
                if abs(r[i, j]) > CORR_CUTOFF:
                    # Drop the lower-ICC of the pair.
                    fi, fj = stable[i], stable[j]
                    if iccs.get(fi, 0.0) >= iccs.get(fj, 0.0):
                        keep[j] = False
                        dropped_corr.append((fj, fi, float(r[i, j])))
                    else:
                        keep[i] = False
                        dropped_corr.append((fi, fj, float(r[i, j])))
                        break
        stable = [f for f, k in zip(stable, keep) if k]
    else:
        dropped_corr = []

    print(f"[stability] feats={len(feats)}  ok_icc={len(stable) + len(nzv) + len(dropped_corr)}"
          f"  pruned_icc={len(dropped_icc)}  pruned_nzv={len(nzv)}  pruned_corr={len(dropped_corr)}"
          f"  -> kept={len(stable)}")

    # Write outputs: keep base CSV's features for the surviving columns only.
    keep_cols = ["patient_id", "class_idx", "class_name", "split"] + stable
    out_df = base[keep_cols].copy()
    out_df.to_csv(out_csv, index=False)

    audit = {
        "icc_cutoff": ICC_CUTOFF,
        "nzv_std_cutoff": NZV_STD_CUTOFF,
        "corr_cutoff": CORR_CUTOFF,
        "icc_split_used": list(use_splits_for_icc),
        "n_features_in": len(feats),
        "n_kept": len(stable),
        "n_dropped_icc": len(dropped_icc),
        "n_dropped_nzv": len(nzv),
        "n_dropped_corr": len(dropped_corr),
        "kept_features": stable,
        "iccs": {f: iccs.get(f) for f in feats},
        "dropped_icc": dropped_icc,
        "dropped_nzv": nzv,
        "dropped_corr": [{"dropped": d, "in_favor_of": k, "r": r} for (d, k, r) in dropped_corr],
    }
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[stability] wrote {out_csv}")
    print(f"[stability] audit: {audit_path}")
    return out_csv


def run(dataset: str) -> None:
    base = RADIOMICS_DIR / f"features_{dataset}.csv"
    dil = RADIOMICS_DIR / f"features_{dataset}_dilate.csv"
    ero = RADIOMICS_DIR / f"features_{dataset}_erode.csv"
    out = RADIOMICS_DIR / f"stable_features_{dataset}.csv"
    audit = RADIOMICS_DIR / f"feature_audit_{dataset}.json"
    for p in (base, dil, ero):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}; run extract.py first (with --perturbation).")
    filter_features(base, dil, ero, out, audit)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["lpcd", "biglunge"])
    args = p.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
