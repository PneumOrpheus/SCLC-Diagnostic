"""PyRadiomics feature extraction for the SCLC pipeline.

For each patient in ``results/splits.json``:

  1. Load raw HU CT and tumor mask (largest-CC selected, see ``_largest_cc``).
  2. Resample both to 1 mm isotropic (radiomics convention; shape features
     require it).
  3. Run PyRadiomics with IBSI-compliant defaults: shape3D + firstorder +
     GLCM + GLRLM + GLSZM + GLDM + NGTDM, bin width 25 HU.
  4. Concatenate per-patient feature rows into a CSV.

LPCT-Dx multi-scan: per the loader's ``_z_then_name`` rule, take the series
with the smallest Z-spacing. No averaging.

BigLunge: ``<pid>_label_tc.nii.gz`` (already exclusion-filtered).

CLI:
    python -m sclc.radiomics.extract --dataset {lpcd, biglunge}
                                     [--patients PID1 PID2 ...]
                                     [--n-jobs N]
                                     [--mask-perturbation {none, dilate, erode}]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from radiomics import featureextractor

# PyRadiomics is verbose by default; mute it for a clean console.
logging.getLogger("radiomics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_PATH = REPO_ROOT / "results" / "splits.json"
OUT_DIR = REPO_ROOT / "results" / "radiomics"

LPCD_DATA_ROOT = Path("/home/data/Lung-PET-CT-Dx-Clean")
BIGLUNGE_DATA_ROOT = Path("/home/data/TrainingData")

EXTRACTOR_SETTINGS: Dict[str, Any] = {
    # Binning + resampling.
    "binWidth": 25,
    "resampledPixelSpacing": [1.0, 1.0, 1.0],
    "interpolator": sitk.sitkBSpline,
    "force2D": False,
    "normalize": False,
    # Cropping behavior. preCrop=True crops the image+mask to the mask bbox
    # plus padDistance before resampling. Big speedup; same features.
    "preCrop": True,
    "padDistance": 10,
    # Tolerate small geometric mismatches (rare on these masks but catches
    # off-by-one float spacing differences).
    "geometryTolerance": 1e-4,
}

ENABLED_FEATURE_CLASSES: Tuple[str, ...] = (
    "shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm",
)


# ---------- mask helpers -----------------------------------------------------

def _largest_cc(mask_arr: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    """Keep the largest connected component (≥ min_voxels) of a binary mask.

    Mirrors the ``largest_cc_min50`` rule in
    ``sclc/data/transforms.py:CropAroundTumord``. If no component meets the
    threshold, falls back to all non-zero voxels (consistent with the CNN
    pipeline's fallback path).
    """
    binary = mask_arr > 0.5
    if not binary.any():
        return binary.astype(np.uint8)

    from scipy.ndimage import label as cc_label
    labeled, n = cc_label(binary)
    if n == 0:
        return binary.astype(np.uint8)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    valid = sizes >= min_voxels
    if not valid.any():
        return binary.astype(np.uint8)
    masked_sizes = sizes.copy()
    masked_sizes[~valid] = 0
    largest_label = int(masked_sizes.argmax())
    return (labeled == largest_label).astype(np.uint8)


def _perturb_mask(mask_arr: np.ndarray, kind: str) -> np.ndarray:
    """Apply 1-voxel 3D ball dilation or erosion. ``kind`` in {dilate, erode}.

    Used by the Phase-2 stability filter to test feature ICC under boundary
    perturbation.
    """
    if kind == "none":
        return mask_arr
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
    struct = generate_binary_structure(3, 1)  # 6-connectivity 3D ball
    if kind == "dilate":
        return binary_dilation(mask_arr > 0.5, structure=struct).astype(np.uint8)
    if kind == "erode":
        return binary_erosion(mask_arr > 0.5, structure=struct).astype(np.uint8)
    raise ValueError(f"Unknown perturbation kind: {kind}")


# ---------- patient -> (image_path, mask_path) -------------------------------

def _z_spacing(p: Path) -> Tuple[float, str]:
    try:
        zooms = nib.load(str(p), mmap=False).header.get_zooms()
        z = float(zooms[2]) if len(zooms) >= 3 else float("inf")
    except Exception:
        z = float("inf")
    return z, p.name


def _resolve_paths(patient_id: str, dataset: str) -> Tuple[Optional[Path], Optional[Path]]:
    if dataset == "lpcd":
        pdir = LPCD_DATA_ROOT / patient_id
        if not pdir.is_dir():
            return None, None
        # Pick thinnest-Z series; mirrors loaders.py:_z_then_name.
        images = sorted(pdir.glob("*_image.nii.gz"))
        if not images:
            return None, None
        img = min(images, key=_z_spacing)
        msk = img.with_name(img.name.replace("_image.nii.gz", "_mask.nii.gz"))
        return (img, msk if msk.is_file() else None)
    if dataset == "biglunge":
        pdir = BIGLUNGE_DATA_ROOT / patient_id
        img = pdir / f"{patient_id}_input.nii.gz"
        msk = pdir / f"{patient_id}_label_tc.nii.gz"
        return (img if img.is_file() else None, msk if msk.is_file() else None)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------- single-patient extraction ----------------------------------------

def _make_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    ex = featureextractor.RadiomicsFeatureExtractor(**EXTRACTOR_SETTINGS)
    ex.disableAllFeatures()
    for c in ENABLED_FEATURE_CLASSES:
        ex.enableFeatureClassByName(c)
    return ex


def _extract_one(
    patient_id: str,
    image_path: Path,
    mask_path: Path,
    perturbation: str = "none",
    min_cc_voxels: int = 50,
) -> Tuple[str, Dict[str, float], Optional[str]]:
    """Run PyRadiomics on (image, largest-CC mask). Returns
    ``(patient_id, features_dict, error_or_None)``.

    The largest-CC mask is computed in nibabel space, then written to a
    SimpleITK image with the original NIfTI affine so PyRadiomics resamples
    consistently with the CT.
    """
    try:
        # Load via nibabel to compute largest CC + apply perturbation.
        nii_img = nib.load(str(image_path), mmap=False)
        nii_msk = nib.load(str(mask_path), mmap=False)
        msk_arr = nii_msk.get_fdata().astype(np.uint8)
        msk_arr = _largest_cc(msk_arr, min_voxels=min_cc_voxels)
        if perturbation != "none":
            msk_arr = _perturb_mask(msk_arr, perturbation)
        if msk_arr.sum() == 0:
            return patient_id, {}, "empty_mask_after_processing"

        # Convert both to SimpleITK for PyRadiomics. Match origin/spacing/direction
        # via SetOrigin+SetSpacing rather than re-reading from disk; the mask
        # is now a derived array.
        img_sitk = sitk.ReadImage(str(image_path))
        msk_sitk = sitk.GetImageFromArray(msk_arr.transpose(2, 1, 0))  # ZYX in sitk
        msk_sitk.SetOrigin(img_sitk.GetOrigin())
        msk_sitk.SetSpacing(img_sitk.GetSpacing())
        msk_sitk.SetDirection(img_sitk.GetDirection())

        ex = _make_extractor()
        feats = ex.execute(img_sitk, msk_sitk)
        # Drop diagnostics_* keys; keep numeric radiomics only.
        out: Dict[str, float] = {}
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        return patient_id, out, None
    except Exception as exc:
        return patient_id, {}, f"{type(exc).__name__}: {exc}"


# ---------- dataset-level driver ---------------------------------------------

def _load_split_patients(dataset: str) -> List[Dict[str, Any]]:
    """Return [{patient_id, class_idx, class_name, split}] from splits.json."""
    if not SPLITS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {SPLITS_PATH}. Run scripts/dump_splits.py first."
        )
    with open(SPLITS_PATH) as f:
        all_splits = json.load(f)
    block = all_splits["lung_pet_ct_dx" if dataset == "lpcd" else "biglunge"]
    rows: List[Dict[str, Any]] = []
    for split, entries in block.items():
        for e in entries:
            rows.append({
                "patient_id": e["patient_id"],
                "class_idx": int(e["class_idx"]),
                "class_name": e["class_name"],
                "split": split,
            })
    return rows


def extract_dataset(
    dataset: str,
    perturbation: str = "none",
    n_jobs: int = 8,
    patient_ids: Optional[List[str]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Extract features for every patient in the dataset (or a subset).

    Writes a CSV with columns ``patient_id, class_idx, class_name, split,
    <feature_1>, <feature_2>, ...``. Patients with empty masks or extraction
    errors are logged to ``<csv>.audit.json``.
    """
    rows = _load_split_patients(dataset)
    if patient_ids is not None:
        keep = set(patient_ids)
        rows = [r for r in rows if r["patient_id"] in keep]

    print(f"[extract] dataset={dataset} perturbation={perturbation} n_patients={len(rows)} n_jobs={n_jobs}")

    # Resolve paths up-front so we know which patients we'll skip.
    tasks: List[Tuple[Dict[str, Any], Path, Path]] = []
    skipped: List[Dict[str, Any]] = []
    for r in rows:
        img, msk = _resolve_paths(r["patient_id"], dataset)
        if img is None or msk is None:
            skipped.append({"patient_id": r["patient_id"], "reason": "image_or_mask_missing"})
            continue
        tasks.append((r, img, msk))
    print(f"[extract] resolvable: {len(tasks)} | skipped (missing files): {len(skipped)}")

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_extract_one)(r["patient_id"], img, msk, perturbation)
        for (r, img, msk) in tasks
    )

    feature_rows: List[OrderedDict] = []
    failures: List[Dict[str, Any]] = []
    for (r, _img, _msk), (pid, feats, err) in zip(tasks, results):
        if err or not feats:
            failures.append({"patient_id": pid, "reason": err or "no_features"})
            continue
        row: "OrderedDict[str, Any]" = OrderedDict()
        row["patient_id"] = r["patient_id"]
        row["class_idx"] = r["class_idx"]
        row["class_name"] = r["class_name"]
        row["split"] = r["split"]
        for k, v in feats.items():
            row[k] = v
        feature_rows.append(row)

    elapsed = time.time() - t0
    print(f"[extract] done in {elapsed/60:.1f} min "
          f"({len(feature_rows)} ok, {len(failures)} failed, {len(skipped)} skipped)")

    if not feature_rows:
        raise RuntimeError("No features extracted; check failures + skips above.")

    df = pd.DataFrame(feature_rows)
    if out_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "" if perturbation == "none" else f"_{perturbation}"
        out_path = OUT_DIR / f"features_{dataset}{suffix}.csv"
    df.to_csv(out_path, index=False)
    audit_path = out_path.with_suffix(".audit.json")
    with open(audit_path, "w") as f:
        json.dump({
            "dataset": dataset,
            "perturbation": perturbation,
            "n_requested": len(rows),
            "n_extracted": len(feature_rows),
            "n_failed": len(failures),
            "n_skipped": len(skipped),
            "failures": failures,
            "skipped": skipped,
            "extractor_settings": {k: (v if not callable(v) else str(v))
                                    for k, v in EXTRACTOR_SETTINGS.items()},
            "feature_classes": list(ENABLED_FEATURE_CLASSES),
        }, f, indent=2, default=str)
    print(f"[extract] wrote {out_path} ({len(df.columns) - 4} feature columns)")
    print(f"[extract] audit: {audit_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["lpcd", "biglunge"])
    p.add_argument("--perturbation", default="none", choices=["none", "dilate", "erode"])
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--patients", nargs="*", default=None,
                   help="Optional explicit patient IDs (default: all in splits.json).")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    extract_dataset(
        dataset=args.dataset,
        perturbation=args.perturbation,
        n_jobs=args.n_jobs,
        patient_ids=args.patients,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
