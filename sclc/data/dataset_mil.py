"""MIL pipeline data builders.

Two dataset kinds, sharing the 1 mm / 2 mm spacing + HU-windowed front-end
defined in ``data/transforms.py``:

1. **Whole-slice DAPT** (``create_dataset_whole_slice``): one sample = one
   tumor-containing axial slice from Lung-PET-CT-Dx. No in-plane tumor crop;
   the backbone sees the full axial FOV at ``img_size`` × ``img_size``. Tumor
   mask is only used to enumerate which slices carry tumor (we reuse the
   ``tumor_slice_index`` cache from the 2D pipeline).

2. **MIL bag** (``create_dataset_mil_bag``): one sample = one patient, output
   shape ``(N, 1, img_size, img_size)``. ``N = bag_size`` slices are sampled
   evenly across the lung mask's axial extent. **No tumor mask is used at
   inference.** The attention head in MILModel decides which instances drive
   the bag-level prediction.

Both paths wrap ``PersistentDataset`` with their own cache directories keyed
on the parameters that affect the cached tensor content.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from monai.data import PersistentDataset
from tqdm import tqdm

from sclc.data.dataset_2d import (
    _testing_subset_balanced,
    get_biglunge_2d_data_list,
    get_lung_pet_ct_dx_2d_data_list,
)
from sclc.data.loaders import (
    get_biglunge_data_list,
)
from sclc.data.exclusions import TRUNCATED_LUNG_MASK
from sclc.data.transforms import (
    get_train_transforms_mil_bag,
    get_train_transforms_whole_slice,
    get_val_transforms_mil_bag,
    get_val_transforms_whole_slice,
)


# -----------------------------------------------------------------------------
# Whole-slice per-slice DAPT (reuses the 2D per-slice data lists)
# -----------------------------------------------------------------------------


def create_dataset_whole_slice(
    data_path: str,
    csv_path: str = "",
    dataset_type: str = "lung_pet_ct_dx",
    img_size: int = 384,
    tumor_mask_suffix: str = "_label_tc.nii.gz",
    max_slices_per_volume: Optional[int] = None,
    min_tumor_pixels: int = 100,
    cache_dir: Optional[str] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    warm_cache: bool = False,
    cache_workers: int = 4,
    strong_augs: bool = False,
    clear_cache: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """Per-slice whole-slice datasets for DAPT.

    Same tumor-slice-index-based entry enumeration as the 2D pipeline, but the
    transforms drop ``CropAroundTumord`` so each sample is the full axial slice
    at ``img_size`` × ``img_size``. Cache is keyed on ``img_size`` only (the
    crop is always "whole slice" here). Safe to coexist with the 2D cache.
    """
    if dataset_type == "big_lunge":
        cache_name = "monai_biglunge_wholeslice"
    elif dataset_type == "lung_pet_ct_dx":
        cache_name = "monai_lung_pet_ct_clean_wholeslice"
    else:
        raise ValueError(f"Unknown dataset_type for whole-slice DAPT: '{dataset_type}'.")

    _mask_tag = ("_mask" if include_mask else "") + ("_bbox" if include_bbox else "")
    cache_root = os.path.join(
        os.path.expanduser("~"), ".cache", cache_name,
        f"img{img_size}_mp{int(min_tumor_pixels)}{_mask_tag}{'_testing' if testing else ''}",
    )
    if clear_cache and os.path.isdir(cache_root):
        import shutil as _shutil
        print(f"[--clear-cache] Removing {cache_root}")
        _shutil.rmtree(cache_root)
    os.makedirs(cache_root, exist_ok=True)

    if dataset_type == "big_lunge":
        if not csv_path:
            raise ValueError("csv_path is required for dataset_type='big_lunge'.")
        all_splits = get_biglunge_2d_data_list(
            data_path=data_path, csv_path=csv_path,
            cache_root=cache_root,
            tumor_mask_suffix=tumor_mask_suffix,
            val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
            min_tumor_pixels=min_tumor_pixels,
            max_slices_per_volume=max_slices_per_volume,
        )
    else:
        all_splits = get_lung_pet_ct_dx_2d_data_list(
            data_path=data_path, cache_root=cache_root,
            val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
            min_tumor_pixels=min_tumor_pixels,
            max_slices_per_volume=max_slices_per_volume,
        )

    datasets: List[PersistentDataset] = []
    for split in ("train", "val", "test"):
        data_list = all_splits[split]
        transforms = (
            get_train_transforms_whole_slice(
                img_size=img_size,
                strong_augs=strong_augs,
                include_mask=include_mask,
                include_bbox=include_bbox,
            )
            if split == "train"
            else get_val_transforms_whole_slice(
                img_size=img_size,
                include_mask=include_mask,
                include_bbox=include_bbox,
            )
        )

        if cache_dir is None:
            current_cache_dir = os.path.join(cache_root, split)
        else:
            current_cache_dir = os.path.join(cache_dir, split)
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"[whole-slice] PersistentDataset cache_dir='{current_cache_dir}'")

        valid_data_file = os.path.join(current_cache_dir, "valid_data.json")
        meta_file = os.path.join(current_cache_dir, "meta.json")
        current_meta = {
            "pipeline": "whole_slice",
            "dataset_type": dataset_type,
            "data_list_len": len(data_list),
            "testing": bool(testing),
            "val_frac": float(val_frac), "test_frac": float(test_frac),
            "seed": int(seed),
            "img_size": int(img_size),
            "tumor_mask_suffix": tumor_mask_suffix,
            "min_tumor_pixels": int(min_tumor_pixels),
            "max_slices_per_volume": max_slices_per_volume,
            "split": split,
            "include_mask": bool(include_mask),
            "include_bbox": bool(include_bbox),
        }

        cached_meta = None
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    cached_meta = json.load(f)
            except Exception:
                cached_meta = None
        cache_valid = (
            os.path.exists(valid_data_file)
            and not warm_cache
            and cached_meta == current_meta
        )

        if cache_valid:
            with open(valid_data_file, "r") as f:
                valid_data = json.load(f)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)
        else:
            ds = PersistentDataset(data=data_list, transform=transforms, cache_dir=current_cache_dir)
            valid_flags = [False] * len(data_list)
            n_workers = max(1, int(cache_workers))

            def _try_one(i: int):
                try:
                    _ = ds[i]
                    return i, None
                except Exception as e:  # noqa: BLE001
                    return i, e

            desc = f"Validating & Caching [whole-slice {split}] (threads={n_workers})"
            if n_workers == 1:
                for i in tqdm(range(len(ds)), desc=desc, unit="slice"):
                    _, err = _try_one(i)
                    if err is None:
                        valid_flags[i] = True
                    else:
                        print(f"Failed sample ({data_list[i].get('image', 'N/A')} @ z={data_list[i].get('slice_idx')}): {err}")
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_try_one, i) for i in range(len(ds))]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="slice"):
                        i, err = fut.result()
                        if err is None:
                            valid_flags[i] = True
                        else:
                            print(f"Failed sample ({data_list[i].get('image', 'N/A')} @ z={data_list[i].get('slice_idx')}): {err}")
            valid_data: List[Dict[str, Any]] = [data_list[i] for i, ok in enumerate(valid_flags) if ok]
            print(f"[whole-slice {split}] Kept {len(valid_data)}/{len(data_list)} valid slices.")
            with open(valid_data_file, "w") as f:
                json.dump(valid_data, f)
            with open(meta_file, "w") as f:
                json.dump(current_meta, f, indent=2)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]


# -----------------------------------------------------------------------------
# Bag-level BigLunge MIL (one entry per patient, lung mask drives bag selection)
# -----------------------------------------------------------------------------


def get_biglunge_mil_data_list(
    data_path: str,
    csv_path: str,
    lung_mask_suffix: str = "_label_lungs.nii.gz",
    tumor_mask_suffix: str = "_label_tc.nii.gz",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """BigLunge data list for MIL. One entry per patient.

    Each entry carries ``image`` (CT), ``lung_mask`` (algorithmic lung-chamber
    segmentation — reliable), ``scan_label``, ``patient_id``. Patients lacking
    the lung mask are dropped (we rely on it to pick the bag's z-extent).
    """
    # Reuse the 3D-splitting helper to get per-patient CT volumes + split.
    # get_biglunge_data_list already attaches 'lung_mask' when the sidecar
    # exists, so we only need to filter to entries with it.
    splits = get_biglunge_data_list(
        data_path=data_path, csv_path=csv_path,
        val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
    )

    # Truncated-lung-mask exclusions live in data/exclusions.py so the 3D
    # pipeline (which uses the same lung mask for its lung-bbox crop) drops
    # the same patients.
    data_root = Path(data_path)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, entries in splits.items():
        kept: List[Dict[str, Any]] = []
        dropped_no_mask = 0
        dropped_dupe_patient = 0
        dropped_truncated = 0
        seen_patients = set()
        for e in entries:
            pid = str(e.get("patient_id", "")).strip()
            if not pid:
                continue
            if pid in TRUNCATED_LUNG_MASK:
                dropped_truncated += 1
                continue
            # Canonicalize: one entry per patient (first CT wins). The 3D data
            # list may emit several .nii.gz per patient if multiple scans exist;
            # MIL operates at patient granularity.
            if pid in seen_patients:
                dropped_dupe_patient += 1
                continue
            lung_mask = e.get("lung_mask") or str(data_root / pid / f"{pid}{lung_mask_suffix}")
            if not os.path.isfile(lung_mask):
                dropped_no_mask += 1
                continue
            tumor_mask = str(data_root / pid / f"{pid}{tumor_mask_suffix}")
            entry: Dict[str, Any] = {
                "image": e["image"],
                "lung_mask": lung_mask,
                "scan_label": int(e["scan_label"]),
                "patient_id": pid,
                # volume_id mirrors image path so validate_epoch_mil can log it
                "volume_id": e["image"],
            }
            if os.path.isfile(tumor_mask):
                entry["tumor_mask"] = tumor_mask
            kept.append(entry)
            seen_patients.add(pid)

        if testing:
            kept = _testing_subset_balanced(kept, max_items=18, num_classes=3)

        cls_counts: Dict[int, int] = {}
        for e in kept:
            cls_counts[e["scan_label"]] = cls_counts.get(e["scan_label"], 0) + 1
        print(
            f"[MIL bag big_lunge {split}] {len(kept)} patients "
            f"(dropped no-lung-mask={dropped_no_mask}, dupe-patient={dropped_dupe_patient}, "
            f"truncated-lung={dropped_truncated}), "
            f"classes={cls_counts}"
        )
        out[split] = kept
    return out


def create_dataset_mil_bag(
    data_path: str,
    csv_path: str = "",
    dataset_type: str = "big_lunge",
    img_size: int = 384,
    bag_size: int = 16,
    lung_mask_suffix: str = "_label_lungs.nii.gz",
    tumor_mask_suffix: str = "_label_tc.nii.gz",
    cache_dir: Optional[str] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    warm_cache: bool = False,
    cache_workers: int = 4,
    strong_augs: bool = False,
    clear_cache: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """Create train/val/test MIL-bag PersistentDatasets for BigLunge.

    Each sample: ``(N, 1, img_size, img_size)`` where ``N = bag_size``.
    """
    if dataset_type != "big_lunge":
        raise ValueError(
            f"MIL bag pipeline is BigLunge-specific; got dataset_type='{dataset_type}'."
        )
    if not csv_path:
        raise ValueError("csv_path is required for dataset_type='big_lunge'.")

    cache_name = "monai_biglunge_mil"
    _mask_tag = ("_mask" if include_mask else "") + ("_bbox" if include_bbox else "")
    cache_root = os.path.join(
        os.path.expanduser("~"), ".cache", cache_name,
        f"img{img_size}_bag{int(bag_size)}{_mask_tag}{'_testing' if testing else ''}",
    )
    if clear_cache and os.path.isdir(cache_root):
        import shutil as _shutil
        print(f"[--clear-cache] Removing {cache_root}")
        _shutil.rmtree(cache_root)
    os.makedirs(cache_root, exist_ok=True)

    all_splits = get_biglunge_mil_data_list(
        data_path=data_path, csv_path=csv_path,
        lung_mask_suffix=lung_mask_suffix,
        tumor_mask_suffix=tumor_mask_suffix,
        val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
    )

    datasets: List[PersistentDataset] = []
    for split in ("train", "val", "test"):
        data_list = all_splits[split]
        transforms = (
            get_train_transforms_mil_bag(
                img_size=img_size,
                bag_size=bag_size,
                strong_augs=strong_augs,
                include_mask=include_mask,
                include_bbox=include_bbox,
            )
            if split == "train"
            else get_val_transforms_mil_bag(
                img_size=img_size,
                bag_size=bag_size,
                include_mask=include_mask,
                include_bbox=include_bbox,
            )
        )

        if cache_dir is None:
            current_cache_dir = os.path.join(cache_root, split)
        else:
            current_cache_dir = os.path.join(cache_dir, split)
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"[MIL bag] PersistentDataset cache_dir='{current_cache_dir}'")

        valid_data_file = os.path.join(current_cache_dir, "valid_data.json")
        meta_file = os.path.join(current_cache_dir, "meta.json")
        current_meta = {
            "pipeline": "mil_bag",
            "dataset_type": dataset_type,
            "data_list_len": len(data_list),
            "testing": bool(testing),
            "val_frac": float(val_frac), "test_frac": float(test_frac),
            "seed": int(seed),
            "img_size": int(img_size),
            "bag_size": int(bag_size),
            "lung_mask_suffix": lung_mask_suffix,
            "split": split,
            "include_mask": bool(include_mask),
            "include_bbox": bool(include_bbox),
        }

        cached_meta = None
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    cached_meta = json.load(f)
            except Exception:
                cached_meta = None
        cache_valid = (
            os.path.exists(valid_data_file)
            and not warm_cache
            and cached_meta == current_meta
        )

        if cache_valid:
            with open(valid_data_file, "r") as f:
                valid_data = json.load(f)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)
        else:
            ds = PersistentDataset(data=data_list, transform=transforms, cache_dir=current_cache_dir)
            valid_flags = [False] * len(data_list)
            n_workers = max(1, int(cache_workers))

            def _try_one(i: int):
                try:
                    _ = ds[i]
                    return i, None
                except Exception as e:  # noqa: BLE001
                    return i, e

            desc = f"Validating & Caching [MIL bag {split}] (threads={n_workers})"
            if n_workers == 1:
                for i in tqdm(range(len(ds)), desc=desc, unit="patient"):
                    _, err = _try_one(i)
                    if err is None:
                        valid_flags[i] = True
                    else:
                        print(f"Failed sample ({data_list[i].get('patient_id', 'N/A')}): {err}")
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_try_one, i) for i in range(len(ds))]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="patient"):
                        i, err = fut.result()
                        if err is None:
                            valid_flags[i] = True
                        else:
                            print(f"Failed sample ({data_list[i].get('patient_id', 'N/A')}): {err}")
            valid_data: List[Dict[str, Any]] = [data_list[i] for i, ok in enumerate(valid_flags) if ok]
            print(f"[MIL bag {split}] Kept {len(valid_data)}/{len(data_list)} valid patients.")
            with open(valid_data_file, "w") as f:
                json.dump(valid_data, f)
            with open(meta_file, "w") as f:
                json.dump(current_meta, f, indent=2)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]
