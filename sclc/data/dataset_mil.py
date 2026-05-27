"""MIL pipeline data builders.

Two dataset kinds share the 1 mm / 2 mm spacing + HU-windowed front-end
from `sclc.data.transforms`:

- whole-slice DAPT: per-slice samples at full axial FOV, tumor mask only
  used to enumerate tumor-bearing slices (reuses the 2D pipeline's index).
- MIL bag: one bag per patient at shape `(N, 1, H, W)`; bag instances are
  selected from the lung-mask z-extent. No tumor mask at inference — the
  attention head decides which instances drive the bag-level prediction.
"""
from __future__ import annotations

import json
import os
import shutil
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
from sclc.data.exclusions import TRUNCATED_LUNG_MASK
from sclc.data.loaders import (
    get_biglunge_data_list,
    get_lung_pet_ct_dx_data_list,
)
from sclc.data.transforms import (
    get_train_transforms_mil_bag,
    get_train_transforms_mil_bag_dapt,
    get_train_transforms_whole_slice,
    get_val_transforms_mil_bag,
    get_val_transforms_mil_bag_dapt,
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
    cv_fold: int = -1,
    cv_folds: int = 5,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """Per-slice whole-slice datasets for DAPT (full axial FOV per slice).

    Same tumor-slice-index entries as the 2D pipeline, with `CropAroundTumord`
    removed. Cache is keyed on `img_size` only and lives separately from the
    2D cache.
    """
    if dataset_type == "big_lunge":
        cache_name = "monai_biglunge_wholeslice"
    elif dataset_type == "lung_pet_ct_dx":
        cache_name = "monai_lung_pet_ct_clean_wholeslice"
    else:
        raise ValueError(f"Unknown dataset_type for whole-slice DAPT: '{dataset_type}'.")

    _mask_tag = ("_mask" if include_mask else "") + ("_bbox" if include_bbox else "")
    cache_root = os.path.join(
        "/home/data/.cache", cache_name,
        f"img{img_size}_mp{int(min_tumor_pixels)}{_mask_tag}{'_testing' if testing else ''}",
    )
    if clear_cache and cv_fold <= 0 and os.path.isdir(cache_root):
        print(f"[--clear-cache] Removing {cache_root}")
        shutil.rmtree(cache_root)
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
            cv_fold=cv_fold, cv_folds=cv_folds,
        )
    else:
        all_splits = get_lung_pet_ct_dx_2d_data_list(
            data_path=data_path, cache_root=cache_root,
            val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
            min_tumor_pixels=min_tumor_pixels,
            max_slices_per_volume=max_slices_per_volume,
            cv_fold=cv_fold, cv_folds=cv_folds,
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

        # Flat cache: MONAI's pickle_hashing is split-agnostic, so per-split
        # subdirs would duplicate the same slice when a patient lands in train
        # in fold k and test in fold j. Filename suffixes carry (fold, split).
        if cache_dir is None:
            current_cache_dir = cache_root
        else:
            current_cache_dir = cache_dir
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"[whole-slice] PersistentDataset cache_dir='{current_cache_dir}' (split='{split}')")

        _fold_suffix = f"_fold{cv_fold}" if cv_fold >= 0 else ""
        valid_data_file = os.path.join(current_cache_dir, f"valid_data{_fold_suffix}_{split}.json")
        meta_file = os.path.join(current_cache_dir, f"meta{_fold_suffix}_{split}.json")
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
            "cv_fold": int(cv_fold),
            "cv_folds": int(cv_folds),
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
    lung_mask_suffix: str = "_label_lungs.nii.gz",  # accepted for back-compat; unused
    tumor_mask_suffix: str = "_label_tc.nii.gz",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    cv_fold: int = -1,
    cv_folds: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """BigLunge data list for MIL — one entry per patient.

    Patients lacking the tumour mask file are dropped: it defines the
    bag's z-extent via `TumorPositiveBagSelectd` (every bag instance must
    sit on a tumour-positive z-index). With the `EMPTY_TUMOR_MASK`
    blocklist applied pre-split this should drop zero patients on the
    current cohort. Truncated-lung-mask exclusions are shared with the
    3D pipeline via `data/exclusions.py`.

    `lung_mask_suffix` is retained in the signature for back-compat with
    callers (main.py forwards `args.lung_mask_suffix`) but is no longer
    consulted; the old apex-to-base lung-mask bag selection has been
    retired.
    """
    del lung_mask_suffix  # unused — see docstring
    splits = get_biglunge_data_list(
        data_path=data_path, csv_path=csv_path,
        val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
        cv_fold=cv_fold, cv_folds=cv_folds,
    )

    data_root = Path(data_path)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, entries in splits.items():
        kept: List[Dict[str, Any]] = []
        dropped_no_tumor_mask = 0
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
            # MIL is patient-granular: first CT wins, drop duplicates.
            if pid in seen_patients:
                dropped_dupe_patient += 1
                continue
            tumor_mask = str(data_root / pid / f"{pid}{tumor_mask_suffix}")
            if not os.path.isfile(tumor_mask):
                dropped_no_tumor_mask += 1
                continue
            entry: Dict[str, Any] = {
                "image": e["image"],
                "tumor_mask": tumor_mask,
                "scan_label": int(e["scan_label"]),
                "patient_id": pid,
                "volume_id": e["image"],
            }
            kept.append(entry)
            seen_patients.add(pid)

        if testing:
            kept = _testing_subset_balanced(kept, max_items=18, num_classes=3)

        cls_counts: Dict[int, int] = {}
        for e in kept:
            cls_counts[e["scan_label"]] = cls_counts.get(e["scan_label"], 0) + 1
        print(
            f"[MIL bag big_lunge {split}] {len(kept)} patients "
            f"(dropped no-tumor-mask={dropped_no_tumor_mask}, dupe-patient={dropped_dupe_patient}, "
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
    cv_fold: int = -1,
    cv_folds: int = 5,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """Train/val/test MIL-bag `PersistentDataset`s for BigLunge.
    Each sample has shape `(bag_size, 1, img_size, img_size)`.
    """
    if dataset_type != "big_lunge":
        raise ValueError(
            f"MIL bag pipeline is BigLunge-specific; got dataset_type='{dataset_type}'."
        )
    if not csv_path:
        raise ValueError("csv_path is required for dataset_type='big_lunge'.")

    # v2 bump: BigLunge MIL switched from `LungAxialBagSelectd` over the
    # full apex-to-base lung extent to `TumorPositiveBagSelectd` over the
    # tumour-positive z-list. Old `monai_biglunge_mil` caches are now
    # semantically wrong and must not be reused — see also dataset_mil.py
    # commentary and transforms.py:TumorPositiveBagSelectd.
    cache_name = "monai_biglunge_mil_v2_tumor_pos"
    _mask_tag = ("_mask" if include_mask else "") + ("_bbox" if include_bbox else "")
    cache_root = os.path.join(
        "/home/data/.cache", cache_name,
        f"img{img_size}_bag{int(bag_size)}{_mask_tag}{'_testing' if testing else ''}",
    )
    if clear_cache and cv_fold <= 0 and os.path.isdir(cache_root):
        print(f"[--clear-cache] Removing {cache_root}")
        shutil.rmtree(cache_root)
    os.makedirs(cache_root, exist_ok=True)

    all_splits = get_biglunge_mil_data_list(
        data_path=data_path, csv_path=csv_path,
        lung_mask_suffix=lung_mask_suffix,
        tumor_mask_suffix=tumor_mask_suffix,
        val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
        cv_fold=cv_fold, cv_folds=cv_folds,
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
            current_cache_dir = cache_root
        else:
            current_cache_dir = cache_dir
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"[MIL bag] PersistentDataset cache_dir='{current_cache_dir}' (split='{split}')")

        _fold_suffix = f"_fold{cv_fold}" if cv_fold >= 0 else ""
        valid_data_file = os.path.join(current_cache_dir, f"valid_data{_fold_suffix}_{split}.json")
        meta_file = os.path.join(current_cache_dir, f"meta{_fold_suffix}_{split}.json")
        current_meta = {
            "pipeline": "mil_bag",
            "dataset_type": dataset_type,
            "data_list_len": len(data_list),
            "testing": bool(testing),
            "val_frac": float(val_frac), "test_frac": float(test_frac),
            "seed": int(seed),
            "img_size": int(img_size),
            "bag_size": int(bag_size),
            "tumor_mask_suffix": tumor_mask_suffix,
            "bag_select": "tumor_positive_zs",
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


# -----------------------------------------------------------------------------
# Bag-level Lung-PET-CT-Dx MIL DAPT (tumour mask drives z-extent if available)
# -----------------------------------------------------------------------------


def get_lung_pet_ct_dx_mil_data_list(
    data_path: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    max_scans_per_patient: int = 2,
    cv_fold: int = -1,
    cv_folds: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lung-PET-CT-Dx data list for MIL bag DAPT — one entry per CT series.

    When the tumour mask exists, `LungAxialBagSelectd` uses it as `source_key`
    to restrict the bag's z-extent; otherwise it falls back to the full CT.
    """
    splits = get_lung_pet_ct_dx_data_list(
        data_path=data_path,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        testing=testing,
        max_scans_per_patient=max_scans_per_patient,
        cv_fold=cv_fold,
        cv_folds=cv_folds,
    )

    data_root = Path(data_path)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, entries in splits.items():
        kept: List[Dict[str, Any]] = []
        for e in entries:
            entry: Dict[str, Any] = {
                "image": e["image"],
                "scan_label": int(e["scan_label"]),
                "patient_id": str(e.get("patient_id", "")),
                "volume_id": e["image"],
            }
            mask_path = e.get("mask") or str(
                data_root / str(e.get("patient_id", "")) / f"{e.get('patient_id', '')}_label_tc.nii.gz"
            )
            if os.path.isfile(mask_path):
                entry["mask"] = mask_path
            kept.append(entry)

        cls_counts: Dict[int, int] = {}
        for e in kept:
            cls_counts[e["scan_label"]] = cls_counts.get(e["scan_label"], 0) + 1
        print(
            f"[MIL bag dapt lung_pet_ct_dx {split}] {len(kept)} volumes, classes={cls_counts}"
        )
        out[split] = kept
    return out


def create_dataset_mil_bag_dapt(
    data_path: str,
    img_size: int = 256,
    bag_size: int = 16,
    cache_dir: Optional[str] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    warm_cache: bool = False,
    cache_workers: int = 4,
    strong_augs: bool = False,
    clear_cache: bool = False,
    max_scans_per_patient: int = 2,
    cv_fold: int = -1,
    cv_folds: int = 5,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """MIL-bag `PersistentDataset`s for DAPT on Lung-PET-CT-Dx."""
    cache_name = "monai_lung_pet_ct_mil_dapt"
    cache_root = os.path.join(
        "/home/data/.cache", cache_name,
        f"img{img_size}_bag{int(bag_size)}{'_testing' if testing else ''}",
    )
    if clear_cache and cv_fold <= 0 and os.path.isdir(cache_root):
        print(f"[--clear-cache] Removing {cache_root}")
        shutil.rmtree(cache_root)
    os.makedirs(cache_root, exist_ok=True)

    all_splits = get_lung_pet_ct_dx_mil_data_list(
        data_path=data_path,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        testing=testing,
        max_scans_per_patient=max_scans_per_patient,
        cv_fold=cv_fold,
        cv_folds=cv_folds,
    )

    datasets: List[PersistentDataset] = []
    for split in ("train", "val", "test"):
        data_list = all_splits[split]
        transforms = (
            get_train_transforms_mil_bag_dapt(img_size=img_size, bag_size=bag_size, strong_augs=strong_augs)
            if split == "train"
            else get_val_transforms_mil_bag_dapt(img_size=img_size, bag_size=bag_size)
        )

        if cache_dir is None:
            current_cache_dir = cache_root
        else:
            current_cache_dir = cache_dir
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"[MIL bag DAPT] PersistentDataset cache_dir='{current_cache_dir}' (split='{split}')")

        _fold_suffix = f"_fold{cv_fold}" if cv_fold >= 0 else ""
        valid_data_file = os.path.join(current_cache_dir, f"valid_data{_fold_suffix}_{split}.json")
        meta_file = os.path.join(current_cache_dir, f"meta{_fold_suffix}_{split}.json")
        current_meta = {
            "pipeline": "mil_bag_dapt",
            "dataset_type": "lung_pet_ct_dx",
            "data_list_len": len(data_list),
            "testing": bool(testing),
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "seed": int(seed),
            "img_size": int(img_size),
            "bag_size": int(bag_size),
            "max_scans_per_patient": int(max_scans_per_patient),
            "split": split,
            "cv_fold": int(cv_fold),
            "cv_folds": int(cv_folds),
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

            desc = f"Validating & Caching [MIL bag DAPT {split}] (threads={n_workers})"
            if n_workers == 1:
                for i in tqdm(range(len(ds)), desc=desc, unit="volume"):
                    _, err = _try_one(i)
                    if err is None:
                        valid_flags[i] = True
                    else:
                        print(f"Failed sample ({data_list[i].get('patient_id', 'N/A')}): {err}")
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_try_one, i) for i in range(len(ds))]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="volume"):
                        i, err = fut.result()
                        if err is None:
                            valid_flags[i] = True
                        else:
                            print(f"Failed sample ({data_list[i].get('patient_id', 'N/A')}): {err}")
            valid_data: List[Dict[str, Any]] = [data_list[i] for i, ok in enumerate(valid_flags) if ok]
            print(f"[MIL bag DAPT {split}] Kept {len(valid_data)}/{len(data_list)} valid volumes.")
            with open(valid_data_file, "w") as f:
                json.dump(valid_data, f)
            with open(meta_file, "w") as f:
                json.dump(current_meta, f, indent=2)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]
