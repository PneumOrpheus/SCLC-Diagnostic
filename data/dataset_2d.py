"""2D per-slice pipeline.

For every volume with a tumor mask, we scan the mask once to enumerate axial
slices that contain tumor. Each such slice becomes a training sample. This
yields thousands of samples from hundreds of volumes — the 2D baseline most
2D medical-imaging papers use.

Shared utilities with the 3D/2.5D data builders live in ``data/data_loader.py``
(class maps, ``load_patient_labels``, patient split logic inside
``get_biglunge_data_list`` / ``get_lung_pet_ct_dx_data_list``). This file only
adds the per-slice expansion + tumor-slice index cache.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from monai.data import PersistentDataset
from tqdm import tqdm

from data.data_loader import (
    get_biglunge_data_list,
    get_lung_pet_ct_dx_data_list,
)
from data.transforms import get_train_transforms_2d, get_val_transforms_2d


def _scan_tumor_slice_indices(
    mask_path: str,
    min_pixels: int = 1,
    pixdim=(1.0, 1.0, 2.0),
) -> List[int]:
    """Return the axial-slice indices (in the RAS + ``pixdim``-resampled grid)
    that contain ``>= min_pixels`` non-zero mask voxels. ``pixdim`` must match
    the value used by ``_build_2d_pipeline``'s Spacingd, otherwise the slice
    indices are off-by-resampling.
    """
    from monai.transforms import (
        Compose, EnsureChannelFirst, LoadImage, Orientation, Spacing,
    )

    loader = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=pixdim, mode="nearest"),
        ]
    )
    mask = loader(mask_path)  # (1, H, W, D) in canonical RAS at target spacing
    arr = mask[0].cpu().numpy() if hasattr(mask[0], "cpu") else np.asarray(mask[0])
    if arr.ndim < 2:
        return []
    reduce_axes = tuple(range(arr.ndim - 1))
    per_slice = (arr > 0.5).sum(axis=reduce_axes)
    return [int(i) for i, c in enumerate(per_slice) if int(c) >= int(min_pixels)]


def _tumor_slice_index(
    mask_paths: List[str],
    cache_path: str,
    min_pixels: int = 1,
) -> Dict[str, List[int]]:
    """Build/refresh a {mask_path: [slice_idx, ...]} cache on disk. Only rescans
    masks whose mtime changed or that aren't in the cache yet.
    """
    index: Dict[str, Any] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                index = json.load(f)
        except Exception:
            index = {}

    dirty = False
    for p in mask_paths:
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            continue
        cached = index.get(p)
        if (
            isinstance(cached, dict)
            and cached.get("mtime") == mtime
            and cached.get("min_pixels") == int(min_pixels)
            and isinstance(cached.get("slices"), list)
        ):
            continue
        try:
            slices = _scan_tumor_slice_indices(p, min_pixels=min_pixels)
        except Exception as e:
            print(f"[2D] tumor-slice scan failed for {p}: {e}")
            slices = []
        index[p] = {"mtime": mtime, "min_pixels": int(min_pixels), "slices": slices}
        dirty = True

    if dirty:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(index, f)
        os.replace(tmp, cache_path)

    return {p: entry["slices"] for p, entry in index.items() if isinstance(entry, dict)}


def _expand_volume_entries_to_slices(
    volume_entries: List[Dict[str, Any]],
    mask_key: str,
    tumor_slice_index: Dict[str, List[int]],
    max_slices_per_volume: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """One volume -> one entry per tumor slice. ``mask_key`` names the
    per-volume tumor mask on the input entry.
    """
    out: List[Dict[str, Any]] = []
    for v in volume_entries:
        mask_path = v.get(mask_key)
        if not mask_path:
            continue
        slices = tumor_slice_index.get(mask_path, [])
        if not slices:
            continue
        # Deterministic subsample: evenly spaced across the tumor extent, so
        # even volumes with 40+ tumor slices don't dominate the training set.
        if max_slices_per_volume and len(slices) > max_slices_per_volume:
            picks = np.linspace(0, len(slices) - 1, max_slices_per_volume).round().astype(int)
            slices = [slices[i] for i in picks]
        for s in slices:
            entry = {k: val for k, val in v.items() if k != mask_key}
            entry["image"] = v["image"]
            entry["tumor_mask"] = mask_path
            entry["slice_idx"] = int(s)
            # Volume identity — used for patient/series-level eval aggregation.
            entry.setdefault("volume_id", v.get("image"))
            out.append(entry)
    return out


def _testing_subset_balanced(
    entries: List[Dict[str, Any]],
    max_items: int = 64,
    num_classes: int = 3,
    label_key: str = "scan_label",
) -> List[Dict[str, Any]]:
    """Deterministically cap a testing split while preserving class coverage.

    Strategy:
    1) Round-robin pick across class buckets in original order.
    2) If fewer than ``max_items`` are selected, fill from the remaining
       entries in original order.
    """
    if max_items <= 0 or len(entries) <= max_items:
        return entries

    buckets: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(num_classes)}
    for e in entries:
        y = e.get(label_key)
        try:
            yi = int(y)
        except (TypeError, ValueError):
            continue
        if 0 <= yi < num_classes:
            buckets[yi].append(e)

    selected: List[Dict[str, Any]] = []
    cursors: Dict[int, int] = {c: 0 for c in range(num_classes)}
    active = [c for c in range(num_classes) if buckets[c]]

    while len(selected) < max_items and active:
        next_active: List[int] = []
        for c in active:
            i = cursors[c]
            if i < len(buckets[c]):
                selected.append(buckets[c][i])
                cursors[c] = i + 1
                if cursors[c] < len(buckets[c]) and len(selected) < max_items:
                    next_active.append(c)
            if len(selected) >= max_items:
                break
        active = next_active

    if len(selected) < max_items:
        selected_ids = {id(x) for x in selected}
        for e in entries:
            if id(e) in selected_ids:
                continue
            selected.append(e)
            selected_ids.add(id(e))
            if len(selected) >= max_items:
                break

    return selected


def get_biglunge_2d_data_list(
    data_path: str,
    csv_path: str,
    cache_root: str,
    tumor_mask_suffix: str = "_label_tumor.nii.gz",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    min_tumor_pixels: int = 1,
    max_slices_per_volume: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """BigLunge 2D data list: one entry per tumor slice. Reuses the standard
    patient-stratified split from ``get_biglunge_data_list``, then attaches the
    per-patient tumor mask + slice index.
    """
    volumes = get_biglunge_data_list(
        data_path=data_path, csv_path=csv_path,
        val_frac=val_frac, test_frac=test_frac, seed=seed, testing=testing,
    )

    data_root = Path(data_path)
    mask_paths: List[str] = []
    for split_entries in volumes.values():
        for v in split_entries:
            pid = v.get("patient_id")
            if pid is None:
                continue
            mask_path = data_root / str(pid) / f"{pid}{tumor_mask_suffix}"
            if mask_path.exists():
                v["tumor_mask"] = str(mask_path)
                mask_paths.append(str(mask_path))

    index_path = os.path.join(cache_root, "tumor_slice_index.json")
    tumor_index = _tumor_slice_index(
        sorted(set(mask_paths)), index_path, min_pixels=min_tumor_pixels
    )

    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, split_entries in volumes.items():
        expanded = _expand_volume_entries_to_slices(
            split_entries, mask_key="tumor_mask",
            tumor_slice_index=tumor_index,
            max_slices_per_volume=max_slices_per_volume,
        )
        if testing:
            expanded = _testing_subset_balanced(expanded, max_items=64, num_classes=3)
        cls_counts: Dict[int, int] = {}
        for e in expanded:
            cls_counts[e["scan_label"]] = cls_counts.get(e["scan_label"], 0) + 1
        print(f"[2D big_lunge {split}] {len(expanded)} slices from {len(split_entries)} volumes, classes={cls_counts}")
        out[split] = expanded
    return out


def get_lung_pet_ct_dx_2d_data_list(
    data_path: str,
    cache_root: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    max_scans_per_patient: int = 2,
    min_tumor_pixels: int = 1,
    max_slices_per_volume: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lung-PET-CT-Dx 2D data list: one entry per tumor slice. The per-series
    ``_mask.nii.gz`` already sits in each volume entry as ``mask`` — rename it
    to ``tumor_mask`` and expand to slices.
    """
    volumes = get_lung_pet_ct_dx_data_list(
        data_path=data_path, val_frac=val_frac, test_frac=test_frac, seed=seed,
        testing=testing, max_scans_per_patient=max_scans_per_patient,
    )

    # Attach patient_id (derived from the Lung-PET-CT-Dx folder name) so the
    # 2D validator can aggregate slice-level probabilities per patient rather
    # than per scan. The 3D builder omits this field; adding it here keeps the
    # existing 3D PersistentDataset cache hash unchanged.
    mask_paths: List[str] = []
    for split_entries in volumes.values():
        for v in split_entries:
            img_path = v.get("image", "")
            v["patient_id"] = Path(img_path).parent.name if img_path else None
            if "mask" in v:
                v["tumor_mask"] = v.pop("mask")
                mask_paths.append(v["tumor_mask"])

    index_path = os.path.join(cache_root, "tumor_slice_index.json")
    tumor_index = _tumor_slice_index(
        sorted(set(mask_paths)), index_path, min_pixels=min_tumor_pixels
    )

    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, split_entries in volumes.items():
        expanded = _expand_volume_entries_to_slices(
            split_entries, mask_key="tumor_mask",
            tumor_slice_index=tumor_index,
            max_slices_per_volume=max_slices_per_volume,
        )
        if testing:
            expanded = _testing_subset_balanced(expanded, max_items=64, num_classes=3)
        cls_counts: Dict[int, int] = {}
        for e in expanded:
            cls_counts[e["scan_label"]] = cls_counts.get(e["scan_label"], 0) + 1
        print(f"[2D lung_pet_ct_dx {split}] {len(expanded)} slices from {len(split_entries)} volumes, classes={cls_counts}")
        out[split] = expanded
    return out


def create_dataset_2d(
    data_path: str,
    csv_path: str = "",
    dataset_type: str = "big_lunge",
    img_size: int = 224,
    tumor_mask_suffix: str = "_label_tumor.nii.gz",
    max_slices_per_volume: Optional[int] = None,
    min_tumor_pixels: int = 1,
    cache_dir: Optional[str] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    warm_cache: bool = False,
    cache_workers: int = 8,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """Create train/val/test ``PersistentDataset``s of 2D tumor slices.
    Samples are (C=1, img_size, img_size). Supported ``dataset_type``:
    ``"big_lunge"`` and ``"lung_pet_ct_dx"``.
    """
    if dataset_type == "big_lunge":
        cache_name = "monai_biglunge_2d"
    elif dataset_type == "lung_pet_ct_dx":
        cache_name = "monai_lung_pet_ct_clean_2d"
    else:
        raise ValueError(f"Unknown dataset_type for 2D: '{dataset_type}'.")

    cache_root = os.path.join(
        os.path.expanduser("~"), ".cache", cache_name,
        f"img{img_size}{'_testing' if testing else ''}",
    )
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
            get_train_transforms_2d(img_size=img_size)
            if split == "train"
            else get_val_transforms_2d(img_size=img_size)
        )

        if cache_dir is None:
            current_cache_dir = os.path.join(cache_root, split)
        else:
            current_cache_dir = os.path.join(cache_dir, split)
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"[2D] PersistentDataset cache_dir='{current_cache_dir}'")

        valid_data_file = os.path.join(current_cache_dir, "valid_data.json")
        meta_file = os.path.join(current_cache_dir, "meta.json")
        current_meta = {
            "pipeline": "2d",
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
                except Exception as e:  # noqa: BLE001 — we want every failure logged, not the first one to abort
                    return i, e

            desc = f"Validating & Caching [2D {split}] (threads={n_workers})"
            if n_workers == 1:
                for i in tqdm(range(len(ds)), desc=desc, unit="slice"):
                    _, err = _try_one(i)
                    if err is None:
                        valid_flags[i] = True
                    else:
                        print(f"Failed sample ({data_list[i].get('image', 'N/A')} @ z={data_list[i].get('slice_idx')}): {err}")
            else:
                # MONAI LoadImage + numpy release the GIL during I/O, so threads
                # parallelize the disk-bound first-pass cache build well.
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_try_one, i) for i in range(len(ds))]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="slice"):
                        i, err = fut.result()
                        if err is None:
                            valid_flags[i] = True
                        else:
                            print(f"Failed sample ({data_list[i].get('image', 'N/A')} @ z={data_list[i].get('slice_idx')}): {err}")
            valid_data: List[Dict[str, Any]] = [data_list[i] for i, ok in enumerate(valid_flags) if ok]
            print(f"[2D {split}] Kept {len(valid_data)}/{len(data_list)} valid slices.")
            with open(valid_data_file, "w") as f:
                json.dump(valid_data, f)
            with open(meta_file, "w") as f:
                json.dump(current_meta, f, indent=2)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]
