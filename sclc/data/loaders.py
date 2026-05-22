import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import pandas as pd
from monai.data import PersistentDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from sclc.data.exclusions import EMPTY_TUMOR_MASK, TRUNCATED_LUNG_MASK
from sclc.data.transforms import (
    get_train_transforms_3d,
    get_val_transforms_3d,
)

# Lung-PET-CT-Dx folder-name letter suffix -> class index.
CLASS_MAP = {"A": 0, "B": 1, "G": 2}
CLASS_NAMES = ["Adenocarcinoma", "Small Cell Carcinoma", "Squamous Cell Carcinoma"]

# BigLunge `MorphologicalGroup` value -> class index.
BIGLUNGE_CLASS_MAP = {
    "Adenocarcinoma": 0,
    "Small cell carcinoma": 1,
    "Squamous cell carcinoma": 2,
}

def load_patient_labels(csv_path: str) -> Dict[str, int]:
    """Load patient ID -> class label mapping from BigLunge CSV.
    Rows whose `MorphologicalGroup` is outside the 3 target classes are skipped.
    """
    df = pd.read_csv(csv_path)
    labels: Dict[str, int] = {}
    for _, row in df.iterrows():
        pid = str(row["Patient"]).strip()
        group = str(row["MorphologicalGroup"]).strip()
        if group in BIGLUNGE_CLASS_MAP:
            labels[pid] = BIGLUNGE_CLASS_MAP[group]
        else:
            print(f"Warning: Unknown morphological group '{group}' for patient {pid} — skipping")
    return labels

def get_lung_pet_ct_dx_data_list(
    data_path: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    max_scans_per_patient: int = 2,
    *,
    cv_fold: int,
    cv_folds: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build {split: data_list} dict with patient-level splitting for Lung-PET-CT-Dx.

    `cv_fold >= 0` selects stratified k-fold CV; `cv_fold == -1` uses a single
    70/15/15 split. Keyword-only to force callers to opt in — defaulting to
    `-1` once silently masked the CV path when a caller forgot the kwarg.
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")

    data_root = Path(data_path)
    
    patient_folders = [p for p in data_root.iterdir() if p.is_dir()]

    if not patient_folders:
        raise ValueError(f"No valid patient folders found in '{data_path}'.")

    all_patients = sorted([p.name for p in patient_folders])

    valid_patients = []
    patient_labels = []

    for pid in all_patients:
        # Folders like 'Lung_Dx-GA-0001' hit two CLASS_MAP keys — fail loud
        # rather than letting dict iteration pick first-match silently.
        matched = [val for key, val in CLASS_MAP.items() if f"-{key}" in pid]
        if len(matched) == 1:
            valid_patients.append(pid)
            patient_labels.append(matched[0])
        elif len(matched) > 1:
            raise ValueError(
                f"[lung_pet_ct_dx] ambiguous CLASS_MAP match for patient '{pid}': "
                f"hits {len(matched)} keys ({matched}). Refusing to silently pick one."
            )

    if cv_fold >= 0:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        all_folds = list(skf.split(valid_patients, patient_labels))
        train_val_idx, test_idx = all_folds[cv_fold]
        test_ids = [valid_patients[i] for i in test_idx]
        train_val_patients = [valid_patients[i] for i in train_val_idx]
        train_val_labels = [patient_labels[i] for i in train_val_idx]
        # Rescale val_frac so the val partition stays ~val_frac of the full set.
        val_frac_inner = min(val_frac / (1.0 - 1.0 / cv_folds), 0.49)
        train_ids, val_ids, _, _ = train_test_split(
            train_val_patients, train_val_labels,
            test_size=val_frac_inner, random_state=seed, stratify=train_val_labels,
        )
        print(f"[lung_pet_ct_dx CV fold {cv_fold}/{cv_folds}] "
              f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    else:
        val_test_frac = val_frac + test_frac
        if val_test_frac > 0:
            train_ids, temp_ids, train_labels, temp_labels = train_test_split(
                valid_patients, patient_labels, test_size=val_test_frac, random_state=seed, stratify=patient_labels
            )
            if test_frac > 0 and val_frac > 0:
                test_ratio = test_frac / val_test_frac
                val_ids, test_ids, val_labels, test_labels = train_test_split(
                    temp_ids, temp_labels, test_size=test_ratio, random_state=seed, stratify=temp_labels
                )
            elif test_frac > 0:
                val_ids, test_ids = [], temp_ids
            else:
                val_ids, test_ids = temp_ids, []
        else:
            train_ids, val_ids, test_ids = valid_patients, [], []

    split_patients = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }

    result: Dict[str, List[Dict[str, Any]]] = {}
    
    for split, selected in split_patients.items():
        print(f"Split '{split}': {len(selected)} patients.")

        data_list = []
        for pid in selected:
            matched = [val for key, val in CLASS_MAP.items() if f"-{key}" in pid]
            if len(matched) != 1:
                raise ValueError(
                    f"[lung_pet_ct_dx] expected exactly one CLASS_MAP match for '{pid}', got {matched}."
                )
            label = matched[0]

            patient_dir = data_root / pid
            image_files = [
                f for f in patient_dir.iterdir()
                if f.is_file() and f.name.endswith("_image.nii.gz")
            ]
            # Sort thinnest-Z first so multi-reconstruction patients prefer
            # the 1mm version over the 5mm one when capped by max_scans_per_patient.
            def _z_then_name(p):
                try:
                    return (float(nib.load(str(p)).header.get_zooms()[2]), p.name)
                except Exception:
                    return (float("inf"), p.name)
            images = sorted(image_files, key=_z_then_name)
            # Cap adeno (~8 scans/patient) so WeightedRandomSampler doesn't
            # over-repeat the rare SCLC volumes. SCLC (class 1) stays uncapped.
            if max_scans_per_patient is not None and max_scans_per_patient > 0 and label != 1:
                images = images[:max_scans_per_patient]

            for img_path in images:
                entry: Dict[str, Any] = {
                    "image": str(img_path),
                    "scan_label": label,
                    "patient_id": pid,
                }

                series_uid = img_path.name.replace("_image.nii.gz", "")

                mask_path = patient_dir / f"{series_uid}_mask.nii.gz"
                if mask_path.exists():
                    entry["mask"] = str(mask_path)

                data_list.append(entry)
                if testing and len(data_list) >= 16:
                    break
            if testing and len(data_list) >= 16:
                break
                
        class_counts: Dict[int, int] = {}
        for item in data_list:
            class_counts[item["scan_label"]] = class_counts.get(item["scan_label"], 0) + 1

        mask_count = sum(1 for d in data_list if 'mask' in d)
        print(f"  {len(data_list)} images ({mask_count} w/ masks), class distribution: {class_counts}")

        result[split] = data_list
        

    return result


def get_biglunge_data_list(
    data_path: str,
    csv_path: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    testing: bool = False,
    *,
    cv_fold: int,
    cv_folds: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build {split: data_list} dict with patient-level splitting for BigLunge.

    `cv_fold >= 0` selects stratified k-fold CV; `cv_fold == -1` uses a single
    70/15/15 split.
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")
    if not os.path.isfile(csv_path):
        raise ValueError(f"CSV file '{csv_path}' does not exist.")

    patient_labels = load_patient_labels(csv_path)

    data_root = Path(data_path)
    patient_folders = sorted(
        e.name for e in data_root.iterdir()
        if e.is_dir() and e.name in patient_labels
    )
    excluded = [pid for pid in patient_folders if pid in TRUNCATED_LUNG_MASK]
    if excluded:
        print(f"[big_lunge] Excluding {len(excluded)} truncated-lung-mask patients: {excluded}")
        patient_folders = [pid for pid in patient_folders if pid not in TRUNCATED_LUNG_MASK]

    excluded_tumor = [pid for pid in patient_folders if pid in EMPTY_TUMOR_MASK]
    if excluded_tumor:
        print(f"[big_lunge] Excluding {len(excluded_tumor)} empty-tumor-mask patients: {excluded_tumor}")
        patient_folders = [pid for pid in patient_folders if pid not in EMPTY_TUMOR_MASK]

    if not patient_folders:
        raise ValueError(
            f"No labeled patient folders found in '{data_path}'. "
            f"Folder names are expected to match the 'Patient' column in '{csv_path}'."
        )

    print(f"Found {len(patient_folders)} patients with labels.")
    
    patient_classes = [patient_labels[pid] for pid in patient_folders]

    if cv_fold >= 0:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        all_folds = list(skf.split(patient_folders, patient_classes))
        train_val_idx, test_idx = all_folds[cv_fold]
        test_ids = [patient_folders[i] for i in test_idx]
        train_val_folders = [patient_folders[i] for i in train_val_idx]
        train_val_classes = [patient_classes[i] for i in train_val_idx]
        # Rescale val_frac so the val partition stays ~val_frac of the full set.
        val_frac_inner = min(val_frac / (1.0 - 1.0 / cv_folds), 0.49)
        train_ids, val_ids, _, _ = train_test_split(
            train_val_folders, train_val_classes,
            test_size=val_frac_inner, random_state=seed, stratify=train_val_classes,
        )
        print(f"[big_lunge CV fold {cv_fold}/{cv_folds}] "
              f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    else:
        print("Using random train/val/test split with fixed seed.")
        val_test_frac = val_frac + test_frac
        if val_test_frac > 0:
            train_ids, temp_ids, train_classes, temp_classes = train_test_split(
                patient_folders, patient_classes, test_size=val_test_frac, random_state=seed, stratify=patient_classes
            )
            if test_frac > 0 and val_frac > 0:
                test_ratio = test_frac / val_test_frac
                val_ids, test_ids, val_classes, test_classes = train_test_split(
                    temp_ids, temp_classes, test_size=test_ratio, random_state=seed, stratify=temp_classes
                )
            elif test_frac > 0:
                val_ids, test_ids = [], temp_ids
            else:
                val_ids, test_ids = temp_ids, []
        else:
            train_ids, val_ids, test_ids = patient_folders, [], []

    split_patients = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, selected in split_patients.items():
        print(f"Split '{split}': {len(selected)} patients.")
        data_list = []
        for pid in selected:
            patient_dir = data_root / str(pid)
            if not patient_dir.is_dir():
                continue
            label = patient_labels[pid]
            # TrainingData layout: {pid}_input.nii.gz (CT) and {pid}_label_*.nii.gz.
            for nii in patient_dir.glob("*.nii*"):
                if "_label_" in nii.name:
                    continue
                entry: Dict[str, Any] = {
                    "image": str(nii),
                    "scan_label": label,
                    "patient_id": pid,
                }
                lung_mask_path = patient_dir / f"{pid}_label_lungs.nii.gz"
                if lung_mask_path.exists():
                    entry["lung_mask"] = str(lung_mask_path)
                # Tumor mask is consumed only for spatial centering (largest CC
                # picks the focal component in multifocal cases) — never as a
                # seg-aux target, since that would be circular distillation.
                tumor_mask_path = patient_dir / f"{pid}_label_tc.nii.gz"
                if tumor_mask_path.exists():
                    entry["mask"] = str(tumor_mask_path)
                data_list.append(entry)
                if testing and len(data_list) >= 32:
                    break
            if testing and len(data_list) >= 32:
                break

        class_counts: Dict[int, int] = {}
        for item in data_list:
            class_counts[item["scan_label"]] = class_counts.get(item["scan_label"], 0) + 1
        print(f"  {len(data_list)} images, class distribution: {class_counts}")
        result[split] = data_list

    return result


def create_dataset(
    dataset_type: str,
    data_path: str,
    csv_path: str = "",
    img_size: int = 224,
    depth_size: int = 64,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False,
    cache_dir: Optional[str] = None,
    num_workers: int = 4,
    use_3d: bool = False,
    testing: bool = False,
    warm_cache: bool = False,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    strong_augs: bool = False,
    clear_cache: bool = False,
    include_bbox: bool = False,
    *,
    cv_fold: int,
    cv_folds: int = 5,
    **kwargs: Any,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """Create train/val/test PersistentDatasets for `big_lunge` or
    `lung_pet_ct_dx`. The cache layout is shared across CV folds (per-patient
    preprocessing output is identical regardless of which fold a patient
    lands in) — fold-specific bookkeeping lives in `*_fold{N}.json` sidecars.
    """
    if dataset_type == "big_lunge":
        all_splits = get_biglunge_data_list(
            data_path=data_path, csv_path=csv_path,
            val_frac=val_frac, test_frac=test_frac, seed=seed,
            testing=testing, cv_fold=cv_fold, cv_folds=cv_folds,
        )
        cache_name = "monai_biglunge"
    elif dataset_type == "lung_pet_ct_dx":
        all_splits = get_lung_pet_ct_dx_data_list(
            data_path=data_path, val_frac=val_frac, test_frac=test_frac, seed=seed,
            testing=testing, cv_fold=cv_fold, cv_folds=cv_folds,
        )
        cache_name = "monai_lung_pet_ct_clean"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if cache_dir is None:
        mode_key = "3d" if use_3d else "2d"
        test_suffix = "_testing" if testing else ""
        run_cache_root = os.path.join(
            "/home/data/.cache", cache_name,
            f"{mode_key}_img{img_size}_d{depth_size}{test_suffix}",
        )
    else:
        run_cache_root = cache_dir
    # Only wipe on fold 0 (or non-CV) — sequential folds share the cache.
    if clear_cache and cv_fold <= 0 and os.path.isdir(run_cache_root):
        print(f"[--clear-cache] Removing {run_cache_root}")
        shutil.rmtree(run_cache_root)

    datasets = []

    for split in ("train", "val", "test"):
        data_list = all_splits[split]

        if use_3d:
            # BigLunge ships per-patient lung-chamber masks; crop a lung bbox
            # so the spatial budget focuses on lung + adjacent mediastinum.
            use_lung_crop = (dataset_type == "big_lunge")

            if split == "train":
                transforms = get_train_transforms_3d(
                    img_size=img_size, depth_size=depth_size,
                    use_lung_crop=use_lung_crop,
                    strong_augs=strong_augs,
                    include_bbox=include_bbox,
                )
            else:
                transforms = get_val_transforms_3d(
                    img_size=img_size, depth_size=depth_size,
                    use_lung_crop=use_lung_crop,
                    include_bbox=include_bbox,
                )


        current_cache_dir = os.path.join(run_cache_root, split)
            
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"PersistentDataset cache_dir='{current_cache_dir}'")

        _fold_suffix = f"_fold{cv_fold}" if cv_fold >= 0 else ""
        valid_data_file = os.path.join(current_cache_dir, f"valid_data{_fold_suffix}.json")
        meta_file = os.path.join(current_cache_dir, f"meta{_fold_suffix}.json")

        # Bump CACHE_SCHEMA_VERSION whenever the deterministic preprocessing
        # prefix changes (Spacingd pixdim, intensity window, centering rule).
        CACHE_SCHEMA_VERSION = 3
        current_meta = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset_type": dataset_type,
            "data_list_len": len(data_list),
            # data_list_keys catches schema drift in entry dicts (e.g. adding
            # `patient_id` for patient-level aggregation) that the schema
            # version doesn't otherwise capture.
            "data_list_keys": sorted(data_list[0].keys()) if data_list else [],
            "testing": bool(testing),
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "seed": int(seed),
            "img_size": int(img_size),
            "depth_size": int(depth_size),
            "split": split,
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
            print(f"Loading verified valid dataset list from {valid_data_file}...")
            with open(valid_data_file, "r") as f:
                valid_data = json.load(f)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)
        else:
            if os.path.exists(valid_data_file) and cached_meta != current_meta:
                print(
                    f"[{split}] Cache meta mismatch — rebuilding.\n"
                    f"  on disk: {cached_meta}\n"
                    f"  current: {current_meta}"
                )
            ds = PersistentDataset(data=data_list, transform=transforms, cache_dir=current_cache_dir)

            valid_data = []
            for i in tqdm(range(len(ds)), desc=f"Validating & Caching [{split}]", unit="img"):
                try:
                    _ = ds[i]
                    valid_data.append(data_list[i])
                except Exception as e:
                    print(f"Failed sample ({data_list[i].get('image', 'N/A')}) - skipping! Error: {e}")

            print(f"[{split}] Kept {len(valid_data)}/{len(data_list)} valid samples.")
            with open(valid_data_file, "w") as f:
                json.dump(valid_data, f)
            with open(meta_file, "w") as f:
                json.dump(current_meta, f, indent=2)

            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]


def get_class_names() -> List[str]:
    return CLASS_NAMES.copy()


def get_num_classes() -> int:
    return len(CLASS_NAMES)

