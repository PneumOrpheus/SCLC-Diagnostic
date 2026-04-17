import os
import pandas as pd
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from monai.data import PersistentDataset
from monai.transforms import Compose
from tqdm import tqdm
import torchvision

from data.transforms import get_train_transforms_3d, get_val_transforms_3d

from sklearn.model_selection import train_test_split

# A: Adenocarcinoma (0), B: Small Cell Carcinoma (1), G: Squamous Cell Carcinoma (2)
CLASS_MAP = {"A": 0, "B": 1, "G": 2}
CLASS_NAMES = ["Adenocarcinoma", "Small Cell Carcinoma", "Squamous Cell Carcinoma"]

BIGLUNGE_CLASS_MAP = {
    # English labels used by /home/data/TrainingData/patients_parameters.csv
    "Adenocarcinoma": 0,
    "Small cell carcinoma": 1,
    "Squamous cell carcinoma": 2,
    # Norwegian labels kept for backward compatibility with the old BigLunge CSV
    "Adenokarsinom": 0,
    "Småcelletkarsinom": 1,
    "Plateepitelkarsinom": 2,
}

def load_patient_labels(csv_path: str) -> Dict[str, int]:
    """Load patient ID -> class label mapping from CSV (for BigLunge).

    Patient IDs are kept as strings (e.g. ``patient_087599``) to match the
    folder naming in /home/data/TrainingData. Rows whose MorphologicalGroup is
    not one of the three target classes (e.g. ``Non-small cell carcinoma``)
    are skipped.
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
) -> Dict[str, List[Dict[str, Any]]]:
    """Build {split: data_list} dict with patient-level splitting for Lung-PET-CT-Dx."""
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")

    data_root = Path(data_path)
    
    patient_folders = [p for p in data_root.iterdir() if p.is_dir()]

    if not patient_folders:
        raise ValueError(f"No valid patient folders found in '{data_path}'.")

    all_patients = sorted([p.name for p in patient_folders])
    
    # Filter patients by valid class mapping and associate their labels
    valid_patients = []
    patient_labels = []
    
    for pid in all_patients:
        label = -1
        for key, val in CLASS_MAP.items():
            if f"-{key}" in pid:
                label = val
                break
        if label != -1:
            valid_patients.append(pid)
            patient_labels.append(label)

    val_test_frac = val_frac + test_frac
    if val_test_frac > 0:
        train_ids, temp_ids, train_labels, temp_labels = train_test_split(
            valid_patients, patient_labels, test_size=val_test_frac, random_state=seed, stratify=patient_labels
        )
        
        # Split temp into val and test
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
            label = -1
            for key, val in CLASS_MAP.items():
                if f"-{key}" in pid:
                    label = val
                    break
            if label == -1:
                continue
                
            patient_dir = data_root / pid
            images = sorted(
                [f for f in patient_dir.iterdir() if f.is_file() and f.name.endswith("_image.nii.gz")],
                key=lambda f: f.name,
            )
            # Flatten adeno dominance (~8 scans/patient) so WeightedRandomSampler
            # doesn't repeat-sample the same SCLC volumes dozens of times per epoch.
            # Deterministic: sorted by filename, take first N.
            # SCLC (class 1) is exempt — it's too rare to cap.
            if max_scans_per_patient is not None and max_scans_per_patient > 0 and label != 1:
                images = images[:max_scans_per_patient]

            for img_path in images:
                entry: Dict[str, Any] = {
                    "image": str(img_path),
                    "scan_label": label,
                }

                series_uid = img_path.name.replace("_image.nii.gz", "")
                
                # Check for mask in the same clean folder
                mask_path = patient_dir / f"{series_uid}_mask.nii.gz"
                if mask_path.exists():
                    entry["mask"] = str(mask_path)
                        
                data_list.append(entry)
                if testing and len(data_list) >= 12:
                    break
            if testing and len(data_list) >= 12:
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
) -> Dict[str, List[Dict[str, Any]]]:
    """Build {split: data_list} dict with patient-level splitting for BigLunge."""
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

    if not patient_folders:
        raise ValueError(
            f"No labeled patient folders found in '{data_path}'. "
            f"Folder names are expected to match the 'Patient' column in '{csv_path}'."
        )

    print(f"Found {len(patient_folders)} patients with labels.")
    
    patient_classes = [patient_labels[pid] for pid in patient_folders]

    from sklearn.model_selection import train_test_split

    val_test_frac = val_frac + test_frac
    if val_test_frac > 0:
        train_ids, temp_ids, train_classes, temp_classes = train_test_split(
            patient_folders, patient_classes, test_size=val_test_frac, random_state=seed, stratify=patient_classes
        )
        
        # Split temp into val and test
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
            # New TrainingData layout: {pid}_input.nii.gz (CT) and
            # {pid}_label_lungs.nii.gz (algorithmic lung-chamber mask).
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
    **kwargs: Any,
) -> Tuple[PersistentDataset, PersistentDataset, PersistentDataset]:
    """
    Unified function to create train/val/test PersistentDatasets for SCLC.
    
    Args:
        dataset_type: "big_lunge" or "lung_pet_ct_dx"
        ...
    """
    if dataset_type == "big_lunge":
        all_splits = get_biglunge_data_list(
            data_path=data_path, csv_path=csv_path,
            val_frac=val_frac, test_frac=test_frac, seed=seed,
            testing=testing,
        )
        cache_name = "monai_biglunge"
    elif dataset_type == "lung_pet_ct_dx":
        all_splits = get_lung_pet_ct_dx_data_list(
            data_path=data_path, val_frac=val_frac, test_frac=test_frac, seed=seed,
            testing=testing
        )
        cache_name = "monai_lung_pet_ct_clean"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    datasets = []
    
    # Import nibabel here to safely read NIfTI headers without fully loading
    import nibabel as nib
    
    for split in ("train", "val", "test"):
        data_list = all_splits[split]

        if use_3d:
            # BigLunge ships per-patient algorithmic lung-chamber masks; use
            # them to crop a generous lung-bbox so the limited spatial budget
            # focuses on lung tissue and adjacent mediastinum.
            use_lung_crop = (dataset_type == "big_lunge")

            if split == "train":
                transforms = get_train_transforms_3d(
                    img_size=img_size, depth_size=depth_size,
                    use_lung_crop=use_lung_crop,
                )
            else:
                transforms = get_val_transforms_3d(
                    img_size=img_size, depth_size=depth_size,
                    use_lung_crop=use_lung_crop,
                )


        if cache_dir is None:
            mode_key = "3d" if use_3d else "2d"
            test_suffix = "_testing" if testing else ""
            current_cache_dir = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                cache_name,
                f"{mode_key}_img{img_size}_d{depth_size}{test_suffix}",
                split,
            )
        else:
            current_cache_dir = os.path.join(cache_dir, split)
            
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"PersistentDataset cache_dir='{current_cache_dir}'")

        valid_data_file = os.path.join(current_cache_dir, "valid_data.json")
        meta_file = os.path.join(current_cache_dir, "meta.json")
        import json

        # Cache key covers everything that can change the split or preprocessing shape.
        # If any of these drift from what's on disk, the cache is rebuilt.
        current_meta = {
            "dataset_type": dataset_type,
            "data_list_len": len(data_list),
            "testing": bool(testing),
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "seed": int(seed),
            "img_size": int(img_size),
            "depth_size": int(depth_size),
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

            # Recreate dataset using only the valid subset
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]


def get_class_names() -> List[str]:
    return CLASS_NAMES.copy()

def get_num_classes() -> int:
    return len(CLASS_NAMES)

