import os
import glob
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

# A: Adenocarcinoma (0), B: Small Cell Carcinoma (1), G: Squamous Cell Carcinoma (2)
CLASS_MAP = {"A": 0, "B": 1, "G": 2}
CLASS_NAMES = ["Adenocarcinoma", "Small Cell Carcinoma", "Squamous Cell Carcinoma"]

NORWEGIAN_CLASS_MAP = {
    "Adenokarsinom": 0,
    "Småcelletkarsinom": 1,
    "Plateepitelkarsinom": 2,
}

def load_patient_labels(csv_path: str) -> Dict[int, int]:
    """Load patient ID -> class label mapping from CSV (for BigLunge)."""
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        pid = int(row["Patient"])
        group = row["MorphologicalGroup"]
        if group in NORWEGIAN_CLASS_MAP:
            labels[pid] = NORWEGIAN_CLASS_MAP[group]
        else:
            print(f"Warning: Unknown morphological group '{group}' for patient {pid}")
    return labels

def get_lung_pet_ct_dx_data_list(
    data_path: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    pet_dir: str = "",
    use_pet: bool = False,
    testing: bool = False,
    img_size: int = 224,    
) -> Dict[str, List[Dict[str, Any]]]:
    """Build {split: data_list} dict with patient-level splitting for Lung-PET-CT-Dx."""
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")

    data_root = Path(data_path)
    
    patient_folders = [p for p in data_root.iterdir() if p.is_dir()]

    if not patient_folders:
        raise ValueError(f"No valid patient folders found in '{data_path}'.")

    all_patients = sorted([p.name for p in patient_folders])

    rng = np.random.default_rng(seed)
    rng.shuffle(all_patients)

    n_total = len(all_patients)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val

    split_patients = {
        "train": set(all_patients[:n_train]),
        "val": set(all_patients[n_train:n_train + n_val]),
        "test": set(all_patients[n_train + n_val:]),
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
            images = [f for f in patient_dir.iterdir() if f.is_file() and f.name.endswith("_image.nii.gz")]

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
                
                # Check for PET if requested
                if use_pet and pet_dir:
                    # PET is {pet_dir}/{pid}_*.nii.gz
                    pet_files = glob.glob(os.path.join(pet_dir, f"{pid}_*.nii.gz"))
                    if pet_files:
                        entry["pet"] = pet_files[0]
                    else:
                        # If use_pet is forced but this patient has no PET, skip this CT scan
                        continue
                        
                data_list.append(entry)
                if testing and len(data_list) >= 12:
                    break
            if testing and len(data_list) >= 12:
                break
                
        class_counts: Dict[int, int] = {}
        for item in data_list:
            class_counts[item["scan_label"]] = class_counts.get(item["scan_label"], 0) + 1

        mask_count = sum(1 for d in data_list if 'mask' in d)
        pet_count = sum(1 for d in data_list if 'pet' in d)
        print(f"  {len(data_list)} images ({mask_count} w/ masks, {pet_count} w/ PET), class distribution: {class_counts}")

        result[split] = data_list
        

    return result


def get_biglunge_data_list(
    data_path: str,
    csv_path: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
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
        int(e.name) for e in data_root.iterdir()
        if e.is_dir() and e.name.isdigit() and int(e.name) in patient_labels
    )

    if not patient_folders:
        raise ValueError(f"No valid patient folders found in '{data_path}'.")

    print(f"Found {len(patient_folders)} patients with labels.")

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_folders)

    n_total = len(patient_folders)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val

    split_patients = {
        "train": patient_folders[:n_train],
        "val": patient_folders[n_train:n_train + n_val],
        "test": patient_folders[n_train + n_val:],
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
            for nii in patient_dir.glob("*.nii*"):
                if "_label_" not in nii.name:
                    data_list.append({"image": str(nii), "scan_label": label, "patient_id": pid})
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
    pet_dir: str = "",
    use_pet: bool = False,
    use_3d: bool = False,
    testing: bool = False,
    warm_cache: bool = False,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
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
            pet_dir=pet_dir, use_pet=use_pet, testing=testing, img_size=img_size
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
            # Pre-filter to discard any volumes that don't have enough Z slices

            if split == "train":
                transforms = get_train_transforms_3d(img_size=img_size, depth_size=depth_size, use_pet=use_pet)
            else:
                transforms = get_val_transforms_3d(img_size=img_size, depth_size=depth_size, use_pet=use_pet)


        if cache_dir is None:
            mode_key = "3d" if use_3d else "2d"
            current_cache_dir = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                cache_name,
                f"{mode_key}_img{img_size}_d{depth_size}",
                split,
            )
        else:
            current_cache_dir = os.path.join(cache_dir, split)
            
        os.makedirs(current_cache_dir, exist_ok=True)
        print(f"PersistentDataset cache_dir='{current_cache_dir}'")
        
        valid_data_file = os.path.join(current_cache_dir, "valid_data.json")
        import json

        if os.path.exists(valid_data_file) and not warm_cache:
            print(f"Loading verified valid dataset list from {valid_data_file}...")
            with open(valid_data_file, "r") as f:
                valid_data = json.load(f)
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)
        else:
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
                
            # Recreate dataset using only the valid subset
            ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]


def get_class_names() -> List[str]:
    return CLASS_NAMES.copy()

def get_num_classes() -> int:
    return len(CLASS_NAMES)

