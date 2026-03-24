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

from data.transforms import get_train_transforms, get_val_transforms, get_train_transforms_3d, get_val_transforms_3d

# A: Adenocarcinoma (0), B: Small Cell Carcinoma (1), G: Squamous Cell Carcinoma (2)
CLASS_MAP = {"A": 0, "B": 1, "G": 2}
CLASS_NAMES = ["Adenocarcinoma", "Small Cell Carcinoma", "Squamous Cell Carcinoma"]

NORWEGIAN_CLASS_MAP = {
    "Adenokarsinom": 0,
    "Småcelletkarsinom": 1,
    "Plateepitelkarsinom": 2,
}

def load_patient_annotations(
    annotation_dir: str,
    patient_short_id: str,
    series_uid: str,
    orig_size: int = 512,
    target_size: int = 224, 
) -> Dict[str, torch.Tensor]:
    """Load and aggregate bounding box annotations from per-slice XML files (for Lung-PET-CT-Dx)."""
    patient_annot_dir = os.path.join(annotation_dir, patient_short_id)
    empty = {
        "boxes": [],
        "labels": [],
    }

    if not os.path.isdir(patient_annot_dir):
        return empty

    # Only load annotations specific to the current series
    xml_pattern = os.path.join(patient_annot_dir, f"*_{series_uid}_slice*.xml")
    xml_files = glob.glob(xml_pattern)
    if not xml_files:
        return empty

    all_boxes: List[tuple] = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None:
                    continue
                letter = name_elem.text.strip()
                if letter not in CLASS_MAP:
                    continue
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                slice_str = xml_path.split("_slice")[-1].replace(".xml", "")
                slice_idx = int(slice_str)
                all_boxes.append((xmin, ymin, xmax, ymax, slice_idx, CLASS_MAP[letter]))
        except (ET.ParseError, AttributeError):
            print(f"Warning: Failed to parse XML '{xml_path}'. Skipping.")
            continue

    if not all_boxes:
        return empty

    scale = target_size / orig_size
    scaled_boxes = []
    labels_list = []

    for b in all_boxes:
        xmin, ymin, xmax, ymax, slice_idx, class_id = b
        zmin, zmax = float(slice_idx), float(slice_idx + 1)
        scaled_boxes.append([xmin * scale, ymin * scale,zmin, xmax * scale, ymax * scale, zmax])
        labels_list.append(class_id + 1) # Detection labels are 1-indexed (0 = background)

    boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
    
    # dont clamp the z dimension since it is used for slice indexing, but clamp x and y to be within the target image size 
    boxes_tensor[:, [0, 3]] = boxes_tensor[:, [0, 3]].clamp(min=0, max=target_size)
    boxes_tensor[:, [1, 4]] = boxes_tensor[:, [1, 4]].clamp(min=0, max=target_size)

    return {
        "boxes": boxes_tensor.tolist(), 
        "labels": labels_tensor.tolist()
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
    annotation_dir: str = "",
    testing: bool = False,
    img_size: int = 224,    
) -> Dict[str, List[Dict[str, Any]]]:
    """Build {split: data_list} dict with patient-level splitting for Lung-PET-CT-Dx."""
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")

    data_root = Path(data_path)
    valid_extensions = (".nii.gz", ".nii", ".npy", ".npz")
    samples = [
        f.name for f in data_root.iterdir()
        if f.is_file() and any(f.name.endswith(ext) for ext in valid_extensions)
    ]

    if not samples:
        raise ValueError(f"No valid data files found in '{data_path}'. Supported: {valid_extensions}")

    patient_files: Dict[str, List[str]] = {}
    for f in samples:
        parts = f.split("_")
        if len(parts) >= 2:
            patient_id = f"{parts[0]}_{parts[1]}"
            patient_files.setdefault(patient_id, []).append(f)

    all_patients = sorted(patient_files.keys())

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

    annotation_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    result: Dict[str, List[Dict[str, Any]]] = {}
    
    for split, selected in split_patients.items():
        print(f"Split '{split}': {len(selected)} patients.")

        data_list = []
        for pid in selected:
            for f in patient_files.get(pid, []):
                label = -1
                for key, val in CLASS_MAP.items():
                    if f"-{key}" in f:
                        label = val
                        break
                if label == -1:
                    continue

                entry: Dict[str, Any] = {
                    "image": str(data_root / f),
                    "scan_label": label,
                }

                if annotation_dir and os.path.isdir(annotation_dir):
                    short_id = pid.split("-")[-1] if "-" in pid else pid
                    
                    # Extract series_uid from the file name (assuming format: patientID_seriesUID.nii.gz)
                    series_uid = f.replace(f"{pid}_", "").split(".nii")[0]
                    cache_key = f"{short_id}_{series_uid}"
                    
                    if cache_key not in annotation_cache:
                        annotation_cache[cache_key] = load_patient_annotations(
                            annotation_dir, short_id, series_uid, orig_size=512, target_size=img_size
                        )
                    annot = annotation_cache[cache_key]
                    entry["boxes"] = annot["boxes"]
                    entry["labels"] = annot["labels"]

                data_list.append(entry)
                if testing and len(data_list) >= 8:
                    break
            if testing and len(data_list) >= 8:
                break
                
        class_counts: Dict[int, int] = {}
        for item in data_list:
            class_counts[item["scan_label"]] = class_counts.get(item["scan_label"], 0) + 1

        if annotation_dir:
            n_with = sum(1 for d in data_list if d.get("boxes") is not None and len(d["boxes"]) > 0)
            print(f"  {len(data_list)} images ({n_with} with annotations), class distribution: {class_counts}")
        else:
            print(f"  {len(data_list)} images, class distribution: {class_counts}")

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
    annotation_dir: str = "",
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
            annotation_dir=annotation_dir, testing=testing, img_size=img_size
        )
        cache_name = "monai_lung_pet_ct"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    datasets = []
    
    # Import nibabel here to safely read NIfTI headers without fully loading
    import nibabel as nib
    
    for split in ("train", "val", "test"):
        data_list = all_splits[split]

        if use_3d:
            # Pre-filter to discard any volumes that don't have enough Z slices
            min_allowed_slices = depth_size - 79 # May change this later 128 - 79 = 49
            filtered_list = []
            
            print(f"[{split}] Filtering 3D volumes (min {min_allowed_slices} slices required)...")
            for item in tqdm(data_list, desc="Checking depth"):
                try:
                    img = nib.load(item["image"])
                    # Standard NIfTI has shape (X, Y, Z) or (X, Y, Z, T)
                    if len(img.shape) >= 3 and img.shape[2] >= min_allowed_slices:
                        filtered_list.append(item)
                    else:
                        print(f"Discarded {item['image']} (depth: {img.shape[2] if len(img.shape) >=3 else 0})")
                except Exception as e:
                    print(f"Skipping {item['image']} due to read error: {e}")
                    
            data_list = filtered_list

            if split == "train":
                transforms = get_train_transforms_3d(img_size=img_size, depth_size=depth_size)
            else:
                transforms = get_val_transforms_3d(img_size=img_size, depth_size=depth_size)
        else:
            get_transforms = get_train_transforms if split == "train" else get_val_transforms
            transforms = get_transforms(
                img_size=img_size,
                convert_to_rgb=convert_to_rgb,
                use_multichannel_windowing=use_multichannel_windowing,
            )

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

        ds = PersistentDataset(data=data_list, transform=transforms, cache_dir=current_cache_dir)

        # PersistentDataset automatically uses existing cache files. 
        # We only explicitly loop (warm up) if the directory is empty or warm_cache is forced.
        is_empty = len(os.listdir(current_cache_dir)) == 0 if os.path.exists(current_cache_dir) else True

        if warm_cache or is_empty:
            for i in tqdm(range(len(ds)), desc=f"Caching [{split}]", unit="img"):
                try:
                    _ = ds[i]
                except Exception as e:
                    print(f"Skipped caching sample {i} due to: {e}")

        datasets.append(ds)

    return datasets[0], datasets[1], datasets[2]


def get_class_names() -> List[str]:
    return CLASS_NAMES.copy()

def get_num_classes() -> int:
    return len(CLASS_NAMES)

