import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from monai.data import CacheDataset  # type: ignore[attr-defined]
from monai.transforms import Compose  # type: ignore[attr-defined]

from data.transforms import get_train_transforms, get_val_transforms

"""
BigLunge Dataset Loader
-----------------------
Data loader for the BigLunge dataset with CSV-based ground truth labels.
The dataset is organized by patient folders with NiFTI files.
"""

# Maps Norwegian cancer type names from CSV to class indices
NORWEGIAN_CLASS_MAP = {
    "Adenokarsinom": 0,
    "Småcelletkarsinom": 1,
    "Plateepitelkarsinom": 2,
}

# English class names for reference - aligned with Lung-PET-CT-Dx: A=0, B=1, G=2
CLASS_NAMES = [
    "Adenocarcinoma",
    "Small Cell Carcinoma", 
    "Squamous Cell Carcinoma",
]

def load_patient_labels(csv_path: str) -> Dict[int, int]:
    """Load patient labels from the CSV file.
    
    Args:
        csv_path: Path to patients_parameters.csv file.
        
    Returns:
        Dictionary mapping patient ID (int) to class label (int).
    """
    df = pd.read_csv(csv_path)
    
    patient_labels = {}
    for _, row in df.iterrows():
        patient_id = int(row["Patient"])
        morphological_group = row["MorphologicalGroup"]
        
        if morphological_group in NORWEGIAN_CLASS_MAP:
            patient_labels[patient_id] = NORWEGIAN_CLASS_MAP[morphological_group]
        else:
            print(f"Warning: Unknown morphological group '{morphological_group}' for patient {patient_id}")
    
    return patient_labels

def get_biglunge_data_list(
    data_path: str,
    csv_path: str,
    split: str = "train",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Create a list of data dictionaries for BigLunge dataset with patient-level splitting.
    
    Args:
        data_path: Path to directory containing patient folders 
                   (e.g., /home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1).
        csv_path: Path to patients_parameters.csv file.
        split: One of 'train', 'val', or 'test'.
        val_frac: Fraction of patients to use for validation.
        test_frac: Fraction of patients to use for testing.
        seed: Random seed for reproducibility.
        
    Returns:
        List of dictionaries with 'image' and 'scan_label' keys.
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")
    
    if not os.path.isfile(csv_path):
        raise ValueError(f"CSV file '{csv_path}' does not exist.")
    
    # Load patient labels from CSV
    patient_labels = load_patient_labels(csv_path)
    
    # List all patient folders
    try:
        all_folders = os.listdir(data_path)
    except OSError as e:
        raise ValueError(f"Unable to list contents of data path '{data_path}': {e}") from e
    
    # Filter to numeric folder name/patient IDs
    patient_folders = []
    for folder in all_folders:
        try:
            patient_id = int(folder)
            if patient_id in patient_labels:
                patient_folders.append(patient_id)
            else:
                print(f"Warning: Patient folder {folder} not found in CSV labels, skipping.")
        except ValueError:
            continue  # Skip non-numeric folders
    
    patient_folders = sorted(patient_folders)
    
    if not patient_folders:
        raise ValueError(f"No valid patient folders found in '{data_path}'.")
    
    print(f"Found {len(patient_folders)} patients with labels in BigLunge dataset.")
    
    # Shuffle patients deterministically
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_folders)
    
    # Calculate split indices
    n_total = len(patient_folders)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val
    
    if split == 'train':
        selected_patients = patient_folders[:n_train]
    elif split == 'val':
        selected_patients = patient_folders[n_train:n_train + n_val]
    elif split == 'test':
        selected_patients = patient_folders[n_train + n_val:]
    else:
        selected_patients = patient_folders
    
    print(f"Data Split '{split}': {len(selected_patients)} patients.")
    
    # Create list of dictionaries for MONAI
    data_list = []
    
    for patient_id in selected_patients:
        patient_folder = os.path.join(data_path, str(patient_id))
        
        if not os.path.isdir(patient_folder):
            continue
        
        # Find CT scan files (exclude lung segmentation masks)
        for f in os.listdir(patient_folder):
            if f.endswith(('.nii.gz', '.nii')):
                # Skip lung segmentation masks
                if '_label_Lungs_auto' in f:
                    continue
                
                data_list.append({
                    "image": os.path.join(patient_folder, f),
                    "scan_label": patient_labels[patient_id],
                    "patient_id": patient_id,
                })
    
    print(f"  -> {len(data_list)} images found for split '{split}'.")
    
    # Print class distribution
    class_counts = {}
    for item in data_list:
        label = item["scan_label"]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"  -> Class distribution: {class_counts}")
    
    return data_list

def create_biglunge_dataset(
    data_path: str,
    csv_path: str,
    split: str = "train",
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False,
    cache_rate: float = 1.0,
    num_workers: int = 4,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> CacheDataset:
    """Create a MONAI CacheDataset for BigLunge.
    
    Args:
        data_path: Path to directory containing patient folders.
        csv_path: Path to patients_parameters.csv file.
        split: One of 'train', 'val', or 'test'.
        img_size: Target image size (height and width).
        convert_to_rgb: Whether to convert grayscale to 3-channel RGB.
        use_multichannel_windowing: Whether to use multi-channel CT windowing.
        cache_rate: Fraction of data to cache (0.0 to 1.0).
        num_workers: Number of workers for caching.
        val_frac: Fraction of patients for validation.
        test_frac: Fraction of patients for testing.
        seed: Random seed for reproducibility.
        
    Returns:
        CacheDataset configured for the specified split.
    """
    data_list = get_biglunge_data_list(
        data_path=data_path,
        csv_path=csv_path,
        split=split,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed
    )
    
    if split == "train":
        transforms = get_train_transforms(
            img_size=img_size,
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing
        )
    else:
        transforms = get_val_transforms(
            img_size=img_size,
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing
        )
    
    dataset = CacheDataset(
        data=data_list,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    
    return dataset

def get_class_names() -> List[str]:
    """Return the list of class names in order."""
    return CLASS_NAMES.copy()

def get_num_classes() -> int:
    """Return the number of classification classes."""
    return len(CLASS_NAMES)
