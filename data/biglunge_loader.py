import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from monai.data import PersistentDataset  # type: ignore[attr-defined]
from monai.transforms import Compose  # type: ignore[attr-defined]
from tqdm import tqdm

from data.transforms import get_train_transforms, get_val_transforms, get_train_transforms_3d, get_val_transforms_3d

"""
BigLunge Dataset Loader — uses MONAI PersistentDataset for disk-cached transforms.
First run applies transforms and caches; subsequent runs load from cache instantly.
"""

NORWEGIAN_CLASS_MAP = {
    "Adenokarsinom": 0,
    "Småcelletkarsinom": 1,
    "Plateepitelkarsinom": 2,
}

CLASS_NAMES = ["Adenocarcinoma", "Small Cell Carcinoma", "Squamous Cell Carcinoma"]


def load_patient_labels(csv_path: str) -> Dict[int, int]:
    """Load patient ID -> class label mapping from CSV."""
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


def get_biglunge_data_list(
    data_path: str,
    csv_path: str,
    split: str = "train",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build list of {image, scan_label, patient_id} dicts with patient-level splitting."""
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

    if split == "train":
        selected = patient_folders[:n_train]
    elif split == "val":
        selected = patient_folders[n_train:n_train + n_val]
    elif split == "test":
        selected = patient_folders[n_train + n_val:]
    else:
        selected = patient_folders

    print(f"Split '{split}': {len(selected)} patients.")

    data_list = []
    for pid in selected:
        patient_dir = data_root / str(pid)
        if not patient_dir.is_dir():
            continue
        label = patient_labels[pid]
        for nii in patient_dir.glob("*.nii*"):
            if "_label_Lungs_auto" not in nii.name:
                data_list.append({"image": str(nii), "scan_label": label, "patient_id": pid})

    class_counts: Dict[int, int] = {}
    for item in data_list:
        class_counts[item["scan_label"]] = class_counts.get(item["scan_label"], 0) + 1
    print(f"  {len(data_list)} images, class distribution: {class_counts}")

    return data_list


def create_biglunge_dataset(
    data_path: str,
    csv_path: str,
    split: str = "train",
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False,
    cache_dir: Optional[str] = None,
    num_workers: int = 4,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    use_3d: bool = False,
    depth_size: int = 16,
    **kwargs: Any,
) -> PersistentDataset:
    """Create a PersistentDataset for BigLunge (disk-cached transforms)."""
    data_list = get_biglunge_data_list(
        data_path=data_path, csv_path=csv_path,
        split=split, val_frac=val_frac, test_frac=test_frac, seed=seed,
    )

    if use_3d:
        if split == "train":
            transforms = get_train_transforms_3d(img_size=img_size, depth_size=depth_size)
        else:
            transforms = get_val_transforms_3d(img_size=img_size, depth_size=depth_size)
    else:
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

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "monai_biglunge", split)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"PersistentDataset cache_dir='{cache_dir}'")

    ds = PersistentDataset(data=data_list, transform=transforms, cache_dir=cache_dir)

    # Warm cache with progress bar (no-op for already-cached items)
    for i in tqdm(range(len(ds)), desc=f"Caching BigLunge [{split}]", unit="img"):
        ds[i]

    return ds


def get_class_names() -> List[str]:
    return CLASS_NAMES.copy()

def get_num_classes() -> int:
    return len(CLASS_NAMES)


if __name__ == "__main__":
    import argparse
    import time
    from monai.data import DataLoader  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(description="BigLunge data loading benchmark")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"Benchmark | img={args.img_size} batch={args.batch_size} workers={args.num_workers}")
    print(f"{'=' * 60}")

    for split in args.splits:
        print(f"\n--- {split} ---")

        t0 = time.perf_counter()
        ds = create_biglunge_dataset(
            data_path=args.data_path, csv_path=args.csv_path,
            split=split, img_size=args.img_size,
            cache_dir=args.cache_dir, num_workers=args.num_workers,
        )
        print(f"  Creation     : {time.perf_counter() - t0:.2f}s  ({len(ds)} samples)")

        t0 = time.perf_counter()
        _ = ds[0]
        print(f"  First access : {time.perf_counter() - t0:.4f}s")

        loader = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False, pin_memory=True)

        for pass_name in ("1st pass", "2nd pass"):
            n = 0
            t0 = time.perf_counter()
            for _ in loader:
                n += 1
            elapsed = time.perf_counter() - t0
            sps = len(ds) / elapsed if elapsed > 0 else float("inf")
            print(f"  {pass_name}      : {elapsed:.2f}s  ({n} batches, {sps:.1f} samples/s)")

        sample = ds[0]
        print(f"  Shape        : {tuple(sample['image'].shape)}, dtype={sample['image'].dtype}")

    print(f"\n{'=' * 60}\nDone.")

# python -m data.biglunge_loader --data-path /home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1 --csv-path /home/data/BigLunge/patients_parameters.csv
