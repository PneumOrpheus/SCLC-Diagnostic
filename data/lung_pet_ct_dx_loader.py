import os
import glob
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from monai.data import PersistentDataset  # type: ignore[attr-defined]
from monai.transforms import Compose  # type: ignore[attr-defined]
from tqdm import tqdm

from data.transforms import get_train_transforms, get_val_transforms, get_train_transforms_3d, get_val_transforms_3d

"""
Lung-PET-CT-Dx Dataset Loader — uses MONAI PersistentDataset for disk-cached transforms.
First run applies transforms and caches; subsequent runs load from cache instantly.
"""

# A: Adenocarcinoma (0), B: Small Cell Carcinoma (1), G: Squamous Cell Carcinoma (2)
CLASS_MAP = {"A": 0, "B": 1, "G": 2}
CLASS_NAMES = ["Adenocarcinoma", "Small Cell Carcinoma", "Squamous Cell Carcinoma"]


def load_patient_annotations(
    annotation_dir: str,
    patient_short_id: str,
    orig_size: int = 512,
    target_size: int = 224,
) -> Dict[str, torch.Tensor]:
    """Load and aggregate bounding box annotations from per-slice XML files.

    Computes a union bounding box across all slices and scales to target_size.
    """
    patient_annot_dir = os.path.join(annotation_dir, patient_short_id)
    empty = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }

    if not os.path.isdir(patient_annot_dir):
        return empty

    xml_files = glob.glob(os.path.join(patient_annot_dir, "*.xml"))
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
                all_boxes.append((xmin, ymin, xmax, ymax, CLASS_MAP[letter]))
        except (ET.ParseError, AttributeError):
            continue

    if not all_boxes:
        return empty

    # Union bounding box + most common class label
    all_xmin = min(b[0] for b in all_boxes)
    all_ymin = min(b[1] for b in all_boxes)
    all_xmax = max(b[2] for b in all_boxes)
    all_ymax = max(b[3] for b in all_boxes)
    class_labels = [b[4] for b in all_boxes]
    most_common = max(set(class_labels), key=class_labels.count)

    scale = target_size / orig_size
    scaled_box = torch.tensor(
        [[all_xmin * scale, all_ymin * scale, all_xmax * scale, all_ymax * scale]],
        dtype=torch.float32,
    ).clamp(min=0, max=target_size)

    # Detection labels are 1-indexed (0 = background for Faster R-CNN)
    det_label = torch.tensor([most_common + 1], dtype=torch.int64)
    return {"boxes": scaled_box, "labels": det_label}


def get_data_list(
    data_path: str,
    split: str = "train",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    annotation_dir: str = "",
) -> List[Dict[str, Any]]:
    """Build list of {image, scan_label, ...} dicts with patient-level splitting."""
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

    # Filename format: Lung_Dx-A0126_1.3.6.1... -> Patient ID: Lung_Dx-A0126
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

    if split == "train":
        selected = set(all_patients[:n_train])
    elif split == "val":
        selected = set(all_patients[n_train:n_train + n_val])
    elif split == "test":
        selected = set(all_patients[n_train + n_val:])
    else:
        selected = set(all_patients)

    print(f"Split '{split}': {len(selected)} patients.")

    data_list = []
    annotation_cache: Dict[str, Dict[str, torch.Tensor]] = {}

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
                if short_id not in annotation_cache:
                    annotation_cache[short_id] = load_patient_annotations(annotation_dir, short_id)
                annot = annotation_cache[short_id]
                entry["boxes"] = annot["boxes"]
                entry["labels"] = annot["labels"]

            data_list.append(entry)

    if annotation_dir:
        n_with = sum(1 for d in data_list if d.get("boxes") is not None and d["boxes"].shape[0] > 0)
        print(f"  {len(data_list)} images ({n_with} with annotations).")
    else:
        print(f"  {len(data_list)} images.")

    return data_list


def create_lung_pet_ct_dataset(
    data_path: str,
    split: str = "train",
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False,
    cache_dir: Optional[str] = None,
    num_workers: int = 4,
    annotation_dir: str = "",
    use_3d: bool = False,
    depth_size: int = 16,
    warm_cache: bool = False,
    **kwargs: Any,
) -> PersistentDataset:
    """Create a PersistentDataset for Lung-PET-CT-Dx (disk-cached transforms)."""
    data_list = get_data_list(data_path, split=split, annotation_dir=annotation_dir)

    if use_3d:
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
        cache_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "monai_lung_pet_ct",
            f"{mode_key}_img{img_size}_d{depth_size}",
            split,
        )
    os.makedirs(cache_dir, exist_ok=True)
    print(f"PersistentDataset cache_dir='{cache_dir}'")

    ds = PersistentDataset(data=data_list, transform=transforms, cache_dir=cache_dir)

    # Optional eager warm cache (off by default to avoid long startup).
    if warm_cache:
        for i in tqdm(range(len(ds)), desc=f"Caching Lung-PET-CT-Dx [{split}]", unit="img"):
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

    def _list_collate(batch):
        """Simple collate that returns a list of dicts (no stacking)."""
        return batch

    parser = argparse.ArgumentParser(description="Lung-PET-CT-Dx data loading benchmark")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--annotation-dir", type=str, default="")
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
        ds = create_lung_pet_ct_dataset(
            data_path=args.data_path, split=split, img_size=args.img_size,
            cache_dir=args.cache_dir, num_workers=args.num_workers,
            annotation_dir=args.annotation_dir,
        )
        print(f"  Creation     : {time.perf_counter() - t0:.2f}s  ({len(ds)} samples)")

        t0 = time.perf_counter()
        _ = ds[0]
        print(f"  First access : {time.perf_counter() - t0:.4f}s")

        loader = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False, pin_memory=False,
                            collate_fn=_list_collate)

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

# python -m data.lung_pet_ct_dx_loader --data-path /home/data/Lung-PET-CT-Dx --annotation-dir /home/data/Annotation
