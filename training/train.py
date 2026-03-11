import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Sequence, Optional

from monai.data import CacheDataset, DataLoader, list_data_collate  # type: ignore[attr-defined]
from monai.transforms import Compose  # type: ignore[attr-defined]

# MONAI transforms for preprocessing
from data.transforms import get_train_transforms, get_val_transforms
from models.model_selection import focal_loss

"""
SCLC Diagnostic System Training
-------------------------------
Implements the training pipeline for the SCLC diagnostic system with support for:
- Backbone model selection
- Resuming from checkpoints for fine-tuning
- Multi-task loss aggregation
- MONAI CacheDataset for efficient data loading
"""


def sclc_collate_fn(batch: List[Dict[str, Any]]) -> tuple:
    """Custom collate function for detection-style training.
    
    Converts MONAI's dictionary batch format to the (scans, targets) tuple format
    expected by the detection model.
    
    Args:
        batch: List of dictionaries with 'image' and target keys.
        
    Returns:
        Tuple of (scans, targets) where scans is a list of tensors and
        targets is a list of dictionaries.
    """
    scans = []
    targets = []
    
    for i, item in enumerate(batch):
        scans.append(item["image"])

        # Ensure scan_label is a tensor
        scan_label = item.get("scan_label", 0)
        if not isinstance(scan_label, torch.Tensor):
            scan_label = torch.tensor(scan_label, dtype=torch.int64)

        target = {
            "boxes": item.get("boxes", torch.zeros((0, 4), dtype=torch.float32)),
            "labels": item.get("labels", torch.zeros((0,), dtype=torch.int64)),
            "scan_label": scan_label,
            "scan_id": torch.tensor(i, dtype=torch.int64),
        }
        targets.append(target)
    
    return tuple(scans), tuple(targets)


def load_patient_annotations(
    annotation_dir: str,
    patient_short_id: str,
    class_map: Dict[str, int],
    orig_size: int = 512,
    target_size: int = 224
) -> Dict[str, Any]:
    """Load and aggregate bounding box annotations for a patient from XML files.
    
    Annotations are per-slice PASCAL VOC XMLs at original DICOM resolution (512x512).
    We aggregate all per-slice boxes into a union bounding box and scale to target size.
    
    Args:
        annotation_dir: Root annotation directory (e.g. /home/data/Annotation).
        patient_short_id: Short patient ID (e.g. 'A0001').
        class_map: Dict mapping class letters to class indices.
        orig_size: Original annotation coordinate space (512).
        target_size: Target image size (224).
        
    Returns:
        Dict with 'boxes' (tensor [N,4]) and 'labels' (tensor [N]) or empty tensors.
    """
    patient_annot_dir = os.path.join(annotation_dir, patient_short_id)
    if not os.path.isdir(patient_annot_dir):
        return {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)}
    
    xml_files = glob.glob(os.path.join(patient_annot_dir, "*.xml"))
    if not xml_files:
        return {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)}
    
    # Collect all bounding boxes across slices
    all_boxes = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None:
                    continue
                class_letter = name_elem.text.strip()
                if class_letter not in class_map:
                    continue
                class_idx = class_map[class_letter]
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                all_boxes.append((xmin, ymin, xmax, ymax, class_idx))
        except (ET.ParseError, AttributeError):
            continue
    
    if not all_boxes:
        return {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)}
    
    # Compute union bounding box across all slices (tumor spans multiple slices)
    all_xmin = min(b[0] for b in all_boxes)
    all_ymin = min(b[1] for b in all_boxes)
    all_xmax = max(b[2] for b in all_boxes)
    all_ymax = max(b[3] for b in all_boxes)
    # Use the most common class label
    class_labels = [b[4] for b in all_boxes]
    most_common_label = max(set(class_labels), key=class_labels.count)
    
    # Scale from original annotation space (512x512) to target size (224x224)
    scale = target_size / orig_size
    scaled_box = torch.tensor(
        [[all_xmin * scale, all_ymin * scale, all_xmax * scale, all_ymax * scale]],
        dtype=torch.float32
    )
    # Clamp to valid range
    scaled_box = scaled_box.clamp(min=0, max=target_size)
    
    # Detection labels are 1-indexed (0 = background for Faster R-CNN)
    det_label = torch.tensor([most_common_label + 1], dtype=torch.int64)
    
    return {"boxes": scaled_box, "labels": det_label}


def get_data_list(
    data_path: str, 
    split: str = "train", 
    val_frac: float = 0.1, 
    test_frac: float = 0.1, 
    seed: int = 42,
    annotation_dir: str = ""
) -> List[Dict[str, Any]]:
    """Create a list of data dictionaries for MONAI dataset with patient-level splitting.
    
    Args:
        data_path: Path to directory containing scan files.
        split: One of 'train', 'val', or 'test'.
        val_frac: Fraction of patients to use for validation.
        test_frac: Fraction of patients to use for testing.
        seed: Random seed for reproducibility.
        annotation_dir: Path to annotation directory (e.g. /home/data/Annotation).
            If provided, bounding boxes will be loaded from XML files.
        
    Returns:
        List of dictionaries with 'image' key pointing to file paths.
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path '{data_path}' does not exist or is not a directory.")
    
    try:
        all_files = os.listdir(data_path)
    except OSError as e:
        raise ValueError(f"Unable to list contents of data path '{data_path}': {e}") from e
    
    # Collect all valid samples for NIfTI and numpy formats
    # TODO: Implement logic to prefer _Eq files if both versions exist
    valid_extensions = ('.nii.gz', '.nii', '.npy', '.npz')
    samples = [f for f in all_files if any(f.endswith(ext) for ext in valid_extensions)]
    
    if not samples:
        raise ValueError(
            f"No valid data files found in data path '{data_path}'. "
            f"Supported formats: {valid_extensions}"
        )
    
    # A: Adenocarcinoma (0), B: Small Cell Carcinoma (1), G: Squamous Cell Carcinoma (2)
    class_map = {'A': 0, 'B': 1, 'G': 2}

    # Filename format: Lung_Dx-A0126_1.3.6.1... -> Patient ID: Lung_Dx-A0126
    patient_files = {}
    for f in samples:
        parts = f.split('_')
        if len(parts) >= 2:
            # Reconstruct "Lung_Dx-A0126"
            patient_id = f"{parts[0]}_{parts[1]}"
            if patient_id not in patient_files:
                patient_files[patient_id] = []
            patient_files[patient_id].append(f)

    all_patients = sorted(list(patient_files.keys()))
    
    # Shuffle patients deterministically
    rng = np.random.default_rng(seed)
    rng.shuffle(all_patients)
    
    # Calculate split indices
    n_total = len(all_patients)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val
    
    if split == 'train':
        selected_patients = set(all_patients[:n_train])
    elif split == 'val':
        selected_patients = set(all_patients[n_train:n_train+n_val])
    elif split == 'test':
        selected_patients = set(all_patients[n_train+n_val:])
    else:
        # Fallback to all if split name is unknown
        selected_patients = set(all_patients)

    print(f"Data Split '{split}': {len(selected_patients)} patients.")

    # Create list of dictionaries for MONAI
    data_list = []
    
    # Cache annotation lookups per patient short ID
    annotation_cache = {}
    
    for pid in selected_patients:
        files = patient_files.get(pid, [])
        for f in files:
            # Determine label from filename (e.g. Lung_Dx-A0126_...)
            label = -1
            for key, val in class_map.items():
                if f"-{key}" in f:
                    label = val
                    break
            
            # Adds to data list with image path and class label
            if label != -1:
                entry = {
                    "image": os.path.join(data_path, f),
                    "scan_label": label
                }
                
                # Load bounding box annotations if annotation_dir is provided
                if annotation_dir and os.path.isdir(annotation_dir):
                    # Extract short patient ID: "Lung_Dx-A0001" -> "A0001"
                    short_id = pid.split("-")[-1] if "-" in pid else pid
                    if short_id not in annotation_cache:
                        annotation_cache[short_id] = load_patient_annotations(
                            annotation_dir, short_id, class_map
                        )
                    annot = annotation_cache[short_id]
                    entry["boxes"] = annot["boxes"]
                    entry["labels"] = annot["labels"]
                
                data_list.append(entry)
    
    # Log annotation statistics
    if annotation_dir:
        n_with_boxes = sum(1 for d in data_list if d.get("boxes") is not None and d["boxes"].shape[0] > 0)
        print(f"  -> {len(data_list)} images found for split '{split}' ({n_with_boxes} with annotations).")
    else:
        print(f"  -> {len(data_list)} images found for split '{split}'.")
    return data_list


def create_dataset(
    data_path: str,
    split: str = "train",
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False,
    cache_rate: float = 1.0,
    num_workers: int = 4,
    annotation_dir: str = ""
) -> CacheDataset:
    """Create a MONAI CacheDataset.
    
    Args:
        data_path: Path to directory containing scan files.
        split: One of 'train', 'val', or 'test'.
        img_size: Target image size (height and width).
        convert_to_rgb: Whether to convert grayscale to 3-channel RGB.
        use_multichannel_windowing: Whether to use multi-channel CT windowing.
        cache_rate: Fraction of data to cache (0.0 to 1.0).
        num_workers: Number of workers for caching.
        annotation_dir: Path to annotation directory for bounding box loading.
        
    Returns:
        CacheDataset configured for the specified split.
    """
    data_list = get_data_list(data_path, split=split, annotation_dir=annotation_dir)
    
    if split == "train":
        transforms = get_train_transforms(
            img_size=img_size,
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing
        )
    else:
        # val and test use val transforms (no augmentation)
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


def train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                use_mixup=False, mixup_alpha=0.2):
    """Train one epoch with optional Mixup augmentation.
    
    Args:
        model: The model to train.
        optimizer: Optimizer.
        data_loader: Training data loader.
        device: Device to train on.
        epoch: Current epoch number.
        print_freq: Print frequency.
        use_mixup: Whether to apply Mixup data augmentation.
        mixup_alpha: Beta distribution parameter for Mixup.
    """
    model.train()
    running_loss = 0.0
    running_det_loss = 0.0
    running_global_loss = 0.0
    
    num_batches = len(data_loader)
    if num_batches == 0:
        print("Warning: Train data loader is empty.")
        return {"loss": 0.0, "det_loss": 0.0, "global_loss": 0.0}

    for i, (scans, targets) in enumerate(data_loader):
        scans = list(scan.to(device) for scan in scans)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        batch_size = len(scans)
        apply_mixup = use_mixup and batch_size > 1 and np.random.random() < 0.5
        
        if apply_mixup:
            # Blend pairs of images and compute mixed loss
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size)
            
            mixed_scans = [lam * scans[j] + (1 - lam) * scans[index[j]] for j in range(batch_size)]
            
            # Forward with mixed images, request logits for proper mixup loss
            loss_dict = model(mixed_scans, targets, return_logits=True)
            
            global_logits = loss_dict.pop("global_logits")
            _ = loss_dict.pop("global_classification_loss")
            
            # Compute proper mixup focal loss
            gt_a = torch.stack([targets[j]["scan_label"] for j in range(batch_size)])
            gt_b = torch.stack([targets[index[j]]["scan_label"] for j in range(batch_size)])
            
            class_w = model.class_weights if hasattr(model, 'class_weights') else None
            global_loss = (
                lam * focal_loss(global_logits, gt_a, alpha=class_w) +
                (1 - lam) * focal_loss(global_logits, gt_b, alpha=class_w)
            )
        else:
            # Standard forward pass
            loss_dict = model(scans, targets)
            global_loss = loss_dict.pop("global_classification_loss")

        # Ensure detection loss is always a tensor on the correct device
        if loss_dict:
            detection_losses = []
            for loss in loss_dict.values():
                if isinstance(loss, torch.Tensor):
                    detection_losses.append(loss.to(device))
                else:
                    detection_losses.append(torch.as_tensor(loss, device=device))
            loss_detection = sum(detection_losses, torch.zeros((), device=device))
        else:
            loss_detection = torch.zeros((), device=device)
        
        # Weighted sum - global classification is the primary task
        total_loss = loss_detection + global_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        running_det_loss += loss_detection.item()
        running_global_loss += global_loss.item()
        
        if i % print_freq == 0:
            extra = " [mixup]" if apply_mixup else ""
            print(f"Epoch [{epoch}], Iteration [{i}/{len(data_loader)}]{extra}, "
                  f"Total loss: {total_loss.item():.4f}, "
                  f"Detection loss: {loss_detection.item():.4f}, "
                  f"Global loss: {global_loss.item():.4f}")

    metrics = {
        "loss": running_loss / num_batches,
        "det_loss": running_det_loss / num_batches,
        "global_loss": running_global_loss / num_batches
    }
    return metrics


@torch.no_grad()
def validate_epoch(model, data_loader, device, phase="val"):
    """
    Computes validation loss. 
    NOTE: Sets model to train() mode with no_grad() because standard torchvision 
    detection models only return losses in train mode (and predictions in eval mode).
    """
    # Preserve original training state 
    was_training = model.training
    model.train() 
    
    running_loss = 0.0
    running_det_loss = 0.0
    running_global_loss = 0.0
    
    num_batches = len(data_loader)
    if num_batches == 0:
        print(f"Warning: {phase} data loader is empty.")
        model.train(was_training)
        return {"loss": 0.0, "det_loss": 0.0, "global_loss": 0.0}
    
    print(f"Starting {phase} evaluation...")
    
    for scans, targets in data_loader:
        scans = list(scan.to(device) for scan in scans)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(scans, targets)
        
        global_loss = loss_dict.pop("global_classification_loss")

        if loss_dict:
            detection_losses = [l.to(device) if isinstance(l, torch.Tensor) else torch.as_tensor(l, device=device) for l in loss_dict.values()]
            loss_detection = sum(detection_losses, torch.zeros((), device=device))
        else:
            loss_detection = torch.zeros((), device=device)
            
        total_loss = loss_detection + 1.0 * global_loss
        
        running_loss += total_loss.item()
        running_det_loss += loss_detection.item()
        running_global_loss += global_loss.item()
        
    avg_loss = running_loss / num_batches
    avg_det = running_det_loss / num_batches
    avg_global = running_global_loss / num_batches
    
    print(f"  {phase.capitalize()} Loss: {avg_loss:.4f} (Det: {avg_det:.4f}, Global: {avg_global:.4f})")
    
    # Restore model state
    model.train(was_training)
    
    return {"loss": avg_loss, "det_loss": avg_det, "global_loss": avg_global}
