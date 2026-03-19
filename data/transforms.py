"""
MONAI Transforms for SCLC Classification
-----------------------------------------
Custom and composed transforms for CT scan preprocessing using MONAI's transform framework.
"""

import numpy as np
import torch
import nibabel as nib
from typing import Any, Dict, Hashable, Mapping, Optional, Union

from monai.config import KeysCollection  # type: ignore[attr-defined]
from monai.transforms import (  # type: ignore[attr-defined]
    MapTransform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


class LoadNiftiWithRGBSupportd(MapTransform):
    """Load NIfTI files with support for RGB structured dtypes and 4D volumes.
    
    Handles:
    - Standard 3D NIfTI volumes (X, Y, Z)
    - RGB structured dtypes (datatype 128)
    - 4D volumes (X, Y, Z, T) - takes first time point
    - Squeezes single-slice dimensions
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            filepath = d[key]
            img = nib.load(filepath)
            raw = np.asanyarray(img.dataobj)
            
            # Check if it's an RGB structured dtype
            if raw.dtype.names is not None and set(raw.dtype.names) == {'R', 'G', 'B'}:
                # Convert RGB to grayscale using standard luminance weights
                r = raw['R'].astype(np.float32)
                g = raw['G'].astype(np.float32)
                b = raw['B'].astype(np.float32)
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                # Scale to CT HU-like range (-1024 to 3071) from 0-255
                arr = (gray / 255.0) * 4095 - 1024
            else:
                # Standard NIfTI - get float data
                arr = img.get_fdata().astype(np.float32)
            
            # Handle 4D volumes by taking first time/frame
            if arr.ndim == 4:
                arr = arr[..., 0]
            
            # Squeeze any remaining single dimensions but keep at least 3D for proper slice extraction
            while arr.ndim > 3:
                # Find dimensions of size 1 and squeeze them
                for ax in range(arr.ndim - 1, 2, -1):
                    if arr.shape[ax] == 1:
                        arr = arr.squeeze(axis=ax)
                        break
                else:
                    break
            
            # Ensure we have a valid 3D volume (at least 3 slices)
            if arr.ndim == 3 and arr.shape[2] < 3:
                # Repeat the few slices to avoid edge cases  
                reps = (3 // arr.shape[2]) + 1
                arr = np.repeat(arr, reps, axis=2)
            
            # Add channel dimension (C, X, Y, Z)
            d[key] = arr[np.newaxis, ...]
        return d


class CreateMultiChannelCTd(MapTransform):
    """Create a 3-channel representation using different CT windows.
    
    Creates three channels optimized for different anatomical structures:
    - Lung window (L:-600, W:1500): Nodules and parenchyma
    - Mediastinal window (L:50, W:350): Lymph nodes and soft tissue
    - Bone/Wide window (L:300, W:2000): Chest wall and spine context
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        # Define window parameters: (center, width)
        self.windows = [
            (-600, 1500),   # Lung window
            (50, 350),      # Mediastinal window
            (300, 2000),    # Bone window
        ]
    
    def _apply_windowing(self, volume: np.ndarray, center: float, width: float) -> np.ndarray:
        img_min = center - (width / 2)
        img_max = center + (width / 2)
        windowed = np.clip(volume, img_min, img_max)
        return ((windowed - img_min) / (img_max - img_min)).astype(np.float32)
    
    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            volume = d[key]
            
            # Remove channel dimension if present for windowing
            if hasattr(volume, 'ndim'):
                if volume.ndim == 4 and volume.shape[0] == 1:
                    volume = volume[0]
                elif volume.ndim == 3 and volume.shape[0] == 1:
                    volume = volume[0]
            
            channels = [
                self._apply_windowing(np.asarray(volume), center, width)
                for center, width in self.windows
            ]
            
            # Stack channels: (3, ...) for channel-first format
            d[key] = np.stack(channels, axis=0)
        return d


class EnsureRGBd(MapTransform):
    """Ensure the image has 3 channels (RGB) by repeating grayscale if needed."""
    
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if isinstance(img, torch.Tensor):
                if img.shape[0] == 1:
                    d[key] = img.expand(3, -1, -1)
            else:
                if img.shape[0] == 1:
                    d[key] = np.repeat(img, 3, axis=0)
        return d


class AddPlaceholderTargetsd(MapTransform):
    """Add placeholder detection targets for samples without annotations."""
    
    def __init__(
        self,
        keys: KeysCollection = ("image",),
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        # Add placeholder targets if not present
        if "boxes" not in d:
            d["boxes"] = []
        if "labels" not in d:
            d["labels"] = []
        if "scan_label" not in d:
            d["scan_label"] = 0
        return d


class ExtractSubVolumed(MapTransform):
    """Extract a fixed number of slices from a 3D volume.
    
    If 'boxes' is present in data, centers the crop around the tumor's Z-axis.
    Otherwise, extracts from the center of the volume.
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_slices: int = 64,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_slices = num_slices

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        
        # Figure out the center Z coordinate from the annotations if available
        target_z_center = None
        if "boxes" in d and len(d["boxes"]) > 0:
            # Boxes are [x1, y1, z1, x2, y2, z2]
            boxes = np.array(d["boxes"])
            # Get the overall min and max Z across all bounding boxes
            z_min = boxes[:, 2].min()
            z_max = boxes[:, 5].max()
            # Calculate the middle of the tumor
            target_z_center = int((z_min + z_max) / 2)

        for key in self.key_iterator(d):
            volume = d[key]
            if not hasattr(volume, 'ndim') or volume.ndim < 4:
                continue

            # volume is (C, X, Y, Z) — depth is along the last axis
            depth = volume.shape[-1]
            target = self.num_slices

            if depth >= target:
                if target_z_center is not None:
                    mid = target_z_center
                else:
                    mid = depth // 2
                
                half = target // 2
                start = mid - half
                end = start + target
                
                # Handle edge cases to ensure we always get exactly 'target' slices
                if start < 0:
                    start = 0
                    end = target
                elif end > depth:
                    end = depth
                    start = depth - target
                    
                if isinstance(volume, np.ndarray):
                    d[key] = volume[..., start:end]
                else:
                    d[key] = volume[..., start:end]
                    
                # IMPORTANT: Adjust the box Z-coordinates so they align with the new cropped volume!
                if key == "image" and "boxes" in d and len(d["boxes"]) > 0:
                    # Subtract the start offset from the Z-coordinates
                    adjusted_boxes = np.array(d["boxes"], dtype=np.float32)
                    adjusted_boxes[:, 2] -= start
                    adjusted_boxes[:, 5] -= start
                    
                    d["boxes"] = adjusted_boxes.tolist()
            else:
                # If the volume is thinner than target slices, pad the Z axis
                pad_size = target - depth
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                
                if isinstance(volume, np.ndarray):
                    # volume is (C, X, Y, Z)
                    d[key] = np.pad(volume, ((0,0), (0,0), (0,0), (pad_before, pad_after)), mode='constant')
                else:
                    import torch.nn.functional as F
                    d[key] = F.pad(volume, (pad_before, pad_after), mode='constant')
                    
                # Adjust the box Z-coordinates for the padding
                if key == "image" and "boxes" in d and len(d["boxes"]) > 0:
                    adjusted_boxes = np.array(d["boxes"], dtype=np.float32)
                    adjusted_boxes[:, 2] += pad_before
                    adjusted_boxes[:, 5] += pad_before
                    d["boxes"] = adjusted_boxes.tolist()
                
        return d


def get_train_transforms(
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False
) -> Compose:
    """Get the training transforms pipeline with data augmentation.
    
    Args:
        img_size: Target image size (height and width).
        convert_to_rgb: Whether to convert grayscale to 3-channel RGB.
        use_multichannel_windowing: Whether to use multi-channel CT windowing
            instead of simple RGB conversion.
    
    Returns:
        Composed MONAI transforms for training.
    """
    transforms = [
        # Load image from file path (handles both standard and RGB NIfTI files)
        LoadNiftiWithRGBSupportd(keys=["image"]),
        
        # Clip to valid HU range for CT
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=-1024,
            b_max=3071,
            clip=True,
        ),
    ]
    
    if use_multichannel_windowing:
        # Create 3-channel representation using CT windows
        transforms.append(CreateMultiChannelCTd(keys=["image"]))
    else:
        # Simple intensity scaling to [0, 1]
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=3071,
                b_min=0,
                b_max=1,
                clip=True,
            )
        )
    
    
    # Convert to RGB if needed (also ensures 3 channels)
    if convert_to_rgb and not use_multichannel_windowing:
        transforms.append(EnsureRGBd(keys=["image"]))
    
    # Resize to model input size
    transforms.append(
        Resized(keys=["image"], spatial_size=(img_size, img_size), mode="bilinear")
    )
    
    # --- Data Augmentation (training only) ---
    
    # Spatial augmentation
    transforms.append(RandFlipd(keys=["image"], prob=0.5, spatial_axis=0))
    transforms.append(RandFlipd(keys=["image"], prob=0.5, spatial_axis=1))
    transforms.append(RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)))
    transforms.append(
        RandAffined(
            keys=["image"],
            prob=0.3,
            rotate_range=(0.15,),
            scale_range=(0.1, 0.1),
            translate_range=(10, 10),
            mode="bilinear",
            padding_mode="zeros",
        )
    )
    
    # Intensity augmentation
    transforms.append(RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5))
    transforms.append(RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5))
    transforms.append(RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02))
    transforms.append(RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)))
    
    # Add placeholder targets
    transforms.append(AddPlaceholderTargetsd(keys=["image"]))
    
    # Convert to tensors
    transforms.append(ToTensord(keys=["image"]))
    
    return Compose(transforms)


def get_val_transforms(
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False
) -> Compose:
    """Get the validation/test transforms pipeline (no augmentation).
    
    Args:
        img_size: Target image size (height and width).
        convert_to_rgb: Whether to convert grayscale to 3-channel RGB.
        use_multichannel_windowing: Whether to use multi-channel CT windowing.
    
    Returns:
        Composed MONAI transforms for validation/test.
    """
    transforms = [
        LoadNiftiWithRGBSupportd(keys=["image"]),
        
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=-1024,
            b_max=3071,
            clip=True,
        ),
    ]
    
    if use_multichannel_windowing:
        transforms.append(CreateMultiChannelCTd(keys=["image"]))
    else:
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=3071,
                b_min=0,
                b_max=1,
                clip=True,
            )
        )
    
    transforms.append(ExtractMultiSliced(keys=["image"], num_slices=5, aggregation="max"))
    
    if convert_to_rgb and not use_multichannel_windowing:
        transforms.append(EnsureRGBd(keys=["image"]))
    
    transforms.append(
        Resized(keys=["image"], spatial_size=(img_size, img_size), mode="bilinear")
    )
    
    # No augmentation for val/test
    
    transforms.append(AddPlaceholderTargetsd(keys=["image"]))
    transforms.append(ToTensord(keys=["image"]))
    
    return Compose(transforms)


def get_train_transforms_3d(
    img_size: int = 224,
    depth_size: int = 64,
) -> Compose:
    """Get the 3D training transforms pipeline for volumetric models.

    Keeps the volume as (C, D, H, W) for 3D Swin Transformer input.
    Uses single-channel grayscale (no RGB conversion) on the FULL scan depth.
    """
    transforms = [
        LoadNiftiWithRGBSupportd(keys=["image"]),

        # Scale intensity to [0, 1] AND clip outliers dynamically
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=0,
            b_max=1,
            clip=True,
        ),

        # Extract a contiguous crop of depth_size from the center of the Z-axis
        ExtractSubVolumed(keys=["image"], num_slices=depth_size),
        
        # Resize only X and Y down to img_size 
        Resized(keys=["image"], spatial_size=(img_size, img_size, -1), mode="trilinear"),
        
        # Intensity augmentation
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),

        AddPlaceholderTargetsd(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
    return Compose(transforms)


def get_val_transforms_3d(
    img_size: int = 224,
    depth_size: int = 64,
) -> Compose:
    """Get the 3D validation/test transforms pipeline for volumetric models."""
    transforms = [
        LoadNiftiWithRGBSupportd(keys=["image"]),

        # Scale intensity and clip outliers dynamically
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=0,
            b_max=1,
            clip=True,
        ),

        ExtractSubVolumed(keys=["image"], num_slices=depth_size),

        Resized(keys=["image"], spatial_size=(img_size, img_size, depth_size), mode="trilinear"),
        
        AddPlaceholderTargetsd(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
    return Compose(transforms)
