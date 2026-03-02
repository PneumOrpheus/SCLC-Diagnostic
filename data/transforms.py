"""
MONAI Transforms for SCLC Classification
-----------------------------------------
Custom and composed transforms for CT scan preprocessing using MONAI's transform framework.
"""

import numpy as np
import torch
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
)


class ExtractMiddleSliced(MapTransform):
    """Extract the middle 2D slice from a 3D volume along the specified axis.
    
    For volumes with shape (C, D, H, W), extracts middle slice to get (C, H, W).
    """
    
    def __init__(
        self, 
        keys: KeysCollection,
        axis: int = 1,  # After channel-first: axis 1 is depth
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.axis = axis
    
    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            volume = d[key]
            if hasattr(volume, 'ndim') and volume.ndim >= 3:
                # For channel-first data (C, D, H, W), depth is axis 1
                depth_axis = self.axis
                if volume.ndim == 4:
                    depth_axis = 1  # (C, D, H, W)
                elif volume.ndim == 3:
                    depth_axis = 0 if volume.shape[0] > volume.shape[-1] else 1
                    
                mid_idx = volume.shape[depth_axis] // 2
                if isinstance(volume, np.ndarray):
                    d[key] = np.take(volume, mid_idx, axis=depth_axis)
                else:
                    d[key] = volume.select(depth_axis, mid_idx)
        return d


class ApplyWindowingd(MapTransform):
    """Apply CT windowing to enhance contrast for specific tissues.
    
    Windowing formula maps HU values to [0, 1] range based on window center and width.
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        window_center: float,
        window_width: float,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.window_center = window_center
        self.window_width = window_width
    
    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            volume = d[key]
            img_min = self.window_center - (self.window_width / 2)
            img_max = self.window_center + (self.window_width / 2)
            
            windowed = np.clip(volume, img_min, img_max)
            d[key] = ((windowed - img_min) / (img_max - img_min)).astype(np.float32)
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
            d["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        if "labels" not in d:
            d["labels"] = torch.zeros((0,), dtype=torch.int64)
        if "scan_label" not in d:
            d["scan_label"] = torch.tensor(0, dtype=torch.int64)
        return d


def get_train_transforms(
    img_size: int = 224,
    convert_to_rgb: bool = True,
    use_multichannel_windowing: bool = False
) -> Compose:
    """Get the training transforms pipeline.
    
    Args:
        img_size: Target image size (height and width).
        convert_to_rgb: Whether to convert grayscale to 3-channel RGB.
        use_multichannel_windowing: Whether to use multi-channel CT windowing
            instead of simple RGB conversion.
    
    Returns:
        Composed MONAI transforms for training.
    """
    transforms = [
        # Load image from file path
        LoadImaged(keys=["image"], ensure_channel_first=True, image_only=True),
        
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
    
    # Extract middle slice for 3D volumes
    transforms.append(ExtractMiddleSliced(keys=["image"]))
    
    # Ensure channel-first format
    transforms.append(EnsureChannelFirstd(keys=["image"]))
    
    # Convert to RGB if needed
    if convert_to_rgb and not use_multichannel_windowing:
        transforms.append(EnsureRGBd(keys=["image"]))
    
    # Resize to model input size
    transforms.append(
        Resized(keys=["image"], spatial_size=(img_size, img_size), mode="bilinear")
    )
    
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
    """Get the validation transforms pipeline.
    
    Same as training transforms but without augmentations.
    
    Args:
        img_size: Target image size (height and width).
        convert_to_rgb: Whether to convert grayscale to 3-channel RGB.
        use_multichannel_windowing: Whether to use multi-channel CT windowing.
    
    Returns:
        Composed MONAI transforms for validation.
    """
    return get_train_transforms(
        img_size=img_size,
        convert_to_rgb=convert_to_rgb,
        use_multichannel_windowing=use_multichannel_windowing
    )
