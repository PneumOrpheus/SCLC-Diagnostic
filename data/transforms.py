import numpy as np
import torch
import torch.nn.functional as F
"""
MONAI Transforms for SCLC Classification
-----------------------------------------
Custom and composed transforms for CT scan preprocessing using MONAI's transform framework.
"""

import nibabel as nib
from typing import Any, Dict, Hashable, Mapping, Optional, Union

from monai.config import KeysCollection  # type: ignore[attr-defined]
from monai.transforms import (
    AsDiscreted,
    ScaleIntensityd,
    ResampleToMatchd,
    ConcatItemsd,  # type: ignore[attr-defined]
    MapTransform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
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
    Orientationd,
    Spacingd,
    CropForegroundd,
    DeleteItemsd,
)
from monai.data import MetaTensor


class LoadNiftiWithRGBSupportd(MapTransform):
    """Load NIfTI files with support for RGB structured dtypes and 4D volumes."""
    
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
            # mmap=False drastically improves stability when dealing with thousands of zipped files
            # on multiple workers by forcing it directly into memory immediately 
            img = nib.load(filepath, mmap=False)
            
            # 1. Grab the affine matrix from the NiBabel object
            affine = img.affine 
            
            # Check if it's an RGB structured dtype bypassing properties
            is_rgb = hasattr(img.dataobj, "dtype") and hasattr(img.dataobj.dtype, "names") and img.dataobj.dtype.names is not None and ('R' in img.dataobj.dtype.names or set(img.dataobj.dtype.names) == {'R', 'G', 'B'})
            
            if is_rgb:
                # Convert RGB to grayscale using standard luminance weights
                raw_rgb = np.asanyarray(img.dataobj)
                r = raw_rgb['R'].astype(np.float32)
                g = raw_rgb['G'].astype(np.float32)
                b = raw_rgb['B'].astype(np.float32)
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                # Scale to CT HU-like range (-1024 to 3071) from 0-255
                arr = (gray / 255.0) * 4095 - 1024
            else:
                # Use get_fdata() directly to force total decompression right now on the CPU thread, 
                # rather than yielding proxy views that crash later on inside Spacingd PyTorch loops
                arr = img.get_fdata(dtype=np.float32)
            
            # Handle extra dimensions correctly
            while arr.ndim > 3:
                # Find dimensions of size 1 and squeeze them
                squeezed = False
                for ax in range(arr.ndim - 1, 2, -1):
                    if arr.shape[ax] == 1:
                        arr = arr.squeeze(axis=ax)
                        squeezed = True
                        break
                
                # If we couldn't squeeze any more 1s, check if the last dimension is RGB (size 3)
                if not squeezed:
                    if arr.shape[-1] == 3:
                        # Convert to grayscale
                        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                        gray = 0.299 * r + 0.587 * g + 0.114 * b
                        # Keep it scaled if it seems like a pseudo-CT RGB or just 0-255 RGB
                        # We'll just map roughly back to HU space if max isn't already huge
                        if arr.max() <= 256:
                            arr = (gray / 255.0) * 4095 - 1024
                        else:
                            arr = gray
                    elif arr.ndim >= 4:
                        # If it's a 4D volume (like time series), just take the first frame
                        arr = arr[..., 0]
                    else:
                        break

            
            # Ensure we have a valid 3D volume (at least 3 slices)
            if arr.ndim == 3 and arr.shape[2] < 3:
                # Repeat the few slices to avoid edge cases  
                reps = (3 // arr.shape[2]) + 1
                arr = np.repeat(arr, reps, axis=2)
            
            # 2. Wrap the numpy array into a MetaTensor with spatial metadata
            d[key] = MetaTensor(arr, affine=affine)
            
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
            d["scan_label"] = -1
        return d


class ExtractSubVolumed(MapTransform):
    """Extract a fixed number of slices from a 3D volume based on actual mask presence."""

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
        
        # Determine center Z from mask if present
        target_z_center = None
        if "mask" in d:
            mask_tensor = d["mask"]
            if isinstance(mask_tensor, torch.Tensor):
                nz = torch.nonzero(mask_tensor > 0)
                if nz.numel() > 0:
                    z_coords = nz[:, -1]
                    z_min = z_coords.min().item()
                    z_max = z_coords.max().item()
                    target_z_center = int((z_min + z_max) / 2)
            else:
                nz = np.nonzero(mask_tensor > 0)
                # nz is tuple of arrays per dimension
                if len(nz[-1]) > 0:
                    z_min = nz[-1].min()
                    z_max = nz[-1].max()
                    target_z_center = int((z_min + z_max) / 2)

        for key in self.key_iterator(d):
            volume = d[key]
            if not hasattr(volume, 'ndim') or volume.ndim < 4:
                continue

            # volume is (C, X, Y, Z) (or C, H, W, D). Depth is last axis
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
            else:
                pad_size = target - depth
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                
                if isinstance(volume, np.ndarray):
                    d[key] = np.pad(volume, ((0,0), (0,0), (0,0), (pad_before, pad_after)), mode='constant')
                else:
                    d[key] = F.pad(volume, (pad_before, pad_after), mode='constant')
                
        return d

def _build_lung_crop_transforms(
    img_keys: list,
    spacing_modes: list,
):
    """Crop spatially to the algorithmic lung mask + a generous margin.

    The mask in /home/data/TrainingData is auto-generated lung-chamber
    segmentation, not a tumor mask, and is not perfectly tight. SCLC frequently
    sits in the mediastinum (between the lungs) and at the apex; both are
    mostly inside an axis-aligned bbox over the two lungs, but to absorb mask
    errors and keep peri-lung context we add a large margin.

    Margin is in voxels at the post-Spacingd resolution (1.5×1.5×2.0 mm), so
    (30, 30, 20) ≈ 45×45×40 mm of context beyond the lung bbox in R/L, A/P
    and I/S respectively.
    """
    crop_keys = list(img_keys) + ["lung_mask"]
    crop_modes = list(spacing_modes) + ["nearest"]
    return [
        # Load and align the lung mask through the same spatial pipeline as
        # the CT (and tumor mask / PET if present), so its bbox lines up
        # voxel-for-voxel with the data we'll crop.
        LoadNiftiWithRGBSupportd(keys=["lung_mask"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["lung_mask"], channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=["lung_mask"], axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=["lung_mask"], pixdim=(1.5, 1.5, 2.0),
            mode=["nearest"], allow_missing_keys=True,
        ),
        CropForegroundd(
            keys=crop_keys,
            source_key="lung_mask",
            select_fn=lambda x: x > 0.5,
            margin=(30, 30, 20),
            allow_smaller=True,
            allow_missing_keys=True,
        ),
        # Drop the lung mask once it's served its purpose; the model never
        # consumes it, and keeping it would waste cache + collate memory.
        DeleteItemsd(keys=["lung_mask"]),
    ]


def get_train_transforms_3d(
    img_size: int = 224,
    depth_size: int = 64,
    use_pet: bool = False,
    use_lung_crop: bool = False,
) -> Compose:
    load_keys = ["image", "mask"]
    if use_pet:
        load_keys.append("pet")

    val_keys = list(load_keys)
    spacing_modes = ["bilinear", "nearest"] + (["bilinear"] if use_pet else [])

    transforms = [
        LoadNiftiWithRGBSupportd(keys=load_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),

        # 1. Standardize Orientation
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),

        # 2. Resample PET to perfectly match CT's affine coordinate grid
        *( [ResampleToMatchd(keys="pet", key_dst="image")] if use_pet else [] ),

        # 3. Standardize Physical Voxel Spacing (example: 1.5mm x 1.5mm x 2.0mm)
        Spacingd(
            keys=val_keys,
            pixdim=(1.5, 1.5, 2.0),
            mode=spacing_modes,
            allow_missing_keys=True
        ),

        # 3b. Lung-bbox crop (BigLunge only). Done before intensity scaling so
        # the original HU values still flow through ScaleIntensityRanged below.
        *( _build_lung_crop_transforms(val_keys, spacing_modes) if use_lung_crop else [] ),

        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
        *( [ScaleIntensityd(keys=["pet"], minv=0.0, maxv=1.0)] if use_pet else [] ),
        AsDiscreted(keys=["mask"], threshold=0.5, allow_missing_keys=True),

        ExtractSubVolumed(keys=val_keys, num_slices=depth_size, allow_missing_keys=True),
        
        # 4. Strict spatial sizes applied to both train and val
        Resized(
            keys=val_keys, 
            spatial_size=(img_size, img_size, depth_size), 
            mode=["trilinear", "nearest"] + (["trilinear"] if use_pet else []),
            allow_missing_keys=True
        ),
        
        RandFlipd(keys=val_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
        RandFlipd(keys=val_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
        RandFlipd(keys=val_keys, prob=0.5, spatial_axis=2, allow_missing_keys=True),

        # Round 3: stronger affine to synthesize more anatomic diversity than the
        # ~26 SCLC train patients naturally provide. Applied to image/mask/pet
        # jointly so spatial correspondence stays intact.
        RandAffined(
            keys=val_keys,
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),       # ~5.7 deg on each axis
            translate_range=(8, 8, 4),
            scale_range=(0.1, 0.1, 0.1),
            mode=["bilinear", "nearest"] + (["bilinear"] if use_pet else []),
            padding_mode="zeros",
            allow_missing_keys=True,
        ),

        # Intensity augmentations: raised prob from 0.3 → 0.5 per Round 3 plan.
        RandScaleIntensityd(keys=["image"] + (["pet"] if use_pet else []), factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"] + (["pet"] if use_pet else []), offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"] + (["pet"] if use_pet else []), prob=0.3, mean=0.0, std=0.01),

        # Round 3 included RandCoarseDropoutd here. Removed in Round 4: on sparse
        # tumors a 20x20x10 cutout can erase the lesion outright, which was the
        # main suspect for why Round 3 under-fit (train F1 stuck at ~0.44).

        NormalizeIntensityd(keys=["image"] + (["pet"] if use_pet else []), nonzero=True, channel_wise=True),
        
        *( [ConcatItemsd(keys=["image", "pet"], name="image", dim=0)] if use_pet else [] ),
        AddPlaceholderTargetsd(keys=["image"]),
        ToTensord(keys=["image", "mask"], allow_missing_keys=True),
    ]
    return Compose(transforms)


def get_val_transforms_3d(
    img_size: int = 224,
    depth_size: int = 64,
    use_pet: bool = False,
    use_lung_crop: bool = False,
) -> Compose:
    load_keys = ["image", "mask"]
    if use_pet:
        load_keys.append("pet")

    val_keys = list(load_keys)
    spacing_modes = ["bilinear", "nearest"] + (["bilinear"] if use_pet else [])

    transforms = [
        LoadNiftiWithRGBSupportd(keys=load_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),

        # 1. Standardize Orientation (Must match train)
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),

        # Resample PET to perfectly match CT's affine coordinate grid
        *( [ResampleToMatchd(keys="pet", key_dst="image")] if use_pet else [] ),

        # 2. Standardize Physical Voxel Spacing (Must match train)
        Spacingd(
            keys=val_keys,
            pixdim=(1.5, 1.5, 2.0),
            mode=spacing_modes,
            allow_missing_keys=True
        ),

        # 2b. Lung-bbox crop (BigLunge only).
        *( _build_lung_crop_transforms(val_keys, spacing_modes) if use_lung_crop else [] ),

        # Scale CT intensity to [0, 1] AND clip outliers
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        
        # Optional: scale PET as well (typically PET includes very high values)
        *( [ScaleIntensityd(keys=["pet"], minv=0.0, maxv=1.0)] if use_pet else [] ),

        # Add allow_missing_keys=True here!
        AsDiscreted(keys=["mask"], threshold=0.5, allow_missing_keys=True),

        # Add allow_missing_keys=True here!
        ExtractSubVolumed(keys=val_keys, num_slices=depth_size, allow_missing_keys=True),
        
        # Add allow_missing_keys=True here!
        Resized(
            keys=val_keys, 
            spatial_size=(img_size, img_size, depth_size), 
            mode=["trilinear", "nearest"] + (["trilinear"] if use_pet else []),
            allow_missing_keys=True
        ),
        
        NormalizeIntensityd(keys=["image"] + (["pet"] if use_pet else []), nonzero=True, channel_wise=True),
        
        *( [ConcatItemsd(keys=["image", "pet"], name="image", dim=0)] if use_pet else [] ),
        AddPlaceholderTargetsd(keys=["image"]),
        
        # Add allow_missing_keys=True here!
        ToTensord(keys=["image", "mask"], allow_missing_keys=True),
    ]
    return Compose(transforms)
