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
    MapTransform,
    Randomizable,
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
    RandGaussianSmoothd,
    RandCoarseDropoutd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    DeleteItemsd,
    SqueezeDimd,
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


class ExtractSubVolumed(MapTransform):
    """Extract a fixed number of slices from a 3D volume around the tumor.

    When ``mask`` is present, the Z-window is centered on the centroid of
    the LARGEST connected component (same convention the 2D pipeline's
    ``CropAroundTumord`` uses). Multifocal masks — common on BigLunge
    auto-seg, ~80% of patients per ``scripts/audit_multifocal.py`` — would
    otherwise center on the unweighted midpoint between distant lesions and
    catch empty parenchyma; the largest CC is overwhelmingly the dominant
    lesion (primary or bulky met) and metastases share the primary's
    histology, so any tumor tissue suffices for histologic-subtype
    classification.

    Components below ``min_component_voxels`` are treated as auto-seg
    noise. If no component meets the threshold, falls back to the
    centroid-of-all-voxels (legacy behaviour); if the mask is absent or
    empty, falls back to the volume center.
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_slices: int = 64,
        min_component_voxels: int = 50,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_slices = num_slices
        self.min_component_voxels = int(min_component_voxels)

    @staticmethod
    def _largest_cc_z_center(
        mask, min_component_voxels: int,
    ) -> Optional[int]:
        """Return Z centroid of the largest connected component (>= min
        voxels) in a (C, X, Y, Z) or (X, Y, Z) mask. None if mask is
        empty or absent.
        """
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)
        binary = arr > 0.5
        if not binary.any():
            return None
        # Strip channel dim if (C, X, Y, Z); keep (X, Y, Z).
        while binary.ndim > 3:
            binary = binary[0] if binary.shape[0] == 1 else binary.any(axis=0)

        from scipy.ndimage import label as cc_label
        labeled, n = cc_label(binary)
        if n == 0:
            return None

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        valid = sizes >= min_component_voxels
        if not valid.any():
            # Sub-threshold mask — fall back to centroid of all non-zero
            # voxels rather than triggering the volume-center path.
            idx = np.argwhere(binary)
        else:
            masked_sizes = sizes.copy()
            masked_sizes[~valid] = 0
            largest_label = int(masked_sizes.argmax())
            idx = np.argwhere(labeled == largest_label)

        if idx.size == 0:
            return None
        # idx is (N, 3) for (X, Y, Z); Z is last column.
        return int(round(idx[:, -1].mean()))

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)

        # Determine center Z from mask if present
        target_z_center = None
        if "mask" in d:
            target_z_center = self._largest_cc_z_center(
                d["mask"], self.min_component_voxels,
            )

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


class BBoxFromMaskd(MapTransform):
    """Compute axis-aligned bounding boxes from a binary mask.

    Writes ``bbox_key`` and ``has_key`` into the sample dict. For 2D masks
    the box is (xmin, ymin, xmax, ymax); for 3D, (xmin, ymin, zmin, xmax,
    ymax, zmax). Coordinates are normalized to [0, 1] so downstream heads can
    remain resolution-agnostic.
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str = "mask",
        bbox_key: str = "bbox",
        has_key: str = "has_bbox",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.bbox_key = bbox_key
        self.has_key = has_key

    @staticmethod
    def _normalize(vals, shape):
        norm = []
        for v, dim in zip(vals, shape):
            denom = max(float(dim - 1), 1.0)
            norm.append(float(v) / denom)
        return norm

    def _bbox_single(self, mask: np.ndarray):
        # Strip single leading channel dim for both (1,H,W) and (1,H,W,D).
        # Without this, (1,H,W) after SqueezeDimd has ndim=3 and falls into the
        # 3D branch below, producing a 6-element bbox for a 2D slice.
        if mask.ndim >= 3 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim > 3:
            mask = mask.any(axis=0)
        binary = mask > 0.5
        if not binary.any():
            return None
        idx = np.argwhere(binary)
        mins = idx.min(axis=0)
        maxs = idx.max(axis=0)

        if binary.ndim == 2:
            y_min, x_min = mins
            y_max, x_max = maxs
            bbox = [x_min, y_min, x_max, y_max]
            bbox = self._normalize(bbox, (binary.shape[1], binary.shape[0], binary.shape[1], binary.shape[0]))
            return bbox
        # 3D: (H, W, D)
        y_min, x_min, z_min = mins
        y_max, x_max, z_max = maxs
        bbox = [x_min, y_min, z_min, x_max, y_max, z_max]
        bbox = self._normalize(
            bbox,
            (
                binary.shape[1],
                binary.shape[0],
                binary.shape[2],
                binary.shape[1],
                binary.shape[0],
                binary.shape[2],
            ),
        )
        return bbox

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        if self.source_key not in d:
            if not self.allow_missing_keys:
                raise KeyError(f"BBoxFromMaskd: '{self.source_key}' not found on sample.")
            return d

        mask = d.get(self.source_key)
        if mask is None:
            d[self.bbox_key] = None
            d[self.has_key] = False
            return d

        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)

        # Bag case: (N, 1, H, W)
        if arr.ndim == 4 and arr.shape[0] > 1 and arr.shape[1] == 1:
            bboxes = []
            has = []
            for i in range(arr.shape[0]):
                bbox = self._bbox_single(arr[i])
                if bbox is None:
                    bboxes.append([0.0, 0.0, 0.0, 0.0])
                    has.append(False)
                else:
                    bboxes.append(bbox)
                    has.append(True)
            d[self.bbox_key] = np.asarray(bboxes, dtype=np.float32)
            d[self.has_key] = np.asarray(has, dtype=bool)
            return d

        bbox = self._bbox_single(arr)
        if bbox is None:
            d[self.bbox_key] = None
            d[self.has_key] = False
        else:
            d[self.bbox_key] = np.asarray(bbox, dtype=np.float32)
            d[self.has_key] = True
        return d

class CropAroundTumord(MapTransform):
    """Crop a fixed-size 3D patch centered on the LARGEST connected
    component of the tumor mask.

    - ``keys``: image/mask keys to crop identically.
    - ``source_key``: tumor segmentation key used to locate the centroid.
      Must share voxel grid with the image (so Load/Orientation/Spacing first).
    - ``patch_size``: (X, Y, Z) voxels at the current (post-Spacingd) resolution.
    - ``min_component_voxels``: connected components smaller than this are
      treated as auto-seg noise / not eligible to define the centroid.

    **Largest-component cropping rationale.** BigLunge tumor masks are
    algorithmic and frequently produce multiple connected components per
    patient (audit on 2026-04-28: 68-77% of patients are multifocal,
    median 2 components, P95 = 8). Centering the crop on the unweighted
    centroid of all non-zero voxels lands the crop between distant lesions
    — capturing empty lung parenchyma instead of any tumor tissue.

    The largest connected component is overwhelmingly the dominant lesion
    (primary mass or bulky metastasis). For histologic-subtype classification
    that's enough: metastases share the primary's histology by definition, so
    presenting any tumor tissue suffices regardless of whether it's the
    anatomical primary.

    If the tumor mask is empty or missing, falls back to the volume center.
    If components exist but all are below ``min_component_voxels``, falls
    back to the centroid-of-all-voxels behavior (the prior implementation).
    Out-of-bounds regions are zero-padded so the output shape is always exact.

    See ``scripts/audit_multifocal.py`` for the multifocal audit script and
    ``output/multifocal_audit.csv`` for the per-patient CC counts.
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        patch_size: tuple = (96, 96, 16),
        allow_missing_keys: bool = False,
        min_component_voxels: int = 50,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.patch_size = tuple(int(p) for p in patch_size)
        self.min_component_voxels = int(min_component_voxels)

    def _centroid(self, mask) -> Optional[tuple]:
        """Return centroid of the largest connected component (≥ min voxels).

        Steps:
          1. Threshold the mask at 0.5.
          2. Run ``scipy.ndimage.label`` (default 6-connectivity in 3D).
          3. Discard components smaller than ``self.min_component_voxels``.
          4. Pick the largest remaining component; return its centroid.
          5. If no component qualifies, fall back to the unweighted centroid
             of all non-zero voxels (legacy behaviour) so a tiny but-real
             tumor mask still produces a meaningful crop instead of a
             None-trigger fallback to the volume center.
        """
        if mask is None:
            return None

        # Convert to numpy for scipy.ndimage.label. The CC-labeling cost is
        # negligible on per-slice masks (~10-20 us) and one-shot on per-volume
        # 3D masks; the conversion overhead is the dominant term either way.
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)

        binary = arr > 0.5
        if not binary.any():
            return None

        from scipy.ndimage import label as cc_label
        # scipy.ndimage.label on an N-d array uses an ND structuring element;
        # default is 1-connectivity (faces only), which is the conservative
        # choice that doesn't merge spatially-disjoint blobs through
        # diagonal voxels. This is what the audit script uses.
        labeled, n = cc_label(binary)
        if n == 0:
            return None

        # bincount over all label IDs; index 0 is background, drop it.
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        valid = sizes >= self.min_component_voxels
        if not valid.any():
            # No component meets the size threshold — fall back to the
            # centroid of all non-zero voxels rather than returning None
            # (which would trigger the volume-center fallback in __call__,
            # producing a near-useless crop for these edge cases).
            idx = np.argwhere(binary)
        else:
            masked_sizes = sizes.copy()
            masked_sizes[~valid] = 0
            largest_label = int(masked_sizes.argmax())
            idx = np.argwhere(labeled == largest_label)

        if idx.size == 0:
            return None
        # idx is (N, ndim). For 2D-after-SliceSelect the mask shape is
        # (C=1, H, W, Z=1) so idx is (N, 4); for 3D masks (C, H, W, Z) it's
        # also (N, 4). Take the last 3 columns to get spatial (X, Y, Z).
        c = idx[:, -3:].mean(axis=0)
        return int(c[0]), int(c[1]), int(c[2])

    def _crop(self, vol, center):
        px, py, pz = self.patch_size
        cx, cy, cz = center
        X, Y, Z = vol.shape[-3:]
        sx, sy, sz = cx - px // 2, cy - py // 2, cz - pz // 2
        ex, ey, ez = sx + px, sy + py, sz + pz
        pad = [max(0, -sx), max(0, ex - X),
               max(0, -sy), max(0, ey - Y),
               max(0, -sz), max(0, ez - Z)]
        sx_c, sy_c, sz_c = max(0, sx), max(0, sy), max(0, sz)
        ex_c, ey_c, ez_c = min(X, ex), min(Y, ey), min(Z, ez)
        out = vol[..., sx_c:ex_c, sy_c:ey_c, sz_c:ez_c]
        if any(p > 0 for p in pad):
            if isinstance(out, torch.Tensor):
                # F.pad last-dim-first: (z_l, z_r, y_l, y_r, x_l, x_r)
                out = F.pad(out, (pad[4], pad[5], pad[2], pad[3], pad[0], pad[1]))
            else:
                out = np.pad(
                    out,
                    ((0, 0), (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])),
                    mode="constant",
                )
        return out

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        center = self._centroid(d.get(self.source_key))
        if center is None:
            # fallback: geometric center of the first keyed volume
            ref = None
            for k in self.key_iterator(d):
                ref = d[k]
                break
            if ref is None:
                return d
            X, Y, Z = ref.shape[-3:]
            center = (X // 2, Y // 2, Z // 2)

        crop_keys = list(self.key_iterator(d))
        if self.source_key in d and self.source_key not in crop_keys:
            crop_keys.append(self.source_key)
        for k in crop_keys:
            d[k] = self._crop(d[k], center)
        return d


def _build_lung_crop_transforms(
    img_keys: list,
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
    return [
        # Load and align the lung mask through the same spatial pipeline as
        # the CT (and tumor mask if present), so its bbox lines up
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


def _aug_block_3d(val_keys: list, strong_augs: bool) -> list:
    """Affine + intensity augs applied jointly to image + mask.

    The default block is the historical 3D config (rotate ~5.7 deg,
    translate 8 vox, scale 0.1, intensity prob 0.5). The strong-augs
    block matches the spirit of the 2D / MIL strong-augs path: heavier
    affine (rotate ~11.5 deg, translate 12 vox, scale 0.15) and intensity
    perturbations at higher probability, plus RandGaussianSmoothd. We
    leave RandCoarseDropoutd off in 3D — sparse holes in a volumetric
    tumor crop can wipe out the whole signal of a small SCLC primary,
    and we don't have the 2D pipeline's per-slice redundancy to absorb
    it.
    """
    if not strong_augs:
        return [
            RandAffined(
                keys=val_keys,
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),       # ~5.7 deg on each axis
                translate_range=(8, 8, 4),
                scale_range=(0.1, 0.1, 0.1),
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
        ]
    return [
        RandAffined(
            keys=val_keys,
            prob=0.8,
            rotate_range=(0.20, 0.20, 0.20),     # ~11.5 deg on each axis
            translate_range=(12, 12, 6),
            scale_range=(0.15, 0.15, 0.15),
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
            allow_missing_keys=True,
        ),
        RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.7),
        RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.7),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
        ),
    ]


def get_train_transforms_3d(
    img_size: int = 224,
    depth_size: int = 64,
    use_lung_crop: bool = False,
    strong_augs: bool = False,
    include_bbox: bool = False,
) -> Compose:
    load_keys = ["image", "mask"]

    val_keys = list(load_keys)
    spacing_modes = ["bilinear", "nearest"]

    transforms = [
        LoadNiftiWithRGBSupportd(keys=load_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),

        # 1. Standardize Orientation
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),

        # 3. Standardize Physical Voxel Spacing (example: 1.5mm x 1.5mm x 2.0mm)
        Spacingd(
            keys=val_keys,
            pixdim=(1.5, 1.5, 2.0),
            mode=spacing_modes,
            allow_missing_keys=True
        ),

        # 3b. Lung-bbox crop (BigLunge only). Done before intensity scaling so
        # the original HU values still flow through ScaleIntensityRanged below.
        *( _build_lung_crop_transforms(val_keys) if use_lung_crop else [] ),

        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
        # AsDiscreted removed: Spacingd(mode=nearest) already keeps the mask
        # binary, and the subsequent Resized(mode=nearest) preserves that.

        ExtractSubVolumed(keys=val_keys, num_slices=depth_size, allow_missing_keys=True),

        # 4. Strict spatial sizes applied to both train and val
        Resized(
            keys=val_keys,
            spatial_size=(img_size, img_size, depth_size),
            mode=["trilinear", "nearest"],
            allow_missing_keys=True,
        ),

        RandFlipd(keys=val_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
        RandFlipd(keys=val_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
        RandFlipd(keys=val_keys, prob=0.5, spatial_axis=2, allow_missing_keys=True),

        *(_aug_block_3d(val_keys=val_keys, strong_augs=strong_augs)),

        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        *([BBoxFromMaskd(keys=["mask"], source_key="mask", bbox_key="bbox", has_key="has_bbox", allow_missing_keys=True)] if include_bbox else []),
        ToTensord(keys=["image", "mask"], allow_missing_keys=True),
    ]
    return Compose(transforms)


def get_val_transforms_3d(
    img_size: int = 224,
    depth_size: int = 64,
    use_lung_crop: bool = False,
    include_bbox: bool = False,
) -> Compose:
    load_keys = ["image", "mask"]

    val_keys = list(load_keys)
    spacing_modes = ["bilinear", "nearest"]

    transforms = [
        LoadNiftiWithRGBSupportd(keys=load_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),

        # 1. Standardize Orientation (Must match train)
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),

        # 2. Standardize Physical Voxel Spacing (Must match train)
        Spacingd(
            keys=val_keys,
            pixdim=(1.5, 1.5, 2.0),
            mode=spacing_modes,
            allow_missing_keys=True
        ),

        # 2b. Lung-bbox crop (BigLunge only).
        *( _build_lung_crop_transforms(val_keys) if use_lung_crop else [] ),

        # Scale CT intensity to [0, 1] AND clip outliers
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        # AsDiscreted removed: Spacingd(mode=nearest) keeps the mask binary
        # and Resized(mode=nearest) preserves that. The threshold step was
        # redundant.

        ExtractSubVolumed(keys=val_keys, num_slices=depth_size, allow_missing_keys=True),

        Resized(
            keys=val_keys,
            spatial_size=(img_size, img_size, depth_size),
            mode=["trilinear", "nearest"],
            allow_missing_keys=True,
        ),

        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        *([BBoxFromMaskd(keys=["mask"], source_key="mask", bbox_key="bbox", has_key="has_bbox", allow_missing_keys=True)] if include_bbox else []),
        ToTensord(keys=["image", "mask"], allow_missing_keys=True),
    ]
    return Compose(transforms)


class PopKeysd(MapTransform):
    """Remove one or more keys from a sample dict, silently skipping absent keys.

    Drop-in replacement for ``DeleteItemsd`` when the key may not be present
    (``DeleteItemsd`` has no ``allow_missing_keys`` in older MONAI versions).
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for k in self.keys:
            d.pop(k, None)
        return d


class SliceSelectd(MapTransform):
    """Select a single axial slice (keeping a length-1 Z axis) from (C, X, Y, Z)
    volumes. The slice index is read from ``slice_key`` on the sample dict.

    Clamped to the volume's Z extent so entries built from a slightly different
    orientation/spacing don't fail at runtime.
    """

    def __init__(
        self,
        keys: KeysCollection,
        slice_key: str = "slice_idx",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.slice_key = slice_key

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        if self.slice_key not in d:
            raise KeyError(f"SliceSelectd: '{self.slice_key}' not found on sample.")
        idx = int(d[self.slice_key])
        for key in self.key_iterator(d):
            vol = d[key]
            Z = vol.shape[-1]
            clamped = max(0, min(Z - 1, idx))
            d[key] = vol[..., clamped:clamped + 1]
        return d


def _build_2d_pipeline(
    img_size: int,
    train: bool,
    strong_augs: bool = False,
    crop_size: int = 96,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> list:
    """Shared 2D pipeline: load CT + tumor mask, spacing, pick the tumor slice
    specified by ``slice_idx`` on the sample, crop in-plane around the 2D tumor
    bbox in that slice, scale, resize, squeeze Z, augment, normalize.

    ``crop_size`` controls the ``CropAroundTumord`` in-plane patch size in
    post-Spacing voxels (pipeline runs at 1 mm X/Y, so crop_size ≈ diameter
    in mm). Default 96 matches the original 2D DAPT setup on Lung-PET-CT-Dx
    where tumors are typically < 60 mm. BigLunge has larger tumors and needs
    ≥ 160 — see the audit in ``data_exploration/BigLunge_expl.ipynb`` for the
    mechanistic rationale (large-mass overflow when centroid is near the
    slice edge).

    ``strong_augs`` (train-only) swaps the default mild augmentation block for
    the thesis experiment documented in ``docs/2d_augmentations.md`` —
    heavier RandAffine + intensity perturbations plus RandGaussianSmooth and
    RandCoarseDropout. Used to address the train-val gap (train MacroF1
    ~0.86-0.98 vs val ~0.61-0.74) observed across all three 2D backbones.
    Does not invalidate the PersistentDataset cache: MONAI only caches up to
    the last deterministic transform, and all augs here are ``Rand*``.
    """
    load_keys = ["image", "tumor_mask"]
    keep_mask = bool(include_mask or include_bbox)
    # Keys and interpolation modes for spatial transforms that must stay
    # aligned with the image (flip, affine, crop, resize).
    _aug_keys = ["image"] + (["tumor_mask"] if keep_mask else [])
    _aug_spatial_modes = ["bilinear"] + (["nearest"] if keep_mask else [])
    transforms: list = [
        LoadNiftiWithRGBSupportd(keys=load_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=load_keys,
            pixdim=(1.0, 1.0, 2.0),
            mode=["bilinear", "nearest"],
            allow_missing_keys=True,
        ),
        # Pick the single axial slice of interest (volume stays 4D with Z=1 so
        # the downstream 3D-style tumor crop can reuse CropAroundTumord).
        SliceSelectd(keys=load_keys, slice_key="slice_idx", allow_missing_keys=True),
        # Crop image AND mask jointly to the same tumor-centered region so
        # bbox coordinates computed later are in the cropped-image frame.
        CropAroundTumord(
            keys=["image"] + (["tumor_mask"] if keep_mask else []),
            source_key="tumor_mask",
            patch_size=(int(crop_size), int(crop_size), 1),
            allow_missing_keys=True,
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
        # Resize image AND mask jointly so spatial dims remain in sync.
        Resized(
            keys=["image"] + (["tumor_mask"] if keep_mask else []),
            spatial_size=(img_size, img_size, 1),
            mode=["trilinear"] + (["nearest"] if keep_mask else []),
        ),
        # Drop the pseudo-Z axis; keep mask only if requested.
        SqueezeDimd(keys=["image"] + (["tumor_mask"] if keep_mask else []), dim=-1),
    ]

    if not keep_mask:
        transforms.append(DeleteItemsd(keys=["tumor_mask"]))

    if train and not strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=_aug_keys,
                prob=0.5,
                rotate_range=(0.26,),          # ~15 deg in-plane
                translate_range=(8, 8),
                scale_range=(0.1, 0.1),
                mode=_aug_spatial_modes,
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
        ]
    elif train and strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=_aug_keys,
                prob=0.8,                      # was 0.5
                rotate_range=(0.35,),          # ~20 deg in-plane (was ~15)
                translate_range=(12, 12),      # was (8, 8)
                scale_range=(0.15, 0.15),      # was (0.10, 0.10)
                mode=_aug_spatial_modes,
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.7),
            RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.7),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
            RandGaussianSmoothd(
                keys=["image"], prob=0.2,
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
            ),
            RandCoarseDropoutd(
                keys=["image"], holes=3, spatial_size=(24, 24),
                fill_value=0.0, prob=0.3,
            ),
        ]

    # BBoxFromMaskd after augmentation so the bbox reflects the augmented, spatially-aligned mask
    if include_bbox:
        transforms.append(
            BBoxFromMaskd(keys=["tumor_mask"], source_key="tumor_mask", bbox_key="bbox", has_key="has_bbox", allow_missing_keys=True)
        )

    transforms += [
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"] + (["tumor_mask"] if keep_mask else []), allow_missing_keys=True),
    ]
    return transforms


def get_train_transforms_2d(
    img_size: int = 224,
    strong_augs: bool = False,
    crop_size: int = 96,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_2d_pipeline(
        img_size=img_size,
        train=True,
        strong_augs=strong_augs,
        crop_size=crop_size,
        include_mask=include_mask,
        include_bbox=include_bbox,
    ))


def get_val_transforms_2d(
    img_size: int = 224,
    crop_size: int = 96,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_2d_pipeline(
        img_size=img_size,
        train=False,
        crop_size=crop_size,
        include_mask=include_mask,
        include_bbox=include_bbox,
    ))


# =============================================================================
# MIL pipeline: whole-slice DAPT + bag-level BigLunge
# =============================================================================
#
# Two pipelines share a single front-end (load, orient, spacing, HU window,
# resize XY). They differ only in what constitutes a sample:
#   * whole-slice DAPT: one tumor-containing axial slice per sample, full
#     in-plane FOV (no tumor-centered crop).
#   * MIL bag: N evenly-spaced axial slices from the lung-mask z-extent per
#     patient, stacked into a (N, 1, H, W) bag for MILModel.
#
# Design choices:
# - We deliberately drop CropAroundTumord from both so BigLunge inference can
#   be tumor-mask-free. The backbone sees the same visual scale (img_size
#   voxels @ 1 mm) in both phases so DAPT transfer is not fighting a scale
#   shift.
# - Bag samples are permuted (C, H, W, N) -> (N, 1, H, W) at the end so the
#   DataLoader produces (B, N, 1, H, W) directly matching MILModel's input.
# - In-plane augs are applied before permutation so MONAI sees them as a 3D
#   volume; rotate/translate/scale on the bag axis are disabled so different
#   instances in one bag stay registered.


class BagAsBatchDimd(MapTransform):
    """Convert a (C=1, H, W, N) bag volume into (N, 1, H, W).

    The depth axis produced by ``LungAxialBagSelectd`` becomes the MIL bag
    dimension; the (former) channel axis becomes per-instance C=1. Output
    shape matches what ``MILModel`` expects per-sample, so DataLoader stacking
    yields ``(B, N, 1, H, W)`` directly.
    """

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            vol = d[key]
            if isinstance(vol, torch.Tensor):
                if vol.ndim == 4:  # (C, H, W, N)
                    d[key] = vol.permute(3, 0, 1, 2).contiguous()
                elif vol.ndim == 3:  # (H, W, N)
                    d[key] = vol.permute(2, 0, 1).contiguous().unsqueeze(1)
                else:
                    raise ValueError(f"BagAsBatchDimd: unexpected ndim {vol.ndim}")
            else:
                arr = np.asarray(vol)
                if arr.ndim == 4:
                    d[key] = np.ascontiguousarray(np.transpose(arr, (3, 0, 1, 2)))
                elif arr.ndim == 3:
                    d[key] = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[:, None])
                else:
                    raise ValueError(f"BagAsBatchDimd: unexpected ndim {arr.ndim}")
        return d


class LungAxialBagSelectd(Randomizable, MapTransform):
    """Sample ``num_slices`` evenly-spaced axial slices from within the
    ``source_key`` mask's z-extent.

    Expects ``source_key`` and each keyed volume to be channel-first 3D
    (C, H, W, Z) with identical Z extent (i.e. pre-aligned via Orientation +
    Spacing). Operates on post-Spacing grid so the slice indices are stable.

    With ``jitter=True`` the linspace grid is shifted by a uniform random
    offset in ``[-stride/2, +stride/2]`` per call — same offset for every
    keyed volume in one call, so CT and any aux volumes stay registered.
    Off by default: val/test/inference must be deterministic. At train
    time, jitter gives stochastic z-coverage across epochs so tumors that
    happen to fall between two evenly-spaced bag slices get sampled
    eventually instead of being permanently invisible.

    If the mask is empty (or has zero z-extent), falls back to the full Z
    extent of the first keyed volume and prints a warning — silent
    fallback would mean MIL is sampling the abdomen on patients with a
    truncated lung mask. The mask itself is *not* included in the output
    if listed in ``keys`` only to participate in the spacing alignment;
    callers typically delete it immediately after.
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str = "lung_mask",
        num_slices: int = 16,
        jitter: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.source_key = source_key
        self.num_slices = int(num_slices)
        self.jitter = bool(jitter)
        self._offset_frac: float = 0.0

    def randomize(self, data=None) -> None:
        # self.R is the MONAI-managed RandomState; using it keeps seeding
        # consistent with other Rand* transforms in the same Compose.
        self._offset_frac = float(self.R.uniform(-0.5, 0.5)) if self.jitter else 0.0

    @staticmethod
    def _z_extent(mask) -> Optional[tuple]:
        """Return (z_min, z_max) inclusive, or None if mask is empty."""
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            m = (mask > 0.5)
            # Reduce everything except the last axis (Z)
            reduce_axes = tuple(range(m.ndim - 1))
            per_z = m.any(dim=reduce_axes) if reduce_axes else m
            nz = torch.nonzero(per_z, as_tuple=False).flatten()
            if nz.numel() == 0:
                return None
            return int(nz.min().item()), int(nz.max().item())
        arr = np.asarray(mask) > 0.5
        reduce_axes = tuple(range(arr.ndim - 1))
        per_z = arr.any(axis=reduce_axes) if reduce_axes else arr
        nz = np.where(per_z)[0]
        if nz.size == 0:
            return None
        return int(nz.min()), int(nz.max())

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        self.randomize()
        extent = self._z_extent(d.get(self.source_key))

        # Resolve Z from the first keyed volume. Fallback for empty mask.
        ref_vol = None
        for k in self.key_iterator(d):
            ref_vol = d[k]
            break
        if ref_vol is None:
            return d
        Z = int(ref_vol.shape[-1])

        if extent is None:
            print(
                f"[LungAxialBagSelectd] WARNING: empty/missing '{self.source_key}', "
                f"falling back to full Z extent (Z={Z}). Bag will include non-lung slices."
            )
            z_min, z_max = 0, max(0, Z - 1)
        else:
            z_min, z_max = extent
            z_min = max(0, min(Z - 1, z_min))
            z_max = max(0, min(Z - 1, z_max))

        if self.num_slices <= 0:
            raise ValueError(f"num_slices must be >= 1, got {self.num_slices}")

        if z_max > z_min:
            base = np.linspace(z_min, z_max, self.num_slices)
            if self.jitter and self.num_slices > 1:
                stride = (z_max - z_min) / (self.num_slices - 1)
                base = base + (self._offset_frac * stride)
                base = np.clip(base, z_min, z_max)
            idxs = base.round().astype(int).tolist()
        else:
            # Degenerate: single-slice extent. Repeat the index num_slices times.
            idxs = [z_min] * self.num_slices

        # Guarantee every idx is in [0, Z-1]
        idxs = [max(0, min(Z - 1, int(i))) for i in idxs]

        for key in self.key_iterator(d):
            vol = d[key]
            if isinstance(vol, torch.Tensor):
                idx_t = torch.tensor(idxs, dtype=torch.long, device=vol.device)
                d[key] = torch.index_select(vol, dim=-1, index=idx_t).contiguous()
            else:
                d[key] = np.take(np.asarray(vol), indices=idxs, axis=-1)
        return d


def _build_whole_slice_pipeline(
    img_size: int,
    train: bool,
    strong_augs: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> list:
    """Whole-slice per-slice pipeline for DAPT.

    Load CT (tumor mask is referenced in the sample dict but not loaded —
    ``slice_idx`` already encodes which axial slice to pick). Spacing to
    1.0 x 1.0 x 2.0 mm. Select the single tumor slice via ``SliceSelectd``. HU
    window, resize XY to ``img_size`` (no in-plane crop), squeeze Z, augment,
    normalize. Output: ``(C=1, img_size, img_size)``.
    """
    keep_mask = bool(include_mask or include_bbox)
    load_keys = ["image"] + (["tumor_mask"] if keep_mask else [])
    _spacing_modes = ["bilinear"] + (["nearest"] if keep_mask else [])
    _aug_keys = ["image"] + (["tumor_mask"] if keep_mask else [])
    _aug_spatial_modes = ["bilinear"] + (["nearest"] if keep_mask else [])
    transforms: list = [
        LoadNiftiWithRGBSupportd(keys=load_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=load_keys,
            pixdim=(1.0, 1.0, 2.0),
            mode=_spacing_modes,
            allow_missing_keys=True,
        ),
        SliceSelectd(keys=load_keys, slice_key="slice_idx", allow_missing_keys=True),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
        # Resize image AND mask jointly so spatial dims stay in sync.
        Resized(
            keys=["image"] + (["tumor_mask"] if keep_mask else []),
            spatial_size=(img_size, img_size, 1),
            mode=["trilinear"] + (["nearest"] if keep_mask else []),
        ),
        SqueezeDimd(keys=["image"] + (["tumor_mask"] if keep_mask else []), dim=-1),
    ]

    if not keep_mask:
        transforms.append(DeleteItemsd(keys=["tumor_mask"]))

    if train and not strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=_aug_keys,
                prob=0.5,
                rotate_range=(0.26,),
                translate_range=(8, 8),
                scale_range=(0.1, 0.1),
                mode=_aug_spatial_modes,
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
        ]
    elif train and strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=_aug_keys,
                prob=0.8,
                rotate_range=(0.35,),
                translate_range=(12, 12),
                scale_range=(0.15, 0.15),
                mode=_aug_spatial_modes,
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.7),
            RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.7),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
            RandGaussianSmoothd(
                keys=["image"], prob=0.2,
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
            ),
            RandCoarseDropoutd(
                keys=["image"], holes=3, spatial_size=(24, 24),
                fill_value=0.0, prob=0.3,
            ),
        ]

    # BBoxFromMaskd AFTER augmentation so the bbox is in the augmented frame.
    if include_bbox:
        transforms.append(
            BBoxFromMaskd(keys=["tumor_mask"], source_key="tumor_mask", bbox_key="bbox", has_key="has_bbox", allow_missing_keys=True)
        )

    transforms += [
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"] + (["tumor_mask"] if keep_mask else []), allow_missing_keys=True),
    ]
    return transforms


def get_train_transforms_whole_slice(
    img_size: int = 384,
    strong_augs: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_whole_slice_pipeline(
        img_size=img_size,
        train=True,
        strong_augs=strong_augs,
        include_mask=include_mask,
        include_bbox=include_bbox,
    ))


def get_val_transforms_whole_slice(
    img_size: int = 384,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_whole_slice_pipeline(
        img_size=img_size,
        train=False,
        include_mask=include_mask,
        include_bbox=include_bbox,
    ))


def _build_mil_bag_pipeline(
    img_size: int,
    bag_size: int,
    train: bool,
    strong_augs: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> list:
    """MIL bag pipeline.

    Load CT + lung mask. Spacing to 1.0 x 1.0 x 2.0 mm (both via the same
    Spacingd so the Z grids stay aligned). Scale intensity. Resize XY to
    ``img_size`` (keep Z). Pick ``bag_size`` evenly-spaced slices from the
    lung mask's axial extent. Augment in-plane only (never across the bag
    axis — instances in one bag must stay registered). Permute to
    (N, 1, H, W). Per-instance normalize. Output: ``(N, 1, H, W)``.
    """
    keep_mask = bool(include_mask or include_bbox)
    load_keys = ["image", "lung_mask"] + (["tumor_mask"] if keep_mask else [])
    # Per-key interpolation modes: image=bilinear, masks=nearest. Extend for
    # tumor_mask when keep_mask=True so mode list length matches load_keys.
    _spacing_modes = ["bilinear", "nearest"] + (["nearest"] if keep_mask else [])
    _resize_modes  = ["trilinear", "nearest"] + (["nearest"] if keep_mask else [])
    _aug_keys = ["image"] + (["tumor_mask"] if keep_mask else [])
    _aug_spatial_modes = ["bilinear"] + (["nearest"] if keep_mask else [])
    transforms: list = [
        LoadNiftiWithRGBSupportd(keys=load_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=load_keys,
            pixdim=(1.0, 1.0, 2.0),
            mode=_spacing_modes,
            allow_missing_keys=True,
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
        Resized(
            keys=load_keys,
            spatial_size=(img_size, img_size, -1),
            mode=_resize_modes,
            allow_missing_keys=True,
        ),
        LungAxialBagSelectd(
            keys=["image"] + (["tumor_mask"] if keep_mask else []),
            source_key="lung_mask",
            num_slices=bag_size,
            jitter=train,
            allow_missing_keys=True,
        ),
        # Lung mask has served its purpose; drop it so it doesn't waste cache
        # memory or get into the collator.
        DeleteItemsd(keys=["lung_mask"]),
    ]

    if train and not strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),  # H
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),  # W
            RandAffined(
                keys=_aug_keys,
                prob=0.5,
                rotate_range=(0.0, 0.0, 0.26),   # in-plane (Z-axis) rotation only
                translate_range=(8, 8, 0),
                scale_range=(0.1, 0.1, 0.0),
                mode=_aug_spatial_modes,
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
        ]
    elif train and strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=_aug_keys,
                prob=0.8,
                rotate_range=(0.0, 0.0, 0.35),
                translate_range=(12, 12, 0),
                scale_range=(0.15, 0.15, 0.0),
                mode=_aug_spatial_modes,
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.7),
            RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.7),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
        ]

    transforms += [
        # Permute BEFORE normalize so channel_wise=True normalizes each
        # instance independently. If we normalized first (on (1,H,W,N)),
        # a single bright slice would skew the whole bag's stats.
        BagAsBatchDimd(keys=["image"] + (["tumor_mask"] if keep_mask else [])),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    if include_bbox:
        transforms.append(
            BBoxFromMaskd(keys=["tumor_mask"], source_key="tumor_mask", bbox_key="bbox", has_key="has_bbox", allow_missing_keys=True)
        )
    if not keep_mask:
        transforms.append(PopKeysd(keys=["tumor_mask"]))
    transforms.append(ToTensord(keys=["image"] + (["tumor_mask"] if keep_mask else []), allow_missing_keys=True))
    return transforms


def get_train_transforms_mil_bag(
    img_size: int = 384,
    bag_size: int = 16,
    strong_augs: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_mil_bag_pipeline(
        img_size=img_size,
        bag_size=bag_size,
        train=True,
        strong_augs=strong_augs,
        include_mask=include_mask,
        include_bbox=include_bbox,
    ))


def get_val_transforms_mil_bag(
    img_size: int = 384,
    bag_size: int = 16,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_mil_bag_pipeline(
        img_size=img_size,
        bag_size=bag_size,
        train=False,
        include_mask=include_mask,
        include_bbox=include_bbox,
    ))
