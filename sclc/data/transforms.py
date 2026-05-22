"""MONAI transforms for SCLC CT preprocessing."""
from typing import Any, Dict, Hashable, Mapping, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection  # type: ignore[attr-defined]
from monai.data import MetaTensor
from scipy.ndimage import label as cc_label
from monai.transforms import (
    Compose,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCoarseDropoutd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Randomizable,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    SqueezeDimd,
    ToTensord,
)


class LoadNiftiWithRGBSupportd(MapTransform):
    """Load NIfTI files, with handling for RGB structured dtypes and 4D volumes."""

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
            # mmap=False: forces decompression in this thread. Memory-mapped
            # access on thousands of gz files across DataLoader workers
            # destabilizes downstream Spacingd loops.
            img = nib.load(filepath, mmap=False)

            affine = img.affine

            is_rgb = hasattr(img.dataobj, "dtype") and hasattr(img.dataobj.dtype, "names") and img.dataobj.dtype.names is not None and ('R' in img.dataobj.dtype.names or set(img.dataobj.dtype.names) == {'R', 'G', 'B'})

            if is_rgb:
                raw_rgb = np.asanyarray(img.dataobj)
                r = raw_rgb['R'].astype(np.float32)
                g = raw_rgb['G'].astype(np.float32)
                b = raw_rgb['B'].astype(np.float32)
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                # Map 0-255 back to HU-like range so the same intensity window applies.
                arr = (gray / 255.0) * 4095 - 1024
            else:
                # get_fdata() forces full decompression on the CPU thread;
                # the proxy view crashes inside Spacingd's PyTorch loops.
                arr = img.get_fdata(dtype=np.float32)

            while arr.ndim > 3:
                squeezed = False
                for ax in range(arr.ndim - 1, 2, -1):
                    if arr.shape[ax] == 1:
                        arr = arr.squeeze(axis=ax)
                        squeezed = True
                        break

                if not squeezed:
                    if arr.shape[-1] == 3:
                        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                        gray = 0.299 * r + 0.587 * g + 0.114 * b
                        if arr.max() <= 256:
                            arr = (gray / 255.0) * 4095 - 1024
                        else:
                            arr = gray
                    elif arr.ndim >= 4:
                        arr = arr[..., 0]
                    else:
                        break

            if arr.ndim == 3 and arr.shape[2] < 3:
                # Spacingd needs at least 3 slices; tile the volume.
                reps = (3 // arr.shape[2]) + 1
                arr = np.repeat(arr, reps, axis=2)

            d[key] = MetaTensor(arr, affine=affine)

        return d


class ExtractSubVolumed(MapTransform):
    """Extract `num_slices` axial slices centered on the largest connected
    component of the tumor mask.

    Multifocal masks (~80% of BigLunge auto-seg) would otherwise center on
    the midpoint between distant lesions, catching empty parenchyma; the
    largest CC is overwhelmingly the dominant lesion, and since metastases
    share the primary's histology any tumor tissue suffices for subtype
    classification. Components below `min_component_voxels` are treated as
    noise; empty/absent masks fall back to the volume center.
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
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)
        binary = arr > 0.5
        if not binary.any():
            return None
        while binary.ndim > 3:
            binary = binary[0] if binary.shape[0] == 1 else binary.any(axis=0)

        labeled, n = cc_label(binary)
        if n == 0:
            return None

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        valid = sizes >= min_component_voxels
        if not valid.any():
            # Sub-threshold mask: fall back to centroid of all non-zero voxels
            # rather than triggering the (worse) volume-center path.
            idx = np.argwhere(binary)
        else:
            masked_sizes = sizes.copy()
            masked_sizes[~valid] = 0
            largest_label = int(masked_sizes.argmax())
            idx = np.argwhere(labeled == largest_label)

        if idx.size == 0:
            return None
        return int(round(idx[:, -1].mean()))

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)

        target_z_center = None
        if "mask" in d:
            target_z_center = self._largest_cc_z_center(
                d["mask"], self.min_component_voxels,
            )

        for key in self.key_iterator(d):
            volume = d[key]
            if not hasattr(volume, 'ndim') or volume.ndim < 4:
                continue

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
    """Axis-aligned bbox from a binary mask, normalized to [0, 1] for
    resolution-agnostic downstream heads. 2D: (xmin,ymin,xmax,ymax);
    3D: (xmin,ymin,zmin,xmax,ymax,zmax).
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
        # Strip channel dim so (1,H,W) after SqueezeDimd doesn't fall into the
        # 3D branch and return a 6-element bbox for a 2D slice.
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

        # Bag input: (N, 1, H, W) — one bbox per instance.
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
    """Crop a fixed-size patch centered on the largest connected component
    of the tumor mask. `source_key` must share the post-Spacingd voxel grid.

    BigLunge auto-seg masks are multifocal in ~70-80% of patients
    (`scripts/audit_multifocal.py`); the unweighted centroid would land
    between distant lesions and catch empty parenchyma. The largest CC is
    the dominant lesion (primary or bulky met), and since metastases share
    the primary's histology, any tumor tissue suffices for subtype work.

    Fallbacks: empty/missing mask -> volume center; all components below
    `min_component_voxels` -> centroid of all non-zero voxels. Out-of-bounds
    regions are zero-padded so the output shape stays exact.
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
        if mask is None:
            return None

        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)

        binary = arr > 0.5
        if not binary.any():
            return None

        # Default 1-connectivity (face-only): does not merge spatially
        # disjoint blobs through diagonals. Same setting as the audit script.
        labeled, n = cc_label(binary)
        if n == 0:
            return None

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        valid = sizes >= self.min_component_voxels
        if not valid.any():
            # Sub-threshold mask: centroid of all non-zero voxels (better
            # than returning None and triggering the volume-center fallback).
            idx = np.argwhere(binary)
        else:
            masked_sizes = sizes.copy()
            masked_sizes[~valid] = 0
            largest_label = int(masked_sizes.argmax())
            idx = np.argwhere(labeled == largest_label)

        if idx.size == 0:
            return None
        # Take the last 3 columns: handles both 3D (C,H,W,Z) and the 2D path
        # (C=1,H,W,Z=1 after SliceSelect) uniformly.
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
                # F.pad orders last-dim-first.
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

    SCLC frequently sits in the mediastinum and apex; the (30, 30, 20)
    margin at 1.5x1.5x2.0 mm spacing (~45x45x40 mm) absorbs mask errors
    and keeps peri-lung context.
    """
    crop_keys = list(img_keys) + ["lung_mask"]
    return [
        # Push the lung mask through the same spatial pipeline as the CT so
        # its bbox lines up voxel-for-voxel with the data being cropped.
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
        DeleteItemsd(keys=["lung_mask"]),
    ]


def _aug_block_3d(val_keys: list, strong_augs: bool) -> list:
    """Affine + intensity augs applied jointly to image + mask.

    RandCoarseDropoutd is intentionally absent: sparse 3D holes can wipe
    out the entire signal of a small SCLC primary, and the 3D pipeline
    has no per-slice redundancy to absorb the loss.
    """
    if not strong_augs:
        return [
            RandAffined(
                keys=val_keys,
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
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
            rotate_range=(0.20, 0.20, 0.20),
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
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=val_keys,
            pixdim=(1.5, 1.5, 2.0),
            mode=spacing_modes,
            allow_missing_keys=True
        ),
        # Lung-bbox crop before intensity scaling so the original HU values
        # flow into ScaleIntensityRanged below.
        *( _build_lung_crop_transforms(val_keys) if use_lung_crop else [] ),

        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),

        ExtractSubVolumed(keys=val_keys, num_slices=depth_size, allow_missing_keys=True),

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
        Orientationd(keys=load_keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=val_keys,
            pixdim=(1.5, 1.5, 2.0),
            mode=spacing_modes,
            allow_missing_keys=True
        ),
        *( _build_lung_crop_transforms(val_keys) if use_lung_crop else [] ),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=0,
            b_max=1,
            clip=True,
        ),

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
    """Drop-in `DeleteItemsd` that tolerates missing keys (older MONAI lacks
    `allow_missing_keys` on `DeleteItemsd`).
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for k in self.keys:
            d.pop(k, None)
        return d


class SliceSelectd(MapTransform):
    """Select one axial slice (keeping a length-1 Z axis) from (C,X,Y,Z) volumes.
    Index is clamped to the volume's Z extent so spacing/orientation drift
    doesn't crash at runtime.
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
    """Shared 2D pipeline: load CT + tumor mask, spacing, pick the slice from
    `slice_idx`, in-plane crop around the tumor bbox, scale, resize, squeeze
    Z, augment, normalize.

    `crop_size` is at the post-Spacing 1 mm X/Y grid (so ~ diameter in mm).
    Default 96 fits Lung-PET-CT-Dx tumors (<60 mm); BigLunge needs >= 160
    to avoid large-mass overflow near the slice edge.
    """
    load_keys = ["image", "tumor_mask"]
    keep_mask = bool(include_mask or include_bbox)
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
        # Volume stays 4D with Z=1 so CropAroundTumord can be reused.
        SliceSelectd(keys=load_keys, slice_key="slice_idx", allow_missing_keys=True),
        CropAroundTumord(
            keys=["image"] + (["tumor_mask"] if keep_mask else []),
            source_key="tumor_mask",
            patch_size=(int(crop_size), int(crop_size), 1),
            allow_missing_keys=True,
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
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

    # BBoxFromMaskd after augmentation so the bbox sits in the augmented frame.
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


# -----------------------------------------------------------------------------
# MIL pipeline: whole-slice DAPT + bag-level BigLunge
# -----------------------------------------------------------------------------
# Both pipelines share the front-end (load/orient/spacing/HU window/resize XY)
# and intentionally omit CropAroundTumord so BigLunge inference can run
# tumor-mask-free; the backbone sees the same visual scale across phases.
# Bag samples are permuted (C,H,W,N) -> (N,1,H,W) at the end so DataLoader
# stacking yields (B,N,1,H,W) for MILModel. Bag-axis rotate/translate/scale
# are disabled so instances within a bag stay registered.


class BagAsBatchDimd(MapTransform):
    """Permute a (C=1, H, W, N) bag volume to (N, 1, H, W), matching
    `MILModel`'s expected per-sample shape.
    """

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            vol = d[key]
            if isinstance(vol, torch.Tensor):
                if vol.ndim == 4:
                    d[key] = vol.permute(3, 0, 1, 2).contiguous()
                elif vol.ndim == 3:
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
    """Sample `num_slices` evenly-spaced axial slices from `source_key`'s z-extent.

    Volumes must be channel-first (C, H, W, Z) pre-aligned via Orientation +
    Spacing. `jitter=True` (train only) shifts the linspace grid by a uniform
    random offset in `[-stride/2, +stride/2]` per call — same offset across
    keyed volumes — so tumors that fall between bag slices get sampled
    eventually.

    Empty/missing mask: warns and falls back to full Z extent. Silent
    fallback would let MIL sample the abdomen on truncated lung masks.
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
        # Use MONAI's managed RandomState for seed-consistency with sibling Rand*.
        self._offset_frac = float(self.R.uniform(-0.5, 0.5)) if self.jitter else 0.0

    @staticmethod
    def _z_extent(mask) -> Optional[tuple]:
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            m = (mask > 0.5)
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
            idxs = [z_min] * self.num_slices

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
    Output shape `(C=1, img_size, img_size)`. No in-plane tumor crop.
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
    """MIL bag pipeline. Output shape `(bag_size, 1, img_size, img_size)`."""
    keep_mask = bool(include_mask or include_bbox)
    load_keys = ["image", "lung_mask"] + (["tumor_mask"] if keep_mask else [])
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
        DeleteItemsd(keys=["lung_mask"]),
    ]

    if train and not strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=_aug_keys,
                prob=0.5,
                # Z-axis rotation only — keeps bag instances registered.
                rotate_range=(0.0, 0.0, 0.26),
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
        # Permute before normalize: channel_wise=True must normalize each
        # bag instance independently; otherwise one bright slice skews the bag.
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


# -----------------------------------------------------------------------------
# MIL bag DAPT pipeline (Lung-PET-CT-Dx — no lung mask available)
# -----------------------------------------------------------------------------


def _build_mil_bag_dapt_pipeline(
    img_size: int,
    bag_size: int,
    train: bool,
    strong_augs: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> list:
    """MIL bag pipeline for DAPT on Lung-PET-CT-Dx — uses the tumour mask
    (key `mask`) for z-extent, with full-CT fallback when absent.
    """
    keep_mask = bool(include_mask or include_bbox)
    _aug_keys = ["image"] + (["mask"] if keep_mask else [])
    _aug_spatial_modes = ["bilinear"] + (["nearest"] if keep_mask else [])
    transforms: list = [
        # Load + preprocess mask so LungAxialBagSelectd sees a tensor (not the
        # raw path). allow_missing_keys handles patients without a mask file.
        LoadNiftiWithRGBSupportd(keys=["image", "mask"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=["image", "mask"], axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 2.0),
            mode=["bilinear", "nearest"],
            allow_missing_keys=True,
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
        Resized(
            keys=["image", "mask"],
            spatial_size=(img_size, img_size, -1),
            mode=["trilinear", "nearest"],
            allow_missing_keys=True,
        ),
        LungAxialBagSelectd(
            keys=["image"] + (["mask"] if keep_mask else []),
            source_key="mask",
            num_slices=bag_size,
            jitter=train,
            allow_missing_keys=True,
        ),
    ]
    if not keep_mask:
        transforms.append(PopKeysd(keys=["mask"]))

    if train and not strong_augs:
        transforms += [
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=_aug_keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=_aug_keys, prob=0.5,
                rotate_range=(0.0, 0.0, 0.26), translate_range=(8, 8, 0), scale_range=(0.1, 0.1, 0.0),
                mode=_aug_spatial_modes, padding_mode="zeros", allow_missing_keys=True,
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
                keys=_aug_keys, prob=0.8,
                rotate_range=(0.0, 0.0, 0.35), translate_range=(12, 12, 0), scale_range=(0.15, 0.15, 0.0),
                mode=_aug_spatial_modes, padding_mode="zeros", allow_missing_keys=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.7),
            RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.7),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
        ]

    transforms += [
        BagAsBatchDimd(keys=["image"] + (["mask"] if keep_mask else [])),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    if include_bbox:
        transforms.append(
            BBoxFromMaskd(keys=["mask"], source_key="mask", bbox_key="bbox", has_key="has_bbox", allow_missing_keys=True)
        )
    transforms.append(ToTensord(keys=["image"] + (["mask"] if keep_mask else []), allow_missing_keys=True))
    return transforms


def get_train_transforms_mil_bag_dapt(
    img_size: int = 256,
    bag_size: int = 16,
    strong_augs: bool = False,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_mil_bag_dapt_pipeline(
        img_size=img_size, bag_size=bag_size, train=True,
        strong_augs=strong_augs, include_mask=include_mask, include_bbox=include_bbox,
    ))


def get_val_transforms_mil_bag_dapt(
    img_size: int = 256,
    bag_size: int = 16,
    include_mask: bool = False,
    include_bbox: bool = False,
) -> Compose:
    return Compose(_build_mil_bag_dapt_pipeline(
        img_size=img_size, bag_size=bag_size, train=False,
        include_mask=include_mask, include_bbox=include_bbox,
    ))
