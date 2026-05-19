"""Unified Grad-CAM tool for the 2D / MIL / 3D pipelines.

Usage:
    python -m sclc.grad_cam.grad_cam --model densenet121_2d
    python -m sclc.grad_cam.grad_cam --model mil_resnet50
    python -m sclc.grad_cam.grad_cam --model swin_unetr

The model name is the only required argument. The script:
  1. Resolves the pipeline (2d / mil / 3d) from the model_type.
  2. Reads ``results/thesis/<pipeline>/per_model/<model>/_provenance.json`` and
     loads ``checkpoints.finetune_pbest_raw`` (or ``dapt_pbest_raw`` if FT is
     missing or --use-dapt is passed).
  3. Preprocesses a hardcoded Lung-PET-CT-Dx ADC scan with the same val
     transforms the model trained against.
  4. Runs hook-based Grad-CAM on a pipeline-appropriate target layer.
  5. Saves a PNG with the input CT slice and its color overlay side-by-side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sclc.data.transforms import (
    get_val_transforms_2d,
    get_val_transforms_3d,
    get_val_transforms_mil_bag,
)
from sclc.models import get_sclc_model
from sclc.models.factory import get_pipeline


CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]
REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = REPO_ROOT / "results" / "thesis"
SPLITS_PATH = REPO_ROOT / "results" / "splits.json"

# Default ADC scan from Lung-PET-CT-Dx (folder letter 'A' = Adenocarcinoma).
# The series UID is picked at runtime: the patient ships multiple
# reconstructions (commonly 5 mm + 1.25 mm) and we want the thinnest one — same
# rule the loader uses (sclc/data/loaders.py:_z_then_name) so grad-cam shows
# the same scan the model trained on.
LPCD_DEFAULT_PATIENT = "Lung_Dx-A0180"
LPCD_DATA_ROOT = Path("/home/data/Lung-PET-CT-Dx-Clean")

# Default ADC scan from BigLunge / TrainingData. patient_025040 is the first ADC
# patient by row order in patients_parameters.csv and is not in the empty-mask
# exclusion list.
BIGLUNGE_DEFAULT_PATIENT = "patient_025040"
BIGLUNGE_DATA_ROOT = Path("/home/data/TrainingData")


# ----- target layer registry --------------------------------------------------

def _swin_bhwc_to_bchw(t: torch.Tensor) -> torch.Tensor:
    """timm Swin layers / norm output (B, H, W, C). Grad-CAM expects (B, C, H, W)."""
    if t.ndim != 4:
        raise ValueError(f"Expected (B, H, W, C), got shape {tuple(t.shape)}")
    return t.permute(0, 3, 1, 2).contiguous()


# (model_type) -> (path-to-target-layer, optional reshape transform)
TARGET_LAYERS: Dict[str, Tuple[str, Optional[Callable[[torch.Tensor], torch.Tensor]]]] = {
    # 2D ImageNet
    "efficientnet_b0_2d": ("efficientnet._bn1", None),
    "densenet121_2d":     ("densenet.features.denseblock4", None),
    "resnet50_2d":        ("backbone.features.layer4", None),
    "swin_tiny_2d":       ("swin.norm", _swin_bhwc_to_bchw),
    # 2D RadImageNet
    "resnet50_2d_rin":    ("backbone.layer4", None),
    "densenet121_2d_rin": ("densenet.features.denseblock4", None),
    # MIL
    "mil_resnet50":       ("mil.net.layer4", None),
    "mil_swin_tiny":      ("mil.net.norm", _swin_bhwc_to_bchw),
    # 3D
    "swin_unetr":         ("swin_unetr.swinViT.layers4.0", None),
}


def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    mod = root
    for part in dotted.split("."):
        if part.isdigit():
            mod = mod[int(part)]
        else:
            mod = getattr(mod, part)
    return mod


# ----- checkpoint resolution --------------------------------------------------

def _resolve_checkpoint(model_type: str, biglunge: bool) -> Tuple[Path, str]:
    """Pick the checkpoint that matches the dataset.

    BigLunge → ``finetune_pbest_raw`` (the FT phase trains on BigLunge);
    Lung-PET-CT-Dx → ``dapt_pbest_raw`` (the DAPT phase trains on it). When
    the preferred checkpoint is missing the other is used as a fallback so the
    tool still runs on partial provenance.
    """
    pipeline = get_pipeline(model_type)
    prov_path = THESIS_ROOT / pipeline / "per_model" / model_type / "_provenance.json"
    if not prov_path.is_file():
        raise FileNotFoundError(f"No provenance for '{model_type}': {prov_path}")
    with open(prov_path) as f:
        info = json.load(f)
    ckpts: Dict[str, str] = info.get("checkpoints", {})
    order = ["finetune_pbest_raw", "dapt_pbest_raw"] if biglunge else ["dapt_pbest_raw", "finetune_pbest_raw"]
    for key in order:
        path = ckpts.get(key)
        if path and Path(path).is_file():
            return Path(path), key
    raise FileNotFoundError(f"No usable checkpoint listed in {prov_path}: {ckpts}")


# ----- splits lookup ----------------------------------------------------------

def _lookup_split_and_gt(patient_id: str, biglunge: bool) -> Tuple[Optional[str], Optional[int]]:
    """Find (split, gt_class) for ``patient_id`` in ``results/splits.json``.

    Returns ``(None, None)`` if the splits file is missing or the patient isn't
    listed (e.g., excluded by EMPTY_TUMOR_MASK / TRUNCATED_LUNG_MASK or the
    user passed an out-of-cohort image).
    """
    if not SPLITS_PATH.is_file():
        return None, None
    try:
        with open(SPLITS_PATH) as f:
            splits = json.load(f)
    except Exception:
        return None, None
    block = splits.get("biglunge" if biglunge else "lung_pet_ct_dx", {})
    for split_name, entries in block.items():
        for e in entries:
            if e.get("patient_id") == patient_id:
                return split_name, int(e.get("class_idx", -1))
    return None, None


# ----- image / slice helpers --------------------------------------------------

def _pick_thinnest_lpcd_series(patient_dir: Path) -> Path:
    """Return the ``_image.nii.gz`` with the smallest Z-spacing in ``patient_dir``.

    Mirrors ``sclc/data/loaders.py:_z_then_name`` so the grad-cam default
    series matches the one the loader would have fed the model.
    """
    candidates = sorted(patient_dir.glob("*_image.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No *_image.nii.gz under {patient_dir}")

    def _z_spacing(p: Path) -> Tuple[float, str]:
        try:
            zooms = nib.load(str(p), mmap=False).header.get_zooms()
            z = float(zooms[2]) if len(zooms) >= 3 else float("inf")
        except Exception:
            z = float("inf")
        return z, p.name

    return min(candidates, key=_z_spacing)


def _resolve_default_paths(biglunge: bool) -> Tuple[Path, Path, Path, str]:
    """Return ``(image, tumor_mask, lung_mask, patient_name)``.

    ``lung_mask`` is empty for Lung-PET-CT-Dx (no lung mask shipped). The MIL
    pipeline falls back to the tumor mask in that case.
    """
    if biglunge:
        pdir = BIGLUNGE_DATA_ROOT / BIGLUNGE_DEFAULT_PATIENT
        img = pdir / f"{BIGLUNGE_DEFAULT_PATIENT}_input.nii.gz"
        tumor = pdir / f"{BIGLUNGE_DEFAULT_PATIENT}_label_tc.nii.gz"
        lung = pdir / f"{BIGLUNGE_DEFAULT_PATIENT}_label_lungs.nii.gz"
        if not img.is_file():
            raise FileNotFoundError(f"Default BigLunge image missing: {img}")
        return img, (tumor if tumor.is_file() else Path("")), (lung if lung.is_file() else Path("")), BIGLUNGE_DEFAULT_PATIENT
    pdir = LPCD_DATA_ROOT / LPCD_DEFAULT_PATIENT
    img = _pick_thinnest_lpcd_series(pdir)
    # Each series UID has a sibling ``<uid>_mask.nii.gz``; derive it from the image name.
    tumor = img.with_name(img.name.replace("_image.nii.gz", "_mask.nii.gz"))
    return img, (tumor if tumor.is_file() else Path("")), Path(""), LPCD_DEFAULT_PATIENT


def _scan_tumor_slice_idx(mask_path: Path, pixdim=(1.0, 1.0, 2.0)) -> int:
    """Return the middle tumor slice (post-Spacing) by scanning the mask.

    Mirrors ``sclc.data.dataset_2d._scan_tumor_slice_indices`` so the chosen
    slice is the same one the 2D dataset would have fed to the model.
    """
    from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Orientation, Spacing
    loader = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=pixdim, mode="nearest"),
    ])
    m = loader(str(mask_path))
    arr = m[0].cpu().numpy() if hasattr(m[0], "cpu") else np.asarray(m[0])
    per_slice = (arr > 0.5).sum(axis=tuple(range(arr.ndim - 1)))
    nz = np.nonzero(per_slice)[0]
    if nz.size == 0:
        return int(arr.shape[-1] // 2)
    return int(nz[nz.size // 2])


# ----- pipeline-specific input builders ---------------------------------------

def _build_2d_input(
    image_path: Path,
    mask_path: Path,
    img_size: int,
    crop_size: int,
) -> Tuple[torch.Tensor, int]:
    if not mask_path or not mask_path.is_file():
        raise FileNotFoundError(
            f"2D pipeline needs a tumor mask for slice + crop selection, none at {mask_path}"
        )
    slice_idx = _scan_tumor_slice_idx(mask_path)
    tx = get_val_transforms_2d(img_size=img_size, crop_size=crop_size)
    sample = tx({
        "image": str(image_path),
        "tumor_mask": str(mask_path),
        "slice_idx": int(slice_idx),
    })
    img = sample["image"]
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    img = img.float()
    if img.ndim == 3:        # (C, H, W)
        img = img.unsqueeze(0)
    return img, slice_idx


def _build_mil_input(
    image_path: Path,
    tumor_mask_path: Path,
    lung_mask_path: Path,
    img_size: int,
    bag_size: int,
) -> torch.Tensor:
    """Build a (1, N, 1, H, W) bag.

    BigLunge ships a real lung mask (``_label_lungs.nii.gz``), use that — same
    source the MIL training pipeline reads. Lung-PET-CT-Dx has no lung mask,
    so fall back to the tumor mask: that concentrates the bag on tumor-bearing
    slices, which is what we want for a single-image visualization.
    """
    tx = get_val_transforms_mil_bag(img_size=img_size, bag_size=bag_size)
    data: Dict[str, Any] = {"image": str(image_path)}
    if lung_mask_path and lung_mask_path.is_file():
        data["lung_mask"] = str(lung_mask_path)
    elif tumor_mask_path and tumor_mask_path.is_file():
        data["lung_mask"] = str(tumor_mask_path)
    sample = tx(data)
    bag = sample["image"]
    if not torch.is_tensor(bag):
        bag = torch.as_tensor(bag)
    return bag.float().unsqueeze(0)  # (1, N, 1, H, W)


def _build_3d_input(
    image_path: Path,
    mask_path: Path,
    img_size: int,
    depth_size: int,
) -> torch.Tensor:
    tx = get_val_transforms_3d(img_size=img_size, depth_size=depth_size)
    data: Dict[str, Any] = {"image": str(image_path)}
    if mask_path and mask_path.is_file():
        data["mask"] = str(mask_path)
    sample = tx(data)
    img = sample["image"]
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    img = img.float()
    if img.ndim == 4:
        img = img.unsqueeze(0)
    return img


# ----- Grad-CAM core ----------------------------------------------------------

class GradCAM:
    """Forward + backward hook Grad-CAM. Layer-output reshape optional.

    Works for any module whose output is reshapable to (B, C, *spatial). Caller
    must supply ``reshape`` for layers that don't already produce that layout
    (e.g. timm Swin emits (B, H, W, C)).
    """

    def __init__(
        self,
        model: nn.Module,
        target: nn.Module,
        reshape: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model
        self.target = target
        self.reshape = reshape
        self.act: Optional[torch.Tensor] = None
        self.grad: Optional[torch.Tensor] = None
        self._h_fwd = target.register_forward_hook(self._fwd_hook)
        self._h_bwd = target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _module, _inp, output):
        self.act = output

    def _bwd_hook(self, _module, _grad_in, grad_out):
        self.grad = grad_out[0]

    def remove(self) -> None:
        self._h_fwd.remove()
        self._h_bwd.remove()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits.detach(), dim=1)
        score = logits[:, class_idx].sum()
        score.backward()

        a = self.act
        g = self.grad
        if a is None or g is None:
            raise RuntimeError("Grad-CAM hook did not capture activations/gradients.")
        if self.reshape is not None:
            a = self.reshape(a)
            g = self.reshape(g)

        spatial = list(range(2, a.ndim))
        weights = g.mean(dim=spatial, keepdim=True)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        # Per-sample 0-1 normalize
        flat = cam.flatten(1)
        cmin = flat.min(dim=1).values.view(-1, *([1] * (cam.ndim - 1)))
        cmax = flat.max(dim=1).values.view(-1, *([1] * (cam.ndim - 1)))
        cam = (cam - cmin) / (cmax - cmin + 1e-8)
        return cam.detach(), class_idx, probs


# ----- visualization ----------------------------------------------------------

def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def _save_overlay_png(
    ct_slice: np.ndarray,
    cam_slice: np.ndarray,
    out_path: Path,
    title: str,
    alpha: float = 0.45,
) -> None:
    """Save a side-by-side PNG: raw CT slice, then CT + jet overlay."""
    ct = _norm01(ct_slice)
    cam = _norm01(cam_slice)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ct, cmap="gray")
    axes[0].set_title("Input CT slice")
    axes[0].axis("off")
    axes[1].imshow(ct, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=alpha)
    axes[1].set_title("Grad-CAM overlay")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----- pipeline runners -------------------------------------------------------

def _title_lines(
    model_type: str,
    patient_name: str,
    pred_class: int,
    pred_prob: float,
    cam_class: int,
    gt_class: Optional[int],
    split: Optional[str],
    extra: str = "",
) -> str:
    gt_part = f"gt={CLASS_NAMES[gt_class]}" if gt_class is not None and 0 <= gt_class < len(CLASS_NAMES) else "gt=?"
    split_part = f" [{split}]" if split else ""
    head = f"{model_type}  |  {patient_name}{split_part}"
    if extra:
        head += f"  |  {extra}"
    return (
        f"{head}\n"
        f"{gt_part}, pred={CLASS_NAMES[pred_class]} (p={pred_prob:.2f}), "
        f"cam={CLASS_NAMES[cam_class]}"
    )


def _run_2d(
    model: nn.Module,
    image_path: Path,
    tumor_mask_path: Path,
    target_path: str,
    reshape: Optional[Callable],
    out_dir: Path,
    model_type: str,
    patient_name: str,
    gt_class: Optional[int],
    split: Optional[str],
    class_idx: Optional[int],
    device: torch.device,
    img_size: int = 224,
    crop_size: int = 96,
) -> Dict[str, Any]:
    img, slice_idx = _build_2d_input(image_path, tumor_mask_path, img_size, crop_size)
    img = img.to(device)
    cam_eng = GradCAM(model, _resolve_module(model, target_path), reshape)
    try:
        cam, pred_class, probs = cam_eng(img, class_idx=class_idx)
    finally:
        cam_eng.remove()

    cam_up = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)
    ct_np = img[0, 0].detach().cpu().numpy()
    cam_np = cam_up[0, 0].detach().cpu().numpy()

    cam_class = pred_class if class_idx is None else int(class_idx)
    title = _title_lines(
        model_type, patient_name, pred_class, float(probs[0, pred_class]),
        cam_class, gt_class, split, extra=f"z={slice_idx}",
    )
    out_path = out_dir / f"{model_type}_{patient_name}_z{slice_idx}_cam{cam_class}.png"
    _save_overlay_png(ct_np, cam_np, out_path, title)
    return {
        "png": str(out_path),
        "slice_idx": slice_idx,
        "pred_class": pred_class,
        "cam_class": cam_class,
        "probs": probs[0].cpu().tolist(),
    }


def _run_mil(
    model: nn.Module,
    image_path: Path,
    tumor_mask_path: Path,
    lung_mask_path: Path,
    target_path: str,
    reshape: Optional[Callable],
    out_dir: Path,
    model_type: str,
    patient_name: str,
    gt_class: Optional[int],
    split: Optional[str],
    class_idx: Optional[int],
    device: torch.device,
    img_size: int,
    bag_size: int,
) -> Dict[str, Any]:
    bag = _build_mil_input(
        image_path, tumor_mask_path, lung_mask_path, img_size, bag_size,
    ).to(device)  # (1, N, 1, H, W)
    cam_eng = GradCAM(model, _resolve_module(model, target_path), reshape)
    try:
        cam, pred_class, probs = cam_eng(bag, class_idx=class_idx)  # cam: (B*N, 1, h, w)
    finally:
        cam_eng.remove()

    # Pick the highest-attention instance for display.
    with torch.no_grad():
        attn = model.attention_weights(bag)  # (B=1, N)
    top_idx = int(attn[0].argmax().item())

    cam_up = F.interpolate(cam, size=bag.shape[-2:], mode="bilinear", align_corners=False)
    cam_up = cam_up.view(bag.shape[0], bag.shape[1], 1, *bag.shape[-2:])  # (B, N, 1, H, W)
    ct_np = bag[0, top_idx, 0].detach().cpu().numpy()
    cam_np = cam_up[0, top_idx, 0].detach().cpu().numpy()

    cam_class = pred_class if class_idx is None else int(class_idx)
    title = _title_lines(
        model_type, patient_name, pred_class, float(probs[0, pred_class]),
        cam_class, gt_class, split,
        extra=f"bag {top_idx}/{bag_size} (attn={attn[0, top_idx]:.2f})",
    )
    out_path = out_dir / f"{model_type}_{patient_name}_inst{top_idx}_cam{cam_class}.png"
    _save_overlay_png(ct_np, cam_np, out_path, title)
    return {
        "png": str(out_path),
        "top_instance": top_idx,
        "attention_top": float(attn[0, top_idx]),
        "pred_class": pred_class,
        "cam_class": cam_class,
        "probs": probs[0].cpu().tolist(),
    }


def _run_3d(
    model: nn.Module,
    image_path: Path,
    tumor_mask_path: Path,
    target_path: str,
    reshape: Optional[Callable],
    out_dir: Path,
    model_type: str,
    patient_name: str,
    gt_class: Optional[int],
    split: Optional[str],
    class_idx: Optional[int],
    device: torch.device,
    img_size: int = 224,
    depth_size: int = 128,
) -> Dict[str, Any]:
    img = _build_3d_input(image_path, tumor_mask_path, img_size, depth_size).to(device)  # (1, 1, H, W, D)
    cam_eng = GradCAM(model, _resolve_module(model, target_path), reshape)
    try:
        cam, pred_class, probs = cam_eng(img, class_idx=class_idx)
    finally:
        cam_eng.remove()

    cam_up = F.interpolate(
        cam, size=img.shape[-3:], mode="trilinear", align_corners=False,
    )  # (1, 1, H, W, D)
    # Pick the slice with highest CAM mass.
    cam_per_slice = cam_up[0, 0].sum(dim=(0, 1))   # (D,)
    z = int(cam_per_slice.argmax().item())
    ct_np = img[0, 0, :, :, z].detach().cpu().numpy()
    cam_np = cam_up[0, 0, :, :, z].detach().cpu().numpy()

    cam_class = pred_class if class_idx is None else int(class_idx)
    title = _title_lines(
        model_type, patient_name, pred_class, float(probs[0, pred_class]),
        cam_class, gt_class, split,
        extra=f"peak-cam z={z}/{depth_size}",
    )
    out_path = out_dir / f"{model_type}_{patient_name}_z{z}_cam{cam_class}.png"
    _save_overlay_png(ct_np, cam_np, out_path, title)

    # Also save the full 3D CAM as a NIfTI for downstream viewers.
    nii_path = out_dir / f"{model_type}_{patient_name}_cam{cam_class}_cam3d.nii.gz"
    nib.save(
        nib.Nifti1Image(cam_up[0, 0].detach().cpu().numpy().astype(np.float32), np.eye(4)),
        str(nii_path),
    )

    return {
        "png": str(out_path),
        "cam3d_nifti": str(nii_path),
        "peak_z": z,
        "pred_class": pred_class,
        "cam_class": cam_class,
        "probs": probs[0].cpu().tolist(),
    }


# ----- top-level entry --------------------------------------------------------

def run_grad_cam(
    model_type: str,
    biglunge: bool = False,
    image_path: Optional[Path] = None,
    tumor_mask_path: Optional[Path] = None,
    lung_mask_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    class_idx: Optional[int] = None,
    device: Optional[str] = None,
    img_size_2d: int = 224,
    crop_size_2d: int = 96,
    img_size_mil: Optional[int] = None,
    bag_size_mil: int = 32,
    img_size_3d: int = 224,
    depth_size_3d: int = 128,
) -> Dict[str, Any]:
    model_type = model_type.lower()
    if model_type not in TARGET_LAYERS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: {sorted(TARGET_LAYERS)}"
        )

    pipeline = get_pipeline(model_type)
    target_path, reshape = TARGET_LAYERS[model_type]
    dataset_name = "BigLunge" if biglunge else "Lung-PET-CT-Dx"

    # Pull defaults for any unspecified path (image, tumor mask, lung mask).
    d_img, d_tumor, d_lung, d_patient = _resolve_default_paths(biglunge)
    image_path = image_path or d_img
    tumor_mask_path = tumor_mask_path if tumor_mask_path is not None else d_tumor
    lung_mask_path = lung_mask_path if lung_mask_path is not None else d_lung
    # Patient name: derive from --image folder if user supplied one, else default.
    if image_path == d_img:
        patient_name = d_patient
    else:
        patient_name = image_path.parent.name or "unknown_patient"

    ckpt_path, ckpt_kind = _resolve_checkpoint(model_type, biglunge=biglunge)
    split, gt_class = _lookup_split_and_gt(patient_name, biglunge)
    gt_label = (
        CLASS_NAMES[gt_class]
        if gt_class is not None and 0 <= gt_class < len(CLASS_NAMES) else None
    )
    def _fmt(p: Optional[Path]) -> str:
        # Path("") -> ".", which is truthy and would otherwise leak through.
        return "(none)" if p is None or str(p) in ("", ".") else str(p)
    print(f"[grad_cam] Dataset:    {dataset_name}")
    print(f"[grad_cam] Patient:    {patient_name}  (split={split or 'unknown'}, gt={gt_label or 'unknown'})")
    print(f"[grad_cam] Checkpoint: {ckpt_kind} -> {ckpt_path}")
    print(f"[grad_cam] Image:      {image_path}")
    print(f"[grad_cam] Tumor mask: {_fmt(tumor_mask_path)}")
    print(f"[grad_cam] Lung mask:  {_fmt(lung_mask_path)}")

    out_dir = Path(output_dir) if output_dir else (REPO_ROOT / "results" / "grad_cam" / model_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load the checkpoint via the factory's existing logic.
    model = get_sclc_model(
        checkpoint_path=str(ckpt_path),
        model_type=model_type,
    )
    model = model.to(run_device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    if pipeline == "2d":
        info = _run_2d(
            model=model, image_path=image_path, tumor_mask_path=tumor_mask_path,
            target_path=target_path, reshape=reshape,
            out_dir=out_dir, model_type=model_type, patient_name=patient_name,
            gt_class=gt_class, split=split,
            class_idx=class_idx, device=run_device,
            img_size=img_size_2d, crop_size=crop_size_2d,
        )
    elif pipeline == "mil":
        # Default img_size depends on the backbone (Swin-Tiny is locked at 224).
        if img_size_mil is None:
            img_size_mil = 224 if "swin" in model_type else 384
        info = _run_mil(
            model=model, image_path=image_path,
            tumor_mask_path=tumor_mask_path, lung_mask_path=lung_mask_path,
            target_path=target_path, reshape=reshape,
            out_dir=out_dir, model_type=model_type, patient_name=patient_name,
            gt_class=gt_class, split=split,
            class_idx=class_idx, device=run_device,
            img_size=img_size_mil, bag_size=bag_size_mil,
        )
    elif pipeline == "3d":
        info = _run_3d(
            model=model, image_path=image_path, tumor_mask_path=tumor_mask_path,
            target_path=target_path, reshape=reshape,
            out_dir=out_dir, model_type=model_type, patient_name=patient_name,
            gt_class=gt_class, split=split,
            class_idx=class_idx, device=run_device,
            img_size=img_size_3d, depth_size=depth_size_3d,
        )
    else:
        raise RuntimeError(f"Unknown pipeline '{pipeline}'")

    def _path_or_none(p: Optional[Path]) -> Optional[str]:
        return None if p is None or str(p) in ("", ".") else str(p)
    info.update({
        "model_type": model_type,
        "pipeline": pipeline,
        "dataset": dataset_name,
        "patient": patient_name,
        "split": split,
        "gt_class": gt_class,
        "gt_label": gt_label,
        "checkpoint": str(ckpt_path),
        "checkpoint_kind": ckpt_kind,
        "target_layer": target_path,
        "image": str(image_path),
        "tumor_mask": _path_or_none(tumor_mask_path),
        "lung_mask": _path_or_none(lung_mask_path),
    })
    info["pred_label"] = CLASS_NAMES[info["pred_class"]]
    info["cam_label"] = CLASS_NAMES[info["cam_class"]]
    info["correct"] = (gt_class is not None and info["pred_class"] == gt_class)

    info_path = Path(info["png"]).with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print()
    print(f"[grad_cam] Pipeline:    {pipeline}")
    print(f"[grad_cam] GT:          {gt_class if gt_class is not None else '?'} ({gt_label or 'unknown'})")
    print(f"[grad_cam] Pred:        {info['pred_class']} ({info['pred_label']})")
    print(f"[grad_cam] CAM target:  {info['cam_class']} ({info['cam_label']})")
    print(f"[grad_cam] Correct:     {info['correct']}")
    print(f"[grad_cam] Probs:       {[round(p, 3) for p in info['probs']]}")
    print(f"[grad_cam] PNG:         {info['png']}")
    print(f"[grad_cam] Metadata:    {info_path}")
    if "cam3d_nifti" in info:
        print(f"[grad_cam] 3D CAM NIfTI: {info['cam3d_nifti']}")
    return info


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Grad-CAM on a 2D / MIL / 3D SCLC classifier.")
    p.add_argument("--model", required=True, choices=sorted(TARGET_LAYERS),
                   help="Model type (used to look up the checkpoint via _provenance.json).")
    p.add_argument("--biglunge", action="store_true",
                   help="Run on BigLunge (loads finetune_pbest_raw + a BigLunge ADC patient). "
                        "Default: Lung-PET-CT-Dx (loads dapt_pbest_raw).")
    p.add_argument("--image", type=Path, default=None,
                   help="CT NIfTI path. Defaults to a hardcoded ADC scan from the chosen dataset.")
    p.add_argument("--tumor-mask", type=Path, default=None,
                   help="Tumor mask NIfTI. Required for 2D; used for 3D Z-centering. "
                        "MIL falls back to this when no lung mask is supplied.")
    p.add_argument("--lung-mask", type=Path, default=None,
                   help="Lung mask NIfTI (BigLunge ships one; LPCT-Dx does not). "
                        "Used by the MIL pipeline to pick bag slices inside the lung extent.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: results/grad_cam/<model>/).")
    p.add_argument("--class-index", type=int, default=None,
                   help="Override target class for Grad-CAM (default: predicted class).")
    p.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto).")
    # Pipeline knobs (sane defaults from configs/experiments/*.yaml)
    p.add_argument("--img-size-2d", type=int, default=224)
    p.add_argument("--crop-size-2d", type=int, default=96)
    p.add_argument("--img-size-mil", type=int, default=None,
                   help="Default 224 for swin-tiny MIL, 384 otherwise.")
    p.add_argument("--bag-size-mil", type=int, default=32)
    p.add_argument("--img-size-3d", type=int, default=224)
    p.add_argument("--depth-size-3d", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_grad_cam(
        model_type=args.model,
        biglunge=args.biglunge,
        image_path=args.image,
        tumor_mask_path=args.tumor_mask,
        lung_mask_path=args.lung_mask,
        output_dir=args.output_dir,
        class_idx=args.class_index,
        device=args.device,
        img_size_2d=args.img_size_2d,
        crop_size_2d=args.crop_size_2d,
        img_size_mil=args.img_size_mil,
        bag_size_mil=args.bag_size_mil,
        img_size_3d=args.img_size_3d,
        depth_size_3d=args.depth_size_3d,
    )


if __name__ == "__main__":
    main()
