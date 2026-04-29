from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from sclc.grad_cam.colorize import colorize_heatmap, colorize_overlay, save_rgb_nifti


DEFAULT_PATIENT_DIR = "/home/data/Lung-PET-CT-Dx-Clean/Lung_Dx-A0043"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "mock"


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _norm_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    return (x - xmin) / (xmax - xmin + eps)


def _robust_norm_ct(ct: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = float(np.percentile(ct, 1.0))
    hi = float(np.percentile(ct, 99.0))
    if hi <= lo:
        return _norm_01(ct, eps=eps)
    return np.clip((ct - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def _gaussian_blur3d(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Small separable Gaussian blur using torch conv3d to avoid scipy dependency."""
    if sigma <= 0:
        return volume.astype(np.float32)

    radius = max(1, int(round(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel_1d /= float(kernel_1d.sum())

    k = torch.from_numpy(kernel_1d).float()
    kx = k.view(1, 1, -1, 1, 1)
    ky = k.view(1, 1, 1, -1, 1)
    kz = k.view(1, 1, 1, 1, -1)

    x = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    x = F.conv3d(F.pad(x, (0, 0, 0, 0, radius, radius), mode="replicate"), kx)
    x = F.conv3d(F.pad(x, (0, 0, radius, radius, 0, 0), mode="replicate"), ky)
    x = F.conv3d(F.pad(x, (radius, radius, 0, 0, 0, 0), mode="replicate"), kz)

    return x.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _elliptical_gaussian(shape: Tuple[int, int, int], center: Tuple[float, float, float], sigma_xyz: Tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = [max(1.0, float(v)) for v in sigma_xyz]
    cx, cy, cz = [float(v) for v in center]

    gx = np.exp(-0.5 * ((np.arange(shape[0], dtype=np.float32) - cx) / sx) ** 2)[:, None, None]
    gy = np.exp(-0.5 * ((np.arange(shape[1], dtype=np.float32) - cy) / sy) ** 2)[None, :, None]
    gz = np.exp(-0.5 * ((np.arange(shape[2], dtype=np.float32) - cz) / sz) ** 2)[None, None, :]
    return (gx * gy * gz).astype(np.float32)


def _connected_components(mask_bin: np.ndarray) -> List[np.ndarray]:
    """Return a list of boolean 3D arrays, one per connected component."""
    try:
        from scipy.ndimage import label as ndi_label  # type: ignore
    except ImportError:
        return [mask_bin > 0] if mask_bin.any() else []
    labeled, n_lab = ndi_label(mask_bin > 0)
    return [(labeled == i) for i in range(1, int(n_lab) + 1)]


def _build_mock_gradcam(ct: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Mock a Grad-CAM-looking heatmap:
      - One ellipsoidal Gaussian blob per connected component of ``mask``, sized
        from the component's bbox so the FWHM roughly matches the lesion.
      - Low-frequency noise modulation so the blob has organic, wavy edges
        instead of a textbook ellipse.
      - Body-mask gating so intensity stays inside the patient.
    The old blurred-bbox terms are dropped entirely — they were the reason the
    output looked rectangular, because bbox masks are axis-aligned boxes.
    """
    ct_norm = _robust_norm_ct(ct)
    mask_bin = (mask > 0.5).astype(np.float32)
    total_mask_voxels = float(mask_bin.sum())

    cam = np.zeros_like(ct_norm, dtype=np.float32)
    blob_count = 0

    if total_mask_voxels > 0.0:
        for comp in _connected_components(mask_bin):
            nz = np.argwhere(comp)
            if nz.size == 0:
                continue
            mins = nz.min(axis=0).astype(np.float32)
            maxs = nz.max(axis=0).astype(np.float32)
            extents = np.maximum((maxs - mins + 1), 3.0)
            center = nz.mean(axis=0).astype(np.float32)
            # sigma ~ 0.55 * extent makes the blob ~lesion-sized at its FWHM
            # while fading smoothly outside — same order across X/Y/Z so the
            # blob is near-spherical when extents are isotropic.
            sigma_xyz = tuple(float(max(4.0, e * 0.55)) for e in extents)
            blob = _elliptical_gaussian(mask_bin.shape, tuple(center.tolist()), sigma_xyz)
            cam = np.maximum(cam, blob)
            blob_count += 1
    else:
        # No mask: drop a plausible central hotspot so the mock still renders.
        shape = np.array(ct_norm.shape, dtype=np.float32)
        center = tuple((shape * np.array([0.52, 0.48, 0.50])).tolist())
        sigma_xyz = tuple(float(s * 0.10) for s in shape)
        cam = _elliptical_gaussian(ct_norm.shape, center, sigma_xyz)

    # Body mask keeps attention inside the patient silhouette.
    body_mask = (ct > np.percentile(ct, 5.0)).astype(np.float32)
    body_soft = _norm_01(_gaussian_blur3d(body_mask, sigma=2.0))

    # Low-frequency noise modulation: multiplies the blob by 1 ± ~35%, breaking
    # the perfect-ellipse look without changing its rough circular footprint.
    noise = rng.normal(loc=0.0, scale=1.0, size=ct_norm.shape).astype(np.float32)
    noise = _gaussian_blur3d(noise, sigma=3.0)
    noise = _norm_01(noise) - 0.5  # centered on 0, range ~[-0.5, 0.5]
    cam = cam * (1.0 + 0.35 * noise)

    cam = cam * body_soft
    cam = np.clip(cam, 0.0, None)
    cam = _norm_01(cam)

    stats = {
        "cam_mean": float(cam.mean()),
        "cam_max": float(cam.max()),
        "cam_q95": float(np.quantile(cam, 0.95)),
        "mask_voxels": total_mask_voxels,
        "blobs": int(blob_count),
    }
    return cam.astype(np.float32), stats


def _save_like_reference(volume: np.ndarray, ref_nii: nib.Nifti1Image, out_path: Path) -> None:
    hdr = ref_nii.header.copy()
    nib.save(nib.Nifti1Image(volume.astype(np.float32), ref_nii.affine, header=hdr), str(out_path))


def _find_series_pairs(patient_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    pairs: List[Tuple[Path, Optional[Path]]] = []
    image_paths = sorted(patient_dir.glob("*_image.nii.gz"))
    for image_path in image_paths:
        mask_path = image_path.with_name(image_path.name.replace("_image.nii.gz", "_mask.nii.gz"))
        pairs.append((image_path, mask_path if mask_path.exists() else None))
    return pairs


def generate_mock_gradcam(
    patient_dir: Path,
    output_dir: Path,
    alpha: float,
    seed: int,
    series_index: int,
) -> List[Path]:
    if not patient_dir.is_dir():
        raise ValueError(f"Patient directory does not exist: {patient_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _find_series_pairs(patient_dir)
    if not pairs:
        raise ValueError(f"No *_image.nii.gz files found in: {patient_dir}")

    if series_index >= 0:
        if series_index >= len(pairs):
            raise ValueError(f"series_index={series_index} is out of range for {len(pairs)} series.")
        pairs = [pairs[series_index]]

    written: List[Path] = []
    run_summary = {
        "patient_dir": str(patient_dir),
        "output_dir": str(output_dir),
        "seed": int(seed),
        "alpha": float(alpha),
        "series": [],
    }

    for idx, (image_path, mask_path) in enumerate(pairs):
        local_seed = seed + idx
        rng = np.random.default_rng(local_seed)

        ct_nii = nib.load(str(image_path), mmap=False)
        ct = ct_nii.get_fdata(dtype=np.float32)

        if mask_path is not None:
            mask_nii = nib.load(str(mask_path), mmap=False)
            mask = mask_nii.get_fdata(dtype=np.float32)
            if mask.shape != ct.shape:
                raise RuntimeError(
                    f"Mask/CT shape mismatch for {image_path.name}: mask={mask.shape}, ct={ct.shape}"
                )
        else:
            mask = np.zeros_like(ct, dtype=np.float32)

        cam, stats = _build_mock_gradcam(ct=ct, mask=mask, rng=rng)
        ct_norm = _robust_norm_ct(ct)

        stem = _strip_nii_suffix(image_path)
        cam_out = output_dir / f"{stem}_mock_gradcam.nii.gz"
        overlay_out = output_dir / f"{stem}_mock_overlay_rgb.nii.gz"
        heatmap_out = output_dir / f"{stem}_mock_gradcam_rgb.nii.gz"
        ct_out = output_dir / f"{stem}_ct_norm.nii.gz"

        _save_like_reference(cam, ct_nii, cam_out)
        _save_like_reference(ct_norm, ct_nii, ct_out)
        save_rgb_nifti(colorize_heatmap(cam), heatmap_out, reference=ct_nii)
        save_rgb_nifti(colorize_overlay(ct_norm, cam, alpha=alpha), overlay_out, reference=ct_nii)

        info = {
            "patient_dir": str(patient_dir),
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
            "seed": int(local_seed),
            "alpha": float(alpha),
            "shape": list(ct.shape),
            "stats": stats,
            "outputs": {
                "mock_gradcam": str(cam_out),
                "mock_gradcam_rgb": str(heatmap_out),
                "mock_overlay_rgb": str(overlay_out),
                "ct_norm": str(ct_out),
            },
        }

        info_out = output_dir / f"{stem}_mock_info.json"
        with open(info_out, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        run_summary["series"].append(info)
        written.extend([cam_out, heatmap_out, overlay_out, ct_out, info_out])

        print(f"[{idx + 1}/{len(pairs)}] Wrote mock Grad-CAM for {image_path.name}")
        print(f"  - {cam_out}")
        print(f"  - {heatmap_out}")
        print(f"  - {overlay_out}")
        print(f"  - {ct_out}")
        print(f"  - {info_out}")

    summary_out = output_dir / "mock_generation_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    written.append(summary_out)
    print(f"Saved run summary: {summary_out}")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation-friendly mock 3D Grad-CAM volumes from CT + mask.")
    parser.add_argument("--patient-dir", type=Path, default=Path(DEFAULT_PATIENT_DIR), help="Patient folder containing *_image.nii.gz and *_mask.nii.gz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save mock volumes")
    parser.add_argument("--alpha", type=float, default=0.40, help="Overlay alpha blend factor")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for texture generation")
    parser.add_argument("--series-index", type=int, default=-1, help="Series index to process (default -1 = all series in patient folder)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_mock_gradcam(
        patient_dir=args.patient_dir,
        output_dir=args.output_dir,
        alpha=float(args.alpha),
        seed=int(args.seed),
        series_index=int(args.series_index),
    )


if __name__ == "__main__":
    main()
