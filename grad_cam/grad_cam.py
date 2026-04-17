from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.visualize import GradCAMpp as GradCAM

# Allow running this file directly: `python grad_cam/grad_cam.py ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import get_val_transforms_3d
from model_selection import get_sclc_model


CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]

# Default target layers must output a spatial tensor [B, C, D, H, W].
# MONAI's GradCAM registers forward/backward hooks here and does the
# weighted sum + upsample + ReLU + normalize for us.
#
# - swin_unetr: layers4[0] is the last BasicLayer of the Swin encoder; its
#   forward returns a single spatial tensor (not a list like swinViT does).
# - resnet/densenet: the last conv stage is the canonical Grad-CAM target.
# - vit / models_genesis: not supported here. ViT's final block outputs a
#   token sequence [B, N, C] (would require a reshape wrapper), and
#   ModelsGenesis' down_tr512 returns a tuple which MONAI's hook can't
#   consume directly. Point at a different layer via --target-layer or
#   add a wrapper module if you need them.
DEFAULT_TARGET_LAYER = {
	"swin_unetr": "swin_unetr.swinViT.layers4.0",
	"resnet50": "resnet.layer4",
	"resnet18": "resnet.layer4",
	"densenet121": "densenet.features.denseblock4",
}

SUPPORTED_MODEL_TYPES = tuple(DEFAULT_TARGET_LAYER.keys())


def _strip_nii_suffix(path: str) -> str:
	name = Path(path).name
	if name.endswith(".nii.gz"):
		return name[:-7]
	if name.endswith(".nii"):
		return name[:-4]
	return Path(name).stem


def _unwrap_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
	"""Extract a model state_dict from common checkpoint wrappers."""
	state = ckpt
	if isinstance(state, dict):
		for key in ("state_dict", "model_state_dict", "model"):
			if key in state and isinstance(state[key], dict):
				state = state[key]
				break

	if not isinstance(state, dict):
		raise ValueError("Checkpoint does not contain a valid state_dict.")

	if any(k.startswith("module.") for k in state.keys()):
		state = {k.replace("module.", "", 1): v for k, v in state.items()}
	return state


def _normalize_np_01(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	vmin = float(np.min(volume))
	vmax = float(np.max(volume))
	return (volume - vmin) / (vmax - vmin + eps)


def _save_nifti(volume: np.ndarray, out_path: str) -> None:
	nii = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
	nib.save(nii, out_path)


def _patient_name_from_image_path(image_path: str) -> str:
	"""Return patient folder name from an input image path."""
	parent_name = Path(image_path).resolve().parent.name.strip()
	return parent_name if parent_name else "unknown_patient"


def _build_input_tensor(
	image_path: str,
	img_size: int,
	depth_size: int,
	device: torch.device,
) -> torch.Tensor:
	transforms = get_val_transforms_3d(img_size=img_size, depth_size=depth_size)

	data: Dict[str, Any] = {"image": image_path}

	transformed = transforms(data)
	image_tensor = transformed["image"]
	if not torch.is_tensor(image_tensor):
		image_tensor = torch.as_tensor(image_tensor)

	image_tensor = image_tensor.float()
	if image_tensor.ndim == 4:
		image_tensor = image_tensor.unsqueeze(0)
	if image_tensor.ndim != 5:
		raise RuntimeError(f"Expected transformed image shape [B,C,D,H,W]-like, got {tuple(image_tensor.shape)}")

	return image_tensor.to(device)


def _load_trained_model(
	checkpoint_path: str,
	model_type: str,
	depth_size: int,
	device: torch.device,
) -> nn.Module:
	model = get_sclc_model(
		checkpoint_path="",
		model_type=model_type,
		in_channels=1,
		depth_size=depth_size,
	)
	ckpt = torch.load(checkpoint_path, map_location=device)
	state_dict = _unwrap_state_dict(ckpt)
	missing, unexpected = model.load_state_dict(state_dict, strict=False)

	matched = len(state_dict) - len(unexpected)
	print(f"Loaded checkpoint: matched {matched}/{len(state_dict)} keys.")
	if missing:
		print(f"Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
	if unexpected:
		print(f"Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
	if matched == 0:
		raise RuntimeError("0 keys matched from checkpoint. Check model type and checkpoint path.")

	model = model.to(device)
	model.eval()
	return model


def use_grad_cam(
	image_path: str,
	checkpoint_path: str,
	model_type: str = "swin_unetr",
	class_index: Optional[int] = None,
	target_layer: Optional[str] = None,
	output_dir: Optional[str] = None,
	img_size: int = 224,
	depth_size: int = 128,
	alpha: float = 0.35,
	device: Optional[str] = None,
) -> Dict[str, Any]:
	"""Generate a 3D Grad-CAM heatmap via monai.visualize.GradCAM and save as NIfTI.

	Outputs are saved in `output_dir/<patient_name>` where patient_name is inferred
	from the parent folder of --image. Default root is `grad_cam/gradcam_output/`.

	Saved files:
	- preprocessed input volume
	- grad-cam heatmap
	- heatmap overlay on input
	- json metadata (prediction probabilities)
	"""
	if not os.path.isfile(image_path):
		raise ValueError(f"Image file does not exist: {image_path}")
	if not os.path.isfile(checkpoint_path):
		raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")

	model_type = model_type.lower()
	if model_type not in SUPPORTED_MODEL_TYPES:
		raise ValueError(
			f"Unsupported model_type '{model_type}'. Supported: {list(SUPPORTED_MODEL_TYPES)}. "
			f"(ViT emits a token sequence and ModelsGenesis returns tuples from its encoder "
			f"blocks — both need a wrapper module before they can feed monai.visualize.GradCAM.)"
		)

	run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
	output_root = Path(output_dir) if output_dir else (Path(__file__).resolve().parent / "gradcam_output")
	patient_name = _patient_name_from_image_path(image_path)
	out_dir = output_root / patient_name
	out_dir.mkdir(parents=True, exist_ok=True)

	model = _load_trained_model(
		checkpoint_path=checkpoint_path,
		model_type=model_type,
		depth_size=depth_size,
		device=run_device,
	)

	input_tensor = _build_input_tensor(
		image_path=image_path,
		img_size=img_size,
		depth_size=depth_size,
		device=run_device,
	)

	target = target_layer or DEFAULT_TARGET_LAYER[model_type]

	# Grad-CAM requires gradients even though the model is in eval() mode.
	# MONAI's GradCAM handles hook registration, backward pass, spatial
	# reshape, ReLU, upsample to input resolution, and per-sample 0-1
	# normalization internally — we only need to call it.
	with torch.enable_grad():
		with torch.no_grad():
			logits = model(input_tensor)
		if logits.ndim == 1:
			logits = logits.unsqueeze(0)
		pred_class = int(torch.argmax(logits, dim=1).item())
		target_class = pred_class if class_index is None else int(class_index)
		if target_class < 0 or target_class >= logits.shape[1]:
			raise ValueError(f"class_index={target_class} is out of range [0, {logits.shape[1] - 1}].")

		cam_engine = GradCAM(nn_module=model, target_layers=target)
		heatmap = cam_engine(x=input_tensor, class_idx=target_class)  # [B, 1, D, H, W]

	cam_np = heatmap[0, 0].detach().cpu().numpy()
	base_np = input_tensor[0, 0].detach().cpu().numpy()
	base_np = _normalize_np_01(base_np)
	overlay_np = np.clip((1.0 - alpha) * base_np + alpha * cam_np, 0.0, 1.0)

	probs = torch.softmax(logits.detach(), dim=1)

	stem = _strip_nii_suffix(image_path)
	cam_out = out_dir / f"{stem}_gradcam_target{target_class}_pred{pred_class}.nii.gz"
	overlay_out = out_dir / f"{stem}_gradcam_overlay_target{target_class}_pred{pred_class}.nii.gz"
	preproc_out = out_dir / f"{stem}_preprocessed_input.nii.gz"
	info_out = out_dir / f"{stem}_gradcam_info.json"

	_save_nifti(cam_np, str(cam_out))
	_save_nifti(overlay_np, str(overlay_out))
	_save_nifti(base_np, str(preproc_out))

	class_probs = probs[0].detach().cpu().tolist()
	info = {
		"image_path": image_path,
		"checkpoint_path": checkpoint_path,
		"patient_name": patient_name,
		"model_type": model_type,
		"target_layer": target,
		"pred_class": pred_class,
		"target_class": target_class,
		"pred_label": CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class),
		"target_label": CLASS_NAMES[target_class] if target_class < len(CLASS_NAMES) else str(target_class),
		"probabilities": class_probs,
		"logits": logits[0].detach().cpu().tolist(),
		"saved_files": {
			"gradcam": str(cam_out),
			"overlay": str(overlay_out),
			"preprocessed_input": str(preproc_out),
		},
	}

	with open(info_out, "w", encoding="utf-8") as f:
		json.dump(info, f, indent=2)

	print("\nGrad-CAM complete")
	print(f"  Predicted class: {pred_class} ({info['pred_label']})")
	print(f"  Target class:    {target_class} ({info['target_label']})")
	print(f"  Target layer:    {target}")
	print(f"  Output folder:   {out_dir}")
	print(f"  Saved heatmap:   {cam_out}")
	print(f"  Saved overlay:   {overlay_out}")
	print(f"  Saved input:     {preproc_out}")
	print(f"  Saved metadata:  {info_out}")

	return info


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate 3D Grad-CAM via monai.visualize and save as NIfTI.")
	parser.add_argument("--image", required=True, help="Path to CT NIfTI (.nii or .nii.gz)")
	parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pth)")
	parser.add_argument(
		"--model-type",
		default="swin_unetr",
		choices=list(SUPPORTED_MODEL_TYPES),
		help="Model type used for the checkpoint",
	)
	parser.add_argument("--class-index", type=int, default=None, help="Target class for Grad-CAM (default: predicted class)")
	parser.add_argument("--target-layer", type=str, default=None, help="Override target layer path (must output a spatial tensor)")
	parser.add_argument("--img-size", type=int, default=224, help="Input XY size used by transforms")
	parser.add_argument("--depth-size", type=int, default=128, help="Input depth used by transforms")
	parser.add_argument("--alpha", type=float, default=0.35, help="Overlay blend factor (0..1)")
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(Path(__file__).resolve().parent / "gradcam_output"),
		help="Root directory for saved outputs; files are written to <output-dir>/<patient_name>/.",
	)
	parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. cuda, cuda:0, cpu)")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	use_grad_cam(
		image_path=args.image,
		checkpoint_path=args.checkpoint,
		model_type=args.model_type,
		class_index=args.class_index,
		target_layer=args.target_layer,
		output_dir=args.output_dir,
		img_size=args.img_size,
		depth_size=args.depth_size,
		alpha=args.alpha,
		device=args.device,
	)


if __name__ == "__main__":
	main()
