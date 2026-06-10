import os

import torch
import torch.nn as nn

from .classifiers_2d import (
    DenseNet2DClassifier,
    EfficientNet2DClassifier,
    SwinV2Tiny2DClassifier,
    TorchVisionResNet2DClassifier,
)
from .classifiers_mil import MILSwinV2TinyClassifier
from .swin_unetr import SwinUNETRClassifier

TWO_D_MODEL_TYPES = (
    "efficientnet_b0_2d",
    "densenet121_2d",
    "resnet50_2d",
    "swinv2_tiny_2d",
)
MIL_MODEL_TYPES = (
    "mil_swinv2_tiny",
)


def is_2d_model_type(model_type: str) -> bool:
    return model_type.lower() in TWO_D_MODEL_TYPES


def is_mil_model_type(model_type: str) -> bool:
    return model_type.lower() in MIL_MODEL_TYPES


def get_pipeline(model_type: str) -> str:
    if is_mil_model_type(model_type):
        return "mil"
    if is_2d_model_type(model_type):
        return "2d"
    return "3d"


def get_sclc_model(
    checkpoint_path: str = "",
    model_type: str = "swin_unetr",
    in_channels: int = 1,
    depth_size: int = 128,
    mil_mode: str = "att",
    mil_trans_blocks: int = 4,
    mil_trans_dropout: float = 0.0,
    use_advanced_fpn: bool = False,
    use_det_seg: bool = False,
    fpn_channels: int = 256,
    tfpn_enabled: bool = True,
    tfpn_heads: int = 4,
    tfpn_layers: int = 1,
    tfpn_levels: int = 1,
    num_classes: int = 3,
) -> nn.Module:
    if model_type.lower() == "mil_swinv2_tiny":
        model = MILSwinV2TinyClassifier(
            num_classes=num_classes,
            mil_mode=mil_mode,
            trans_blocks=mil_trans_blocks,
            trans_dropout=mil_trans_dropout,
            use_advanced_fpn=use_advanced_fpn,
            use_det_seg=use_det_seg,
            fpn_channels=fpn_channels,
            tfpn_enabled=tfpn_enabled,
            tfpn_heads=tfpn_heads,
            tfpn_layers=tfpn_layers,
            tfpn_levels=tfpn_levels,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            probe_keys = list(state_dict.keys())
            is_dapt_ckpt = any(k.startswith("swin.") for k in probe_keys)
            if is_dapt_ckpt:
                print(f"[*] Loading DAPT SwinV2-Tiny backbone into MIL-SwinV2Tiny model from {checkpoint_path}")
                model.load_backbone_from_dapt(state_dict)
            else:
                print(f"[*] Loading MIL-SwinV2Tiny checkpoint from {checkpoint_path}")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                matched = len(state_dict) - len(unexpected)
                print(f"[*] Matched {matched}/{len(state_dict)} keys (missing={len(missing)}).")
        return model
    if model_type.lower() == "efficientnet_b0_2d":
        model = EfficientNet2DClassifier(
            num_classes=num_classes,
            use_advanced_fpn=use_advanced_fpn,
            use_det_seg=use_det_seg,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading 2D checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "densenet121_2d":
        model = DenseNet2DClassifier(
            num_classes=num_classes,
            use_advanced_fpn=use_advanced_fpn,
            use_det_seg=use_det_seg,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading 2D checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "resnet50_2d":
        model = TorchVisionResNet2DClassifier(
            num_classes=num_classes,
            model_name="resnet50",
            use_advanced_fpn=use_advanced_fpn,
            use_det_seg=use_det_seg,
            fpn_channels=fpn_channels,
            tfpn_enabled=tfpn_enabled,
            tfpn_heads=tfpn_heads,
            tfpn_layers=tfpn_layers,
            tfpn_levels=tfpn_levels,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading 2D checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "swinv2_tiny_2d":
        model = SwinV2Tiny2DClassifier(
            num_classes=num_classes,
            use_advanced_fpn=use_advanced_fpn,
            use_det_seg=use_det_seg,
            fpn_channels=fpn_channels,
            tfpn_enabled=tfpn_enabled,
            tfpn_heads=tfpn_heads,
            tfpn_layers=tfpn_layers,
            tfpn_levels=tfpn_levels,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading SwinV2-Tiny checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "swin_unetr":
        model = SwinUNETRClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            use_advanced_fpn=use_advanced_fpn,
            use_det_seg=use_det_seg,
            fpn_channels=fpn_channels,
            tfpn_enabled=tfpn_enabled,
            tfpn_heads=tfpn_heads,
            tfpn_layers=tfpn_layers,
            tfpn_levels=tfpn_levels,
        )
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print(f"[*] Loading SwinUNETR weights from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # MONAI BTCV checkpoint's seg head has 14 output channels; drop
                # so our 3-class head loads strict=False without a shape clash.
                if 'out.conv.conv.weight' in state_dict:
                    state_dict.pop('out.conv.conv.weight')
                if 'out.conv.conv.bias' in state_dict:
                    state_dict.pop('out.conv.conv.bias')
                missing, unexpected = model.swin_unetr.load_state_dict(state_dict, strict=False)
                matched = len(state_dict) - len(unexpected)
                print(f"[*] Pretrained weights loaded. Matched {matched}/{len(state_dict)} keys.")
                if matched == 0:
                    raise RuntimeError(
                        f"Loaded 0 keys from {checkpoint_path}; checkpoint format "
                        f"does not match SwinUNETRClassifier or the BTCV pretrained layout."
                    )
            else:
                print(f"[!] Warning: Checkpoint path {checkpoint_path} does not exist. Initializing from scratch.")
    return model
