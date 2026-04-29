import os
import torch
import torch.nn as nn

from .swin_unetr import SwinUNETRClassifier
from .classifiers_2d import (
    EfficientNet2DClassifier,
    DenseNet2DClassifier,
    TorchVisionResNet2DClassifier,
    SwinTiny2DClassifier,
)
from .classifiers_rin import (
    RadImageNetResNet502DClassifier,
    RadImageNetDenseNet1212DClassifier,
)
from .classifiers_mil import MILResNet50Classifier, MILSwinTinyClassifier


TWO_D_MODEL_TYPES = (
    "efficientnet_b0_2d",
    "densenet121_2d",
    "resnet50_2d",
    "swin_tiny_2d",
    "resnet50_2d_rin",
    "densenet121_2d_rin",
)
MIL_MODEL_TYPES = (
    "mil_resnet50",
    "mil_swin_tiny",
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


def get_sclc_model(checkpoint_path: str = "", model_type: str = "swin_unetr", in_channels: int = 1, depth_size: int = 128, mil_mode: str = "att", mil_trans_blocks: int = 4, mil_trans_dropout: float = 0.0) -> nn.Module:
    if model_type.lower() == "mil_resnet50":
        # MIL model for the BigLunge fine-tune / inference phase.
        # DAPT for this model_type uses TorchVisionResNet2DClassifier instead
        # — callers route that via main.py (see the mil_resnet50 branch there),
        # not here, because the DAPT architecture is a plain 2D classifier.
        model = MILResNet50Classifier(
            num_classes=3, mil_mode=mil_mode,
            trans_blocks=mil_trans_blocks, trans_dropout=mil_trans_dropout,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # Two possibilities for checkpoint_path:
            #   (a) a DAPT TorchVisionResNet2DClassifier checkpoint — keys
            #       prefixed 'backbone.features.' ; load only the backbone.
            #   (b) a prior MIL checkpoint — full MILResNet50Classifier state
            #       dict; load strict=False.
            probe_keys = list(state_dict.keys())
            is_dapt_ckpt = any(k.startswith("backbone.features.") for k in probe_keys)
            if is_dapt_ckpt:
                print(f"[*] Loading DAPT ResNet50 backbone into MIL model from {checkpoint_path}")
                model.load_backbone_from_dapt(state_dict)
            else:
                print(f"[*] Loading MIL checkpoint from {checkpoint_path}")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                matched = len(state_dict) - len(unexpected)
                print(f"[*] Matched {matched}/{len(state_dict)} keys (missing={len(missing)}).")
        return model
    if model_type.lower() == "mil_swin_tiny":
        # MIL model for the BigLunge fine-tune / inference phase, Swin-Tiny
        # backbone variant. DAPT for this model_type uses SwinTiny2DClassifier
        # (routed via main.py); this branch builds the MIL classifier and
        # accepts either a DAPT SwinTiny2DClassifier checkpoint (keys prefixed
        # 'swin.') or a prior MIL-Swin checkpoint.
        #
        # checkpoint_path semantics here mirror the ``swin_tiny_2d`` branch:
        # an empty string falls back to the RadImageNet default inside
        # MILSwinTinyClassifier; a non-empty path is detected as either
        # RadImageNet or DAPT/MIL by inspecting the state_dict keys.
        is_radimagenet = bool(checkpoint_path) and "radimagenet" in checkpoint_path.lower()
        if is_radimagenet or not checkpoint_path:
            # Let MILSwinTinyClassifier load RadImageNet directly (uses default
            # path when checkpoint_path is empty).
            model = MILSwinTinyClassifier(
                num_classes=3, mil_mode=mil_mode,
                radimagenet_ckpt=checkpoint_path,
                trans_blocks=mil_trans_blocks, trans_dropout=mil_trans_dropout,
            )
            return model

        # Non-RadImageNet checkpoint: build a fresh model (with default RIN
        # init) and overlay the user-supplied state dict.
        model = MILSwinTinyClassifier(
            num_classes=3, mil_mode=mil_mode,
            trans_blocks=mil_trans_blocks, trans_dropout=mil_trans_dropout,
        )
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            probe_keys = list(state_dict.keys())
            is_dapt_ckpt = any(k.startswith("swin.") for k in probe_keys)
            if is_dapt_ckpt:
                print(f"[*] Loading DAPT SwinTiny backbone into MIL-Swin model from {checkpoint_path}")
                model.load_backbone_from_dapt(state_dict)
            else:
                print(f"[*] Loading MIL-Swin checkpoint from {checkpoint_path}")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                matched = len(state_dict) - len(unexpected)
                print(f"[*] Matched {matched}/{len(state_dict)} keys (missing={len(missing)}).")
        return model
    if model_type.lower() == "efficientnet_b0_2d":
        model = EfficientNet2DClassifier(num_classes=3)
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
        model = DenseNet2DClassifier(num_classes=3)
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
        model = TorchVisionResNet2DClassifier(num_classes=3, model_name="resnet50")
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading 2D checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "resnet50_2d_rin":
        # checkpoint_path here is overloaded: a Microsoft-style RadImageNet
        # ResNet50 dump (keys like ``backbone.<idx>.<...>``) or a previously
        # saved SCLC checkpoint (``backbone.<torchvision_name>.<...>`` plus
        # ``classification_head.*``). Detected by inspecting key prefixes.
        rin_default = "/home/data/RadImageNet/ResNet50/ResNet50.pt"
        rin_path = checkpoint_path if checkpoint_path else rin_default
        loaded_via_constructor = False
        sd_for_after_build = None
        if rin_path and os.path.exists(rin_path):
            probe = torch.load(rin_path, map_location="cpu", weights_only=False)
            if isinstance(probe, dict) and "state_dict" in probe:
                probe = probe["state_dict"]
            keys = list(probe.keys()) if isinstance(probe, dict) else []
            is_ms_rin = any(k.startswith("backbone.") and k.split(".", 2)[1].isdigit() for k in keys)
            if is_ms_rin:
                model = RadImageNetResNet502DClassifier(num_classes=3, in_channels=in_channels, radimagenet_ckpt=rin_path)
                loaded_via_constructor = True
            else:
                model = RadImageNetResNet502DClassifier(num_classes=3, in_channels=in_channels, radimagenet_ckpt="")
                sd_for_after_build = probe
        else:
            print(f"[!] resnet50_2d_rin: checkpoint not found at {rin_path}; backbone is random-initialized.")
            model = RadImageNetResNet502DClassifier(num_classes=3, in_channels=in_channels, radimagenet_ckpt="")
        if not loaded_via_constructor and sd_for_after_build is not None:
            print(f"[*] Loading SCLC checkpoint into resnet50_2d_rin from {rin_path}")
            missing, unexpected = model.load_state_dict(sd_for_after_build, strict=False)
            matched = len(sd_for_after_build) - len(unexpected)
            print(f"[*] Matched {matched}/{len(sd_for_after_build)} keys (missing={len(missing)}).")
        return model
    if model_type.lower() == "densenet121_2d_rin":
        rin_default = "/home/data/RadImageNet/DenseNet/DenseNet121.pt"
        rin_path = checkpoint_path if checkpoint_path else rin_default
        loaded_via_constructor = False
        sd_for_after_build = None
        if rin_path and os.path.exists(rin_path):
            probe = torch.load(rin_path, map_location="cpu", weights_only=False)
            if isinstance(probe, dict) and "state_dict" in probe:
                probe = probe["state_dict"]
            keys = list(probe.keys()) if isinstance(probe, dict) else []
            is_ms_rin = any(k.startswith("backbone.0.") for k in keys)
            if is_ms_rin:
                model = RadImageNetDenseNet1212DClassifier(num_classes=3, in_channels=in_channels, radimagenet_ckpt=rin_path)
                loaded_via_constructor = True
            else:
                model = RadImageNetDenseNet1212DClassifier(num_classes=3, in_channels=in_channels, radimagenet_ckpt="")
                sd_for_after_build = probe
        else:
            print(f"[!] densenet121_2d_rin: checkpoint not found at {rin_path}; backbone is random-initialized.")
            model = RadImageNetDenseNet1212DClassifier(num_classes=3, in_channels=in_channels, radimagenet_ckpt="")
        if not loaded_via_constructor and sd_for_after_build is not None:
            print(f"[*] Loading SCLC checkpoint into densenet121_2d_rin from {rin_path}")
            missing, unexpected = model.load_state_dict(sd_for_after_build, strict=False)
            matched = len(sd_for_after_build) - len(unexpected)
            print(f"[*] Matched {matched}/{len(sd_for_after_build)} keys (missing={len(missing)}).")
        return model
    if model_type.lower() == "swin_tiny_2d":
        # checkpoint_path here is overloaded: it may be either the RadImageNet
        # pretrain (untouched 165-class head, MS-style layer keys) or a
        # previously saved SCLC checkpoint (3-class head, timm-style keys).
        # We route RadImageNet through the wrapper's key-remapping loader;
        # an SCLC checkpoint is loaded after construction the same way the
        # other 2D branches do it.
        is_radimagenet = bool(checkpoint_path) and "radimagenet" in checkpoint_path.lower()
        if not is_radimagenet and checkpoint_path and os.path.exists(checkpoint_path):
            # Quick peek: if top-level has a 'model' key with 'head.weight'
            # shape[0] != 3, it's a RadImageNet-style dump; otherwise SCLC.
            probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            probe_sd = probe.get("model", probe) if isinstance(probe, dict) else probe
            head_w = probe_sd.get("head.weight") if isinstance(probe_sd, dict) else None
            if head_w is not None and int(head_w.shape[0]) != 3:
                is_radimagenet = True

        if is_radimagenet:
            model = SwinTiny2DClassifier(num_classes=3, radimagenet_ckpt=checkpoint_path)
        else:
            model = SwinTiny2DClassifier(num_classes=3)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"[*] Loading SCLC checkpoint from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                matched = len(state_dict) - len(unexpected)
                print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "swin_unetr":
        model = SwinUNETRClassifier(in_channels=in_channels, num_classes=3)
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print(f"[*] Loading pretrained SwinUNETR weights from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                # MONAI checkpoints sometimes wrap the weights in a 'state_dict' key
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                
                # Load into the underlying swin_unetr backbone
                # Pop out the final segmentation layer mismatched sizes so strict=False can work efficiently
                if 'out.conv.conv.weight' in state_dict:
                    state_dict.pop('out.conv.conv.weight')
                if 'out.conv.conv.bias' in state_dict:
                    state_dict.pop('out.conv.conv.bias')
                        
                missing, unexpected = model.swin_unetr.load_state_dict(state_dict, strict=False)
                matched = len(state_dict) - len(unexpected)
                print(f"[*] Pretrained weights loaded. Matched {matched}/{len(state_dict)} keys.")
                if matched == 0:
                    print(f"[!] Warning: 0 keys matched! Checkpoint {checkpoint_path} is likely for a different architecture.")
            else:
                print(f"[!] Warning: Checkpoint path {checkpoint_path} does not exist. Initializing from scratch.")
    return model
