import os
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, EfficientNetBN, TorchVisionFCModel
import timm


class EfficientNet2DClassifier(nn.Module):
    """2D classifier: one axial slice -> EfficientNet-B0 (ImageNet-pretrained).

    The slice is fed single-channel. MONAI's EfficientNetBN re-initializes the
    stem conv *randomly* when ``in_channels != 3`` — it does not average the
    pretrained RGB weights. We fix that here by copying the mean of the
    pretrained 3-channel stem into our 1-channel stem, the standard grayscale-
    transfer trick (same thing TorchVisionResNet2DClassifier._adapt_stem does).
    Without this the frozen-backbone linear probe operates on a random stem,
    and even the full-DAPT run shows a 10-epoch Adeno-only cold start before
    the stem gradient catches up.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.efficientnet = EfficientNetBN(
            "efficientnet-b0",
            pretrained=True,
            spatial_dims=2,
            in_channels=1,
            num_classes=num_classes,
        )
        self._init_1ch_stem_from_rgb()

    def _init_1ch_stem_from_rgb(self) -> None:
        ref = EfficientNetBN(
            "efficientnet-b0", pretrained=True, spatial_dims=2, in_channels=3,
        )
        with torch.no_grad():
            rgb = ref._conv_stem.weight  # (32, 3, 3, 3)
            avg = rgb.mean(dim=1, keepdim=True)  # (32, 1, 3, 3)
            self.efficientnet._conv_stem.weight.copy_(avg)
        del ref

    def forward(self, x, return_segmentation=False):
        # The 2D pipeline outputs (B, 1, H, W).
        # Fallback to catch accidental depth dimensions:
        if x.ndim == 5:
            # e.g., (B, 1, H, W, 1) -> (B, 1, H, W)
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            # e.g., (B, 1, 1, H, W) -> (B, 1, H, W)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        
        cls_logits = self.efficientnet(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits


class DenseNet2DClassifier(nn.Module):
    """2D DenseNet121 (ImageNet-pretrained) for single-slice classification.
    MONAI's DenseNet121 builds the classification head when ``out_channels``
    equals the number of classes.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.densenet = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            pretrained=True,
        )

    def forward(self, x, return_segmentation=False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        cls_logits = self.densenet(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits


class TorchVisionResNet2DClassifier(nn.Module):
    """2D ResNet-50 wired via MONAI's TorchVisionFCModel with torchvision
    ImageNet weights. MONAI's ``in_channels`` kwarg doesn't actually swap the
    stem for torchvision backbones in this version, so we replace ``conv1``
    ourselves and average the RGB pretrained weights across the 3 channel dim
    (standard grayscale-transfer trick). FC head is replaced for ``num_classes``.
    """

    def __init__(self, num_classes: int = 3, model_name: str = "resnet50", in_channels: int = 1):
        super().__init__()
        self.backbone = TorchVisionFCModel(
            model_name=model_name,
            num_classes=num_classes,
            dim=2,
            pretrained=True,
            pool=None,
            use_conv=False,
        )
        if in_channels != 3:
            self._adapt_stem(in_channels)

    def _adapt_stem(self, in_channels: int) -> None:
        # torchvision ResNet stores the stem as ``.features.0`` after
        # MONAI's NetAdapter strips the FC head. Fall back to a recursive
        # search so we also handle ResNet18/34 and similar if swapped in.
        stem = None
        stem_parent = None
        stem_attr = None
        for parent_name, parent in self.backbone.named_modules():
            for attr, mod in parent.named_children():
                if isinstance(mod, nn.Conv2d) and mod.in_channels == 3 and mod.kernel_size == (7, 7):
                    stem = mod
                    stem_parent = parent
                    stem_attr = attr
                    break
            if stem is not None:
                break
        if stem is None:
            raise RuntimeError("Could not locate 3-channel 7x7 stem conv to adapt.")
        new_conv = nn.Conv2d(
            in_channels, stem.out_channels,
            kernel_size=stem.kernel_size, stride=stem.stride,
            padding=stem.padding, bias=stem.bias is not None,
        )
        with torch.no_grad():
            avg = stem.weight.mean(dim=1, keepdim=True)  # (out, 1, 7, 7)
            new_conv.weight.copy_(avg.repeat(1, in_channels, 1, 1))
            if stem.bias is not None:
                new_conv.bias.copy_(stem.bias)
        setattr(stem_parent, stem_attr, new_conv)

    def forward(self, x, return_segmentation=False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        cls_logits = self.backbone(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits




class SwinTiny2DClassifier(nn.Module):
    """2D Swin-Tiny (swin_tiny_patch4_window7_224) backbone with a
    RadImageNet-pretrained init, adapted to 1-channel CT slice classification.

    We use the ``timm`` implementation of Swin-Tiny because it's the maintained
    open-source reference, but the official RadImageNet checkpoint
    (``rin_swintf.pth``) was trained against the original Microsoft Swin code
    base, which numbers the ``PatchMerging`` downsample blocks one stage
    earlier than timm. Two differences need reconciling at load time:

      * timm stores downsample weights at ``layers.{1,2,3}.downsample.*``;
        MS stores them at ``layers.{0,1,2}.downsample.*``. We shift the
        indices by +1 on load.
      * The MS checkpoint includes pre-computed buffers
        (``attn.relative_position_index``, ``attn_mask``) that timm
        recomputes from scratch — drop them rather than force-load.

    The 3-channel patch_embed stem is averaged across RGB → 1-channel at load
    time, the same grayscale-transfer trick used by
    ``TorchVisionResNet2DClassifier`` and ``EfficientNet2DClassifier`` to
    preserve pretrained feature scale on a non-RGB input.
    """

    def __init__(self, num_classes: int = 3, radimagenet_ckpt: str = ""):
        super().__init__()


        # num_classes=3 at construction so timm builds a fresh 3-class head
        # (head.fc.{weight,bias}). The pretrained 165-class head from the
        # checkpoint is dropped in _load_radimagenet_ckpt — SCLC has nothing
        # to learn from RadImageNet class semantics.
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes,
            in_chans=1,
        )
        if radimagenet_ckpt and os.path.exists(radimagenet_ckpt):
            self._load_radimagenet_ckpt(radimagenet_ckpt)

    @staticmethod
    def _remap_ms_to_timm(state_dict: "dict") -> "dict":
        remapped = {}
        for k, v in state_dict.items():
            # MS layer i ↔ timm layer i+1 for the downsample path. timm's
            # layers.0.downsample is an Identity (no params), so index 0 is
            # never a target here.
            new_k = k
            for i_ms in (2, 1, 0):  # descending to avoid accidental chain-renames
                prefix = f"layers.{i_ms}.downsample."
                if k.startswith(prefix):
                    new_k = f"layers.{i_ms + 1}.downsample." + k[len(prefix):]
                    break
            remapped[new_k] = v
        return remapped

    def _load_radimagenet_ckpt(self, path: str) -> None:
        print(f"[*] SwinTiny2DClassifier: loading RadImageNet weights from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            sd = ck["model"]
        elif isinstance(ck, dict) and "state_dict" in ck:
            sd = ck["state_dict"]
        else:
            sd = ck

        # Drop buffers timm recomputes at construction — loading them here
        # would raise "unexpected key" without any benefit.
        sd = {
            k: v for k, v in sd.items()
            if not (k.endswith(".relative_position_index") or k.endswith(".attn_mask"))
        }
        # Drop the 165-class classification head; timm re-created a 3-class one.
        sd = {k: v for k, v in sd.items() if not (k == "head.weight" or k == "head.bias")}
        # Remap MS-style downsample keys to timm-style (+1 stage).
        sd = self._remap_ms_to_timm(sd)
        # Average the 3-channel stem → 1-channel weights. Matches
        # TorchVisionResNet2DClassifier._adapt_stem and
        # EfficientNet2DClassifier._init_1ch_stem_from_rgb.
        stem_key = "patch_embed.proj.weight"
        if stem_key in sd and sd[stem_key].shape[1] == 3:
            avg = sd[stem_key].mean(dim=1, keepdim=True)  # (96, 1, 4, 4)
            sd[stem_key] = avg

        missing, unexpected = self.swin.load_state_dict(sd, strict=False)
        matched = len(sd) - len(unexpected)
        print(
            f"[*] RadImageNet weights: matched {matched}/{len(sd)} keys "
            f"(unexpected={len(unexpected)}, missing={len(missing)})"
        )
        # Missing should be just the head (head.fc.weight, head.fc.bias). Anything
        # else in the missing list indicates the checkpoint/architecture drifted.
        unexpected_non_head = [k for k in unexpected if not k.startswith("head.")]
        missing_non_head = [k for k in missing if not k.startswith("head.")]
        if unexpected_non_head:
            print(f"[!] Unexpected non-head keys (investigate): {unexpected_non_head[:5]}")
        if missing_non_head:
            print(f"[!] Missing non-head keys (investigate): {missing_non_head[:5]}")

    def forward(self, x, return_segmentation=False):
        # Pipeline outputs (B, 1, H, W). Fallback for accidental 5D inputs.
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        cls_logits = self.swin(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits
