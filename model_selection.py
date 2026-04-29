import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR, DenseNet121, EfficientNetBN, TorchVisionFCModel, MILModel
import os
import timm
# Direct torchvision imports — used only by the RadImageNet 2D variants
# (resnet50_2d_rin, densenet121_2d_rin), which build their backbones from
# torchvision and load Microsoft-style RadImageNet checkpoints. The
# existing MONAI-based 2D loaders (resnet50_2d, densenet121_2d) are
# unaffected.
from torchvision.models import resnet50 as _tv_resnet50, densenet121 as _tv_densenet121


class SwinUNETRClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # We initialize the base SwinUNETR.
        # (out_channels is required but ignored since we bypass the decoder for now)
        # use_v2=False matches the BTCV V1 checkpoint we load
        # (`model_swin_unetr_btcv_segmentation_v1.pt`). use_v2=True adds
        # `swinViT.layers{1,2,3}c.*` residual conv blocks (~6% of encoder
        # params) that have no V1 source weights and would silently start
        # from random init under strict=False. Empirical key coverage
        # comparison (2026-04-29): V1 model loads 157/159 keys (98.7%),
        # V2 model loads 157/167 keys (94.0%) with 10 randomly-initialized
        # V2-only weights — flip back to V2 only when a V2-trained ckpt
        # is wired up.
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=1,
            feature_size=48,
            spatial_dims=3,
            use_checkpoint=True,
            use_v2=False,
        )
        
        # In MONAI's SwinUNETR, the deepest feature channel size is feature_size * 16 (i.e., 768)
        bottleneck_channels = 48 * 16

        self.global_pool = nn.AdaptiveMaxPool3d(1)
        # Single linear head: with ~300 training volumes and 3 classes, a 2-layer
        # head (197K params) overfits. The pretrained encoder features should be
        # linearly separable if they encode the right information.
        self.classification_head = nn.Linear(bottleneck_channels, num_classes)
        
        # Hook captures the deepest swinViT feature map during the full
        # forward pass (used only when return_segmentation=True, which calls
        # self.swin_unetr(x) and we have no other way to grab the encoder
        # bottleneck without re-running the encoder). Gate by a flag so the
        # hook is a no-op when we hit swinViT directly in the cls-only path.
        self.deepest_features = None
        self._capture_deepest = False
        def vit_hook(module, input, output):
            if self._capture_deepest:
                self.deepest_features = output[-1]
        self.swin_unetr.swinViT.register_forward_hook(vit_hook)

    def forward(self, x, return_segmentation=False):
        self.deepest_features = None
        if return_segmentation:
            # populates self.deepest_features with the final encoder bottleneck before the decoder.
            self._capture_deepest = True
            try:
                seg_logits = self.swin_unetr(x)
            finally:
                self._capture_deepest = False

            # Pool spatially to [B, C, 1, 1, 1] then flatten to [B, C]
            pooled = self.global_pool(self.deepest_features).flatten(1)

            # Get class logits [B, num_classes]
            cls_logits = self.classification_head(pooled)
            return cls_logits, seg_logits
        else:
            # Bypass the heavy decoder entirely for classification-only runs.
            # The hook fires here too (we call swinViT directly), but
            # _capture_deepest=False so it's a no-op write.
            hidden_states = self.swin_unetr.swinViT(x.contiguous())
            bottleneck = hidden_states[-1]
            pooled = self.global_pool(bottleneck).flatten(1)
            cls_logits = self.classification_head(pooled)
            return cls_logits



# Pipeline selection by model-type suffix.
#   _2d  -> single axial slice containing tumor, 2D CNN (in_channels=1)
#   mil_ -> attention-MIL bag of axial slices
#   else -> full 3D volume
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


class RadImageNetResNet502DClassifier(nn.Module):
    """2D ResNet-50 with **RadImageNet** pretrained weights, 1-channel CT.

    Drop-in alternative to ``TorchVisionResNet2DClassifier`` (which uses
    MONAI's TorchVisionFCModel + ImageNet weights). Built directly from
    ``torchvision.models.resnet50`` so the Microsoft-style RadImageNet
    checkpoint at ``/home/data/RadImageNet/ResNet50/ResNet50.pt`` can be
    loaded after a small key-prefix remap.

    **Checkpoint layout (Microsoft RadImageNet release).** Keys look like
    ``backbone.<idx>.<rest>`` where ``<idx>`` indexes an
    ``nn.Sequential(conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)``.
    We remap idx ``0,1,4,5,6,7`` to torchvision names
    ``conv1, bn1, layer1, layer2, layer3, layer4`` and load non-strict.
    The release ships encoder-only (no classifier head) — our fresh
    ``classification_head`` linear layer takes its place.

    **1-ch stem averaging.** Stem is loaded at 3 channels (matching the
    RadImageNet RGB checkpoint) and then averaged to 1 channel post-load,
    same trick used by ``TorchVisionResNet2DClassifier`` and the
    ``SwinTiny2DClassifier``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        radimagenet_ckpt: str = "/home/data/RadImageNet/ResNet50/ResNet50.pt",
    ):
        super().__init__()
        # Build a vanilla 3-ch ResNet50 (random init); the RadImageNet
        # checkpoint will overwrite the encoder weights, then we swap the
        # 3-ch stem for a 1-ch one initialized from the RGB mean.
        self.backbone = _tv_resnet50(weights=None)
        # Drop the random-init fc; our classification_head replaces it.
        self.backbone.fc = nn.Identity()

        if radimagenet_ckpt and os.path.exists(radimagenet_ckpt):
            self._load_radimagenet_ckpt(radimagenet_ckpt)
        else:
            print(
                f"[!] RadImageNet ResNet50 ckpt not found at {radimagenet_ckpt}; "
                f"backbone is random-initialized."
            )

        if in_channels != 3:
            self._adapt_stem(in_channels)

        # 2048 = ResNet50's penultimate feature dim (after avgpool+flatten).
        self.classification_head = nn.Linear(2048, num_classes)

    @staticmethod
    def _remap_ms_to_torchvision(state_dict: dict) -> dict:
        """``backbone.<idx>.<rest>`` → ``<torchvision_name>.<rest>`` for the
        ResNet50 sequential index → name mapping. Non-matching keys are
        silently dropped (e.g., a future release that adds a head).
        """
        idx_to_name = {
            "0": "conv1", "1": "bn1",
            "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4",
        }
        out: dict = {}
        for k, v in state_dict.items():
            if not k.startswith("backbone."):
                continue
            rest = k[len("backbone."):]
            idx, _, tail = rest.partition(".")
            name = idx_to_name.get(idx)
            if name is None:
                continue
            out[f"{name}.{tail}"] = v
        return out

    def _load_radimagenet_ckpt(self, path: str) -> None:
        print(f"[*] RadImageNetResNet502DClassifier: loading weights from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            sd = ck["state_dict"]
        elif isinstance(ck, dict) and "model" in ck:
            sd = ck["model"]
        else:
            sd = ck
        remapped = self._remap_ms_to_torchvision(sd)
        target_n = len(self.backbone.state_dict())
        missing, unexpected = self.backbone.load_state_dict(remapped, strict=False)
        matched = target_n - len(missing)
        print(
            f"[*] RadImageNet ResNet50: matched {matched}/{target_n} backbone keys "
            f"(source contributed {len(remapped)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        # Missing keys we'd expect: fc.* (we replaced fc with Identity).
        non_fc_missing = [k for k in missing if not k.startswith("fc.")]
        if non_fc_missing:
            print(f"[!] RIN-ResNet50: unexplained missing keys: {non_fc_missing[:5]}")

    def _adapt_stem(self, in_channels: int) -> None:
        stem = self.backbone.conv1  # (64, 3, 7, 7)
        if stem.in_channels == in_channels:
            return
        new_conv = nn.Conv2d(
            in_channels, stem.out_channels,
            kernel_size=stem.kernel_size, stride=stem.stride,
            padding=stem.padding, bias=stem.bias is not None,
        )
        with torch.no_grad():
            avg = stem.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            new_conv.weight.copy_(avg.repeat(1, in_channels, 1, 1))
            if stem.bias is not None:
                new_conv.bias.copy_(stem.bias)
        self.backbone.conv1 = new_conv

    def forward(self, x, return_segmentation: bool = False):
        # Mirror the existing 2D wrappers' fallback for accidental 5D input.
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        feats = self.backbone(x)               # (B, 2048)
        cls_logits = self.classification_head(feats)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits


class RadImageNetDenseNet1212DClassifier(nn.Module):
    """2D DenseNet-121 with **RadImageNet** pretrained weights, 1-channel CT.

    Drop-in alternative to ``DenseNet2DClassifier`` (MONAI DenseNet121 +
    ImageNet weights). Built from ``torchvision.models.densenet121`` so the
    Microsoft-style RadImageNet checkpoint at
    ``/home/data/RadImageNet/DenseNet/DenseNet121.pt`` can be loaded with a
    one-prefix remap.

    **Checkpoint layout.** All encoder keys live under ``backbone.0.<rest>``;
    ``backbone.0`` IS the entire ``features`` module of torchvision DenseNet121.
    We strip ``backbone.0.`` and prepend ``features.`` for a clean load. No
    classifier head is shipped — our ``classification_head`` provides one.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        radimagenet_ckpt: str = "/home/data/RadImageNet/DenseNet/DenseNet121.pt",
    ):
        super().__init__()
        self.densenet = _tv_densenet121(weights=None)
        self.densenet.classifier = nn.Identity()

        if radimagenet_ckpt and os.path.exists(radimagenet_ckpt):
            self._load_radimagenet_ckpt(radimagenet_ckpt)
        else:
            print(
                f"[!] RadImageNet DenseNet121 ckpt not found at {radimagenet_ckpt}; "
                f"backbone is random-initialized."
            )

        if in_channels != 3:
            self._adapt_stem(in_channels)

        # 1024 = DenseNet121 feature dim post-features-norm-relu-pool.
        self.classification_head = nn.Linear(1024, num_classes)

    @staticmethod
    def _remap_ms_to_torchvision(state_dict: dict) -> dict:
        out: dict = {}
        for k, v in state_dict.items():
            if not k.startswith("backbone.0."):
                continue
            tail = k[len("backbone.0."):]
            out[f"features.{tail}"] = v
        return out

    def _load_radimagenet_ckpt(self, path: str) -> None:
        print(f"[*] RadImageNetDenseNet1212DClassifier: loading weights from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            sd = ck["state_dict"]
        elif isinstance(ck, dict) and "model" in ck:
            sd = ck["model"]
        else:
            sd = ck
        remapped = self._remap_ms_to_torchvision(sd)
        target_n = len(self.densenet.state_dict())
        missing, unexpected = self.densenet.load_state_dict(remapped, strict=False)
        matched = target_n - len(missing)
        print(
            f"[*] RadImageNet DenseNet121: matched {matched}/{target_n} backbone keys "
            f"(source contributed {len(remapped)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        # Missing keys we'd expect: classifier.* (set to Identity above).
        non_clf_missing = [k for k in missing if not k.startswith("classifier.")]
        if non_clf_missing:
            print(f"[!] RIN-DenseNet121: unexplained missing keys: {non_clf_missing[:5]}")

    def _adapt_stem(self, in_channels: int) -> None:
        stem = self.densenet.features.conv0  # (64, 3, 7, 7)
        if stem.in_channels == in_channels:
            return
        new_conv = nn.Conv2d(
            in_channels, stem.out_channels,
            kernel_size=stem.kernel_size, stride=stem.stride,
            padding=stem.padding, bias=stem.bias is not None,
        )
        with torch.no_grad():
            avg = stem.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg.repeat(1, in_channels, 1, 1))
            if stem.bias is not None:
                new_conv.bias.copy_(stem.bias)
        self.densenet.features.conv0 = new_conv

    def forward(self, x, return_segmentation: bool = False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        feats = self.densenet(x)               # (B, 1024)
        cls_logits = self.classification_head(feats)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits


class MILResNet50Classifier(nn.Module):
    """BigLunge MIL classifier: attention pooling over whole-slice bags.

    Wraps ``monai.networks.nets.MILModel`` with a torchvision ResNet-50
    backbone (``mil_mode='att'`` by default). CT input is 1-channel; we
    adapt the pretrained 3-channel RGB stem by averaging across channels,
    the same grayscale-transfer trick the other 2D classifiers in this file
    use. Expected forward input: ``(B, N, 1, H, W)``.

    **Batch-norm leakage mitigation.** ``MILModel`` reshapes
    ``(B, N, C, H, W)`` to ``(B*N, C, H, W)`` before the backbone, so BN
    running statistics span every instance in every bag of the minibatch.
    That leaks bag-label information into per-instance features and inflates
    train metrics. We keep all BN modules in ``eval()`` mode during training
    (reusing the frozen running stats from ImageNet/DAPT pretraining) —
    attention and the final classifier still train normally. This is the
    standard fix for CNN-backbone MIL. Override with
    ``self._freeze_bn_for_training = False`` to disable for experiments.

    **Loading a DAPT backbone.** Call ``load_backbone_from_dapt(state_dict)``
    with a checkpoint produced by training ``TorchVisionResNet2DClassifier``.
    Keys like ``backbone.features.conv1.weight`` are re-routed to ``net.*``
    to initialize the MIL backbone; the DAPT classifier head is dropped
    (MIL has its own attention + ``myfc``).
    """

    def __init__(
        self,
        num_classes: int = 3,
        mil_mode: str = "att",
        pretrained_backbone: bool = True,
        trans_blocks: int = 4,
        trans_dropout: float = 0.0,
    ):
        super().__init__()
        self.mil = MILModel(
            num_classes=num_classes,
            mil_mode=mil_mode,
            pretrained=pretrained_backbone,
            backbone="resnet50",
            trans_blocks=trans_blocks,
            trans_dropout=trans_dropout,
        )
        self._adapt_stem_1ch()
        # BN was previously frozen (eval()) during MIL training to prevent
        # bag-label leakage through the (B*N)-wide batch statistics. In
        # practice on 1-ch CT, ImageNet running stats applied to grayscale
        # inputs produced activations far enough out-of-distribution to NaN
        # the attention softmax after one optimizer step. Letting BN update
        # its running stats during MIL fine-tune is the lesser evil here.
        self._freeze_bn_for_training = False

    def _adapt_stem_1ch(self) -> None:
        """Replace the 3-ch torchvision stem with a 1-ch conv initialized
        from the mean of the pretrained RGB weights.
        """
        stem = self.mil.net.conv1  # (64, 3, 7, 7) torchvision ResNet50 stem
        if stem.in_channels == 1:
            return
        new_conv = nn.Conv2d(
            1, stem.out_channels,
            kernel_size=stem.kernel_size, stride=stem.stride,
            padding=stem.padding, bias=stem.bias is not None,
        )
        with torch.no_grad():
            avg = stem.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            new_conv.weight.copy_(avg)
            if stem.bias is not None:
                new_conv.bias.copy_(stem.bias)
        self.mil.net.conv1 = new_conv

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_bn_for_training:
            for m in self.mil.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def load_backbone_from_dapt(self, state_dict, logger=None):
        """Load a DAPT ``TorchVisionResNet2DClassifier`` state_dict into
        ``self.mil.net``.

        DAPT keys look like ``backbone.features.<torchvision_key>``;
        ``self.mil.net`` keys look like ``<torchvision_key>``. We strip the
        ``backbone.features.`` prefix and load non-strict. Any head params
        (``backbone.fc.*``) are dropped — MIL has ``attention`` + ``myfc``
        which stay at their current init.
        """
        src_prefix = "backbone.features."
        target_sd = self.mil.net.state_dict()
        mapped: dict = {}
        for k, v in state_dict.items():
            if k.startswith(src_prefix):
                new_k = k[len(src_prefix):]
                if new_k in target_sd:
                    mapped[new_k] = v
        missing, unexpected = self.mil.net.load_state_dict(mapped, strict=False)
        matched = len(target_sd) - len(missing)
        if logger is not None:
            logger.info(
                f"[MIL] Loaded DAPT backbone: matched {matched}/{len(target_sd)} target keys "
                f"(source contributed {len(mapped)} entries, missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if 0 < len(missing) <= 10:
                logger.info(f"[MIL] Missing keys: {missing}")
        return matched, missing, unexpected

    def forward(self, x, return_segmentation: bool = False):
        if x.ndim != 5:
            raise ValueError(
                f"MILResNet50Classifier expects (B, N, C, H, W); got shape {tuple(x.shape)}"
            )
        cls_logits = self.mil(x)
        if return_segmentation:
            # No segmentation output for MIL; return a zero tensor for
            # API parity with the other wrappers in this file.
            seg = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)
            return cls_logits, seg
        return cls_logits

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-instance attention weights, shape ``(B, N)``.

        Only valid for ``mil_mode in {"att", "att_trans"}``. Useful for
        thesis figures and sanity checks (does attention land on
        tumor-bearing slices in BigLunge where we happen to have a tumor
        mask available for overlay?).
        """
        if self.mil.mil_mode not in ("att", "att_trans"):
            raise RuntimeError(
                f"attention_weights unsupported for mil_mode={self.mil.mil_mode}"
            )
        sh = x.shape
        flat = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        feats = self.mil.net(flat).reshape(sh[0], sh[1], -1)
        if self.mil.mil_mode == "att_trans" and self.mil.transformer is not None:
            feats = feats.permute(1, 0, 2)
            feats = self.mil.transformer(feats)
            feats = feats.permute(1, 0, 2)
        a = self.mil.attention(feats)
        a = torch.softmax(a, dim=1)  # (B, N, 1)
        return a.squeeze(-1)  # (B, N)






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
