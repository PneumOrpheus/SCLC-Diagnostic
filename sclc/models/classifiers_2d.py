import os
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, EfficientNetBN, TorchVisionFCModel
import timm
from torchvision.models import resnet50 as _tv_resnet50
try:
    from torchvision.models import ResNet50_Weights
except Exception:  # pragma: no cover - older torchvision
    ResNet50_Weights = None

from .advanced_fpn import AdvancedFPNNeck, MultiTaskHead


class EfficientNet2DClassifier(nn.Module):
    """2D EfficientNet-B0 (ImageNet-pretrained) for single-slice classification.

    Non-FPN mode uses MONAI's EfficientNetBN.  MONAI re-initialises the stem
    conv randomly for in_channels != 3, so we copy the channel-mean of the
    pretrained RGB stem into the 1-channel stem (grayscale-transfer trick).

    FPN mode switches to timm's EfficientNet-B0 (``efficientnet_b0``) with
    ``features_only=True`` so we can extract a four-level pyramid at strides
    4/8/16/32.  timm automatically averages the pretrained RGB patch_embed to
    1-channel when ``in_chans=1`` is passed alongside ``pretrained=True``,
    matching what the other FPN classifiers do.  Features are already NCHW —
    no permute needed.
    """

    def __init__(
        self,
        num_classes: int = 3,
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        fpn_channels: int = 256,
        tfpn_enabled: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: tuple = (14, 14),
        tfpn_levels: int = 1,
        **_unused,
    ):
        super().__init__()
        self._use_advanced_fpn = bool(use_advanced_fpn or use_det_seg)
        self._use_det_seg = bool(use_det_seg)

        if not self._use_advanced_fpn:
            self.efficientnet = EfficientNetBN(
                "efficientnet-b0",
                pretrained=True,
                spatial_dims=2,
                in_channels=1,
                num_classes=num_classes,
            )
            self._init_1ch_stem_from_rgb()
        else:
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=1,
            )
            self.fpn = AdvancedFPNNeck(
                num_levels=4,
                out_channels=fpn_channels,
                spatial_dims=2,
                use_tfpn=bool(tfpn_enabled),
                tfpn_heads=tfpn_heads,
                tfpn_layers=tfpn_layers,
                tfpn_pool=tfpn_pool,
                tfpn_levels=tfpn_levels,
            )
            self.head = MultiTaskHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                spatial_dims=2,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )

    def _init_1ch_stem_from_rgb(self) -> None:
        ref = EfficientNetBN(
            "efficientnet-b0", pretrained=True, spatial_dims=2, in_channels=3,
        )
        with torch.no_grad():
            rgb = ref._conv_stem.weight  # (32, 3, 3, 3)
            avg = rgb.mean(dim=1, keepdim=True)  # (32, 1, 3, 3)
            self.efficientnet._conv_stem.weight.copy_(avg)
        del ref

    def forward(self, x, return_segmentation=False, return_detection: bool = False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)

        if not self._use_advanced_fpn:
            cls_logits = self.efficientnet(x)
            if return_segmentation:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
                if return_detection:
                    box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                    return cls_logits, seg_logits, box_pred
                return cls_logits, seg_logits
            if return_detection:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                return cls_logits, None, box_pred
            return cls_logits

        feats = self.backbone(x)  # list of 4 NCHW tensors (strides 4,8,16,32)
        fused, _ = self.fpn(feats)
        cls_logits, seg_logits, box_pred = self.head(fused, x.shape)
        if return_segmentation or return_detection:
            if return_segmentation and seg_logits is None:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            if return_detection and box_pred is None:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
            if return_detection:
                return cls_logits, seg_logits, box_pred
            return cls_logits, seg_logits
        return cls_logits


class DenseNet2DClassifier(nn.Module):
    """2D DenseNet-121 (ImageNet-pretrained) for single-slice classification.

    Non-FPN mode uses MONAI's DenseNet121 (handles 1-channel input natively).

    FPN mode switches to timm's ``densenet121`` with ``features_only=True``
    to extract a four-level pyramid.  timm averages the pretrained RGB stem
    to 1-channel when ``in_chans=1, pretrained=True``.  DenseNet features are
    already NCHW.  Note that DenseNet's dense connections make each stage
    output wider than typical ResNet stages; LazyConv2d in AdvancedFPNNeck
    handles the variable widths without manual specification.
    """

    def __init__(
        self,
        num_classes: int = 3,
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        fpn_channels: int = 256,
        tfpn_enabled: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: tuple = (14, 14),
        tfpn_levels: int = 1,
        **_unused,
    ):
        super().__init__()
        self._use_advanced_fpn = bool(use_advanced_fpn or use_det_seg)
        self._use_det_seg = bool(use_det_seg)

        if not self._use_advanced_fpn:
            self.densenet = DenseNet121(
                spatial_dims=2,
                in_channels=1,
                out_channels=num_classes,
                pretrained=True,
            )
        else:
            self.backbone = timm.create_model(
                "densenet121",
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=1,
            )
            self.fpn = AdvancedFPNNeck(
                num_levels=4,
                out_channels=fpn_channels,
                spatial_dims=2,
                use_tfpn=bool(tfpn_enabled),
                tfpn_heads=tfpn_heads,
                tfpn_layers=tfpn_layers,
                tfpn_pool=tfpn_pool,
                tfpn_levels=tfpn_levels,
            )
            self.head = MultiTaskHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                spatial_dims=2,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )

    def forward(self, x, return_segmentation=False, return_detection: bool = False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)

        if not self._use_advanced_fpn:
            cls_logits = self.densenet(x)
            if return_segmentation:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
                if return_detection:
                    box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                    return cls_logits, seg_logits, box_pred
                return cls_logits, seg_logits
            if return_detection:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                return cls_logits, None, box_pred
            return cls_logits

        feats = self.backbone(x)  # list of 4 NCHW tensors (strides 4,8,16,32)
        fused, _ = self.fpn(feats)
        cls_logits, seg_logits, box_pred = self.head(fused, x.shape)
        if return_segmentation or return_detection:
            if return_segmentation and seg_logits is None:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            if return_detection and box_pred is None:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
            if return_detection:
                return cls_logits, seg_logits, box_pred
            return cls_logits, seg_logits
        return cls_logits


class TorchVisionResNet2DClassifier(nn.Module):
    """2D ResNet-50 wired via MONAI's TorchVisionFCModel with torchvision
    ImageNet weights. MONAI's ``in_channels`` kwarg doesn't actually swap the
    stem for torchvision backbones in this version, so we replace ``conv1``
    ourselves and average the RGB pretrained weights across the 3 channel dim
    (standard grayscale-transfer trick). FC head is replaced for ``num_classes``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        model_name: str = "resnet50",
        in_channels: int = 1,
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        fpn_channels: int = 256,
        tfpn_enabled: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: tuple = (14, 14),
        tfpn_levels: int = 1,
    ):
        super().__init__()
        self._use_advanced_fpn = bool(use_advanced_fpn or use_det_seg)
        self._use_det_seg = bool(use_det_seg)
        self._in_channels = int(in_channels)
        self._num_classes = int(num_classes)

        if not self._use_advanced_fpn:
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
        else:
            weights = None
            if ResNet50_Weights is not None:
                weights = ResNet50_Weights.IMAGENET1K_V2
            self.resnet = _tv_resnet50(weights=weights)
            self.resnet.fc = nn.Identity()
            if in_channels != 3:
                self._adapt_resnet_stem(in_channels)
            self.fpn = AdvancedFPNNeck(
                num_levels=4,
                out_channels=fpn_channels,
                spatial_dims=2,
                use_tfpn=bool(tfpn_enabled),
                tfpn_heads=tfpn_heads,
                tfpn_layers=tfpn_layers,
                tfpn_pool=tfpn_pool,
                tfpn_levels=tfpn_levels,
            )
            self.head = MultiTaskHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                spatial_dims=2,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )

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

    def _adapt_resnet_stem(self, in_channels: int) -> None:
        stem = self.resnet.conv1
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
        self.resnet.conv1 = new_conv

    def _forward_features(self, x: torch.Tensor) -> list:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        c2 = self.resnet.layer1(x)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)
        return [c2, c3, c4, c5]

    def forward(self, x, return_segmentation=False, return_detection: bool = False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        if not self._use_advanced_fpn:
            cls_logits = self.backbone(x)
            if return_segmentation:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
                if return_detection:
                    box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                    return cls_logits, seg_logits, box_pred
                return cls_logits, seg_logits
            if return_detection:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                return cls_logits, None, box_pred
            return cls_logits

        feats = self._forward_features(x)
        fused, _ = self.fpn(feats)
        cls_logits, seg_logits, box_pred = self.head(fused, x.shape)
        if return_segmentation or return_detection:
            if return_segmentation and seg_logits is None:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            if return_detection and box_pred is None:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
            if return_detection:
                return cls_logits, seg_logits, box_pred
            return cls_logits, seg_logits
        return cls_logits




class SwinV2Base2DClassifier(nn.Module):
    """2D SwinV2-Base (swinv2_base_window8_256.ms_in1k) backbone,
    ImageNet-pretrained via timm, adapted to 1-channel CT slice classification.

    timm automatically averages the 3-channel pretrained patch_embed stem to
    1 channel when ``in_chans=1`` is passed alongside ``pretrained=True``,
    using the same channel-mean trick applied in the other 2D classifiers here.
    Input resolution must be 256 × 256 (window8 constraint).
    """

    def __init__(
        self,
        num_classes: int = 3,
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        fpn_channels: int = 256,
        tfpn_enabled: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: tuple = (14, 14),
        tfpn_levels: int = 1,
    ):
        super().__init__()
        self._use_advanced_fpn = bool(use_advanced_fpn or use_det_seg)
        self._use_det_seg = bool(use_det_seg)

        if not self._use_advanced_fpn:
            # timm adapts the 3-channel stem to 1 channel automatically when
            # in_chans differs from the pretrained model's in_chans.
            self.swin = timm.create_model(
                "swinv2_base_window8_256.ms_in1k",
                pretrained=True,
                num_classes=num_classes,
                in_chans=1,
            )
        else:
            self.swin = timm.create_model(
                "swinv2_base_window8_256.ms_in1k",
                pretrained=True,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                in_chans=1,
            )
            self.fpn = AdvancedFPNNeck(
                num_levels=4,
                out_channels=fpn_channels,
                spatial_dims=2,
                use_tfpn=bool(tfpn_enabled),
                tfpn_heads=tfpn_heads,
                tfpn_layers=tfpn_layers,
                tfpn_pool=tfpn_pool,
                tfpn_levels=tfpn_levels,
            )
            self.head = MultiTaskHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                spatial_dims=2,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )

    def forward(self, x, return_segmentation=False, return_detection: bool = False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        if not self._use_advanced_fpn:
            cls_logits = self.swin(x)
            if return_segmentation:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
                if return_detection:
                    box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                    return cls_logits, seg_logits, box_pred
                return cls_logits, seg_logits
            if return_detection:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                return cls_logits, None, box_pred
            return cls_logits

        feats = self.swin(x)
        # timm SwinV2 features_only returns NHWC; FPN expects NCHW
        feats_nchw = []
        for f, info in zip(feats, self.swin.feature_info):
            expected_c = info["num_chs"]
            if f.ndim == 4 and f.shape[-1] == expected_c and f.shape[1] != expected_c:
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        fused, _ = self.fpn(feats_nchw)
        cls_logits, seg_logits, box_pred = self.head(fused, x.shape)
        if return_segmentation or return_detection:
            if return_segmentation and seg_logits is None:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            if return_detection and box_pred is None:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
            if return_detection:
                return cls_logits, seg_logits, box_pred
            return cls_logits, seg_logits
        return cls_logits


class SwinV2Tiny2DClassifier(nn.Module):
    """2D SwinV2-Tiny (swinv2_tiny_window8_256.ms_in1k) backbone,
    ImageNet-pretrained via timm, 1-channel CT slice classification.

    Smaller counterpart to SwinV2Base2DClassifier (~28M vs ~88M params).
    Stage channels: [96, 192, 384, 768]. BACKBONE_NUM_FEATURES = 768.
    Input must be 256×256 (window8 constraint). No RadImageNet required.
    timm adapts the 3-channel stem to 1-channel automatically via in_chans=1.
    """

    def __init__(
        self,
        num_classes: int = 3,
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        fpn_channels: int = 256,
        tfpn_enabled: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: tuple = (14, 14),
        tfpn_levels: int = 1,
    ):
        super().__init__()
        self._use_advanced_fpn = bool(use_advanced_fpn or use_det_seg)
        self._use_det_seg = bool(use_det_seg)

        if not self._use_advanced_fpn:
            self.swin = timm.create_model(
                "swinv2_tiny_window8_256.ms_in1k",
                pretrained=True, num_classes=num_classes, in_chans=1,
            )
        else:
            self.swin = timm.create_model(
                "swinv2_tiny_window8_256.ms_in1k",
                pretrained=True, features_only=True, out_indices=(0, 1, 2, 3), in_chans=1,
            )
            self.fpn = AdvancedFPNNeck(
                num_levels=4, out_channels=fpn_channels, spatial_dims=2,
                use_tfpn=bool(tfpn_enabled), tfpn_heads=tfpn_heads,
                tfpn_layers=tfpn_layers, tfpn_pool=tfpn_pool, tfpn_levels=tfpn_levels,
            )
            self.head = MultiTaskHead(
                in_channels=fpn_channels, num_classes=num_classes, spatial_dims=2,
                use_seg=bool(use_det_seg), use_det=bool(use_det_seg),
            )

    def forward(self, x, return_segmentation=False, return_detection: bool = False):
        if x.ndim == 5:
            x = x.squeeze(-1) if x.shape[-1] == 1 else x.squeeze(2)
        if not self._use_advanced_fpn:
            cls_logits = self.swin(x)
            if return_segmentation:
                seg = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
                if return_detection:
                    return cls_logits, seg, torch.zeros((x.shape[0], 4), device=x.device)
                return cls_logits, seg
            if return_detection:
                return cls_logits, None, torch.zeros((x.shape[0], 4), device=x.device)
            return cls_logits

        feats = self.swin(x)
        # timm SwinV2 features_only returns NHWC; FPN expects NCHW
        feats_nchw = []
        for f, info in zip(feats, self.swin.feature_info):
            expected_c = info["num_chs"]
            if f.ndim == 4 and f.shape[-1] == expected_c and f.shape[1] != expected_c:
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        fused, _ = self.fpn(feats_nchw)
        cls_logits, seg_logits, box_pred = self.head(fused, x.shape)
        if return_segmentation or return_detection:
            if return_segmentation and seg_logits is None:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            if return_detection and box_pred is None:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
            if return_detection:
                return cls_logits, seg_logits, box_pred
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

    def __init__(
        self,
        num_classes: int = 3,
        radimagenet_ckpt: str = "",
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        fpn_channels: int = 256,
        tfpn_enabled: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: tuple = (14, 14),
        tfpn_levels: int = 1,
    ):
        super().__init__()
        self._use_advanced_fpn = bool(use_advanced_fpn or use_det_seg)
        self._use_det_seg = bool(use_det_seg)

        if not self._use_advanced_fpn:
            # num_classes=3 at construction so timm builds a fresh 3-class head
            self.swin = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                num_classes=num_classes,
                in_chans=1,
            )
            if radimagenet_ckpt and os.path.exists(radimagenet_ckpt):
                self._load_radimagenet_ckpt(radimagenet_ckpt)
        else:
            self.swin = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                features_only=True,
                out_indices=(0, 1, 2, 3),  # Swin-Tiny has 4 stages: valid indices are 0-3
                in_chans=1,
            )
            if radimagenet_ckpt and os.path.exists(radimagenet_ckpt):
                self._load_radimagenet_ckpt(radimagenet_ckpt)
            self.fpn = AdvancedFPNNeck(
                num_levels=4,
                out_channels=fpn_channels,
                spatial_dims=2,
                use_tfpn=bool(tfpn_enabled),
                tfpn_heads=tfpn_heads,
                tfpn_layers=tfpn_layers,
                tfpn_pool=tfpn_pool,
                tfpn_levels=tfpn_levels,
            )
            self.head = MultiTaskHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                spatial_dims=2,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )

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

    def forward(self, x, return_segmentation=False, return_detection: bool = False):
        # Pipeline outputs (B, 1, H, W). Fallback for accidental 5D inputs.
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        if not self._use_advanced_fpn:
            cls_logits = self.swin(x)
            if return_segmentation:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
                if return_detection:
                    box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                    return cls_logits, seg_logits, box_pred
                return cls_logits, seg_logits
            if return_detection:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
                return cls_logits, None, box_pred
            return cls_logits

        feats = self.swin(x)
        # timm Swin features_only returns NHWC (B,H,W,C); FPN expects NCHW
        feats_nchw = []
        for f, info in zip(feats, self.swin.feature_info):
            expected_c = info["num_chs"]
            if f.ndim == 4 and f.shape[-1] == expected_c and f.shape[1] != expected_c:
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        fused, _ = self.fpn(feats_nchw)
        cls_logits, seg_logits, box_pred = self.head(fused, x.shape)
        if return_segmentation or return_detection:
            if return_segmentation and seg_logits is None:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            if return_detection and box_pred is None:
                box_pred = torch.zeros((x.shape[0], 4), device=x.device)
            if return_detection:
                return cls_logits, seg_logits, box_pred
            return cls_logits, seg_logits
        return cls_logits
