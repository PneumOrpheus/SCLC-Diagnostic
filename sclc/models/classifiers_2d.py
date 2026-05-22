import timm
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, EfficientNetBN, TorchVisionFCModel
from torchvision.models import resnet50 as _tv_resnet50

try:
    from torchvision.models import ResNet50_Weights
except Exception:  # pragma: no cover - older torchvision
    ResNet50_Weights = None

from .advanced_fpn import AdvancedFPNNeck, MultiTaskHead


class EfficientNet2DClassifier(nn.Module):
    """2D EfficientNet-B0 for 1-channel slice classification.

    Non-FPN mode: MONAI EfficientNetBN with the stem re-averaged from the
    pretrained RGB stem (MONAI random-inits when in_channels != 3).
    FPN mode: timm `efficientnet_b0` features_only (timm averages the stem
    automatically when `in_chans=1, pretrained=True`).
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
    """2D DenseNet-121 for 1-channel slice classification.

    DenseNet stage widths differ from ResNet — the FPN's LazyConv2d handles
    the variable widths without manual specification.
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
    """2D ResNet-50 with torchvision ImageNet weights.

    MONAI's TorchVisionFCModel `in_channels` kwarg does not actually swap the
    stem for torchvision backbones, so we replace `conv1` ourselves and
    average the pretrained RGB weights along the channel dim.
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
        # Recursive search instead of a hard-coded path: MONAI's NetAdapter
        # restructures the model, and ResNet18/34 may be swapped in later.
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


class SwinV2Tiny2DClassifier(nn.Module):
    """2D SwinV2-Tiny (swinv2_tiny_window8_256.ms_in1k), 1-channel CT slices.

    Input must be 256x256 (window=8 constraint). Stage channels [96,192,384,768].
    timm adapts the RGB stem to 1-channel automatically via `in_chans=1`.
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

