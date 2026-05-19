import os
import torch
import torch.nn as nn
import timm
from torchvision.models import resnet50 as _tv_resnet50, densenet121 as _tv_densenet121

from .advanced_fpn import AdvancedFPNNeck, MultiTaskHead


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
        if self._use_advanced_fpn:
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

    def _forward_features(self, x: torch.Tensor) -> list:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return [c2, c3, c4, c5]

    def forward(self, x, return_segmentation: bool = False, return_detection: bool = False):
        # Mirror the existing 2D wrappers' fallback for accidental 5D input.
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        if not self._use_advanced_fpn:
            feats = self.backbone(x)               # (B, 2048)
            cls_logits = self.classification_head(feats)
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
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        **_unused,
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

        if use_advanced_fpn or use_det_seg:
            print("[!] RadImageNetDenseNet1212DClassifier: advanced FPN not supported; using baseline head.")

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

    def forward(self, x, return_segmentation: bool = False, return_detection: bool = False):
        if x.ndim == 5:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        feats = self.densenet(x)               # (B, 1024)
        cls_logits = self.classification_head(feats)
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
