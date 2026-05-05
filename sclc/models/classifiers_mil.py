import os
import timm
import torch
import torch.nn as nn
from monai.networks.nets import MILModel
from torchvision.models import resnet50 as _tv_resnet50
try:
    from torchvision.models import ResNet50_Weights
except Exception:  # pragma: no cover - older torchvision
    ResNet50_Weights = None

from .advanced_fpn import AdvancedFPNNeck
from .classifiers_2d import SwinTiny2DClassifier


class _MILAttentionPool(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        hidden = max(64, in_dim // 2)
        self.att = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.cls = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor):
        # feats: (B, N, C)
        att = self.att(feats)
        weights = torch.softmax(att, dim=1)
        pooled = (weights * feats).sum(dim=1)
        return self.cls(pooled), weights.squeeze(-1)


class _MILInstanceHead(nn.Module):
    def __init__(self, in_channels: int, use_seg: bool, use_det: bool):
        super().__init__()
        self.use_seg = bool(use_seg)
        self.use_det = bool(use_det)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.seg_head = nn.Conv2d(in_channels, 1, kernel_size=1) if self.use_seg else None
        self.box_head = nn.Linear(in_channels, 4) if self.use_det else None

    def forward(self, feat_map: torch.Tensor, input_shape: torch.Size):
        pooled = self.pool(feat_map).flatten(1)
        seg_logits = None
        box_pred = None
        if self.use_seg and self.seg_head is not None:
            seg_logits = self.seg_head(feat_map)
            if seg_logits.shape[2:] != tuple(input_shape[2:]):
                seg_logits = torch.nn.functional.interpolate(seg_logits, size=input_shape[2:], mode="nearest")
        if self.use_det and self.box_head is not None:
            box_pred = torch.sigmoid(self.box_head(pooled))
        return pooled, seg_logits, box_pred


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
        self._last_attention = None
        if not self._use_advanced_fpn:
            self.mil = MILModel(
                num_classes=num_classes,
                mil_mode=mil_mode,
                pretrained=pretrained_backbone,
                backbone="resnet50",
                trans_blocks=trans_blocks,
                trans_dropout=trans_dropout,
            )
            self._adapt_stem_1ch()
        else:
            if mil_mode not in ("att", "att_trans"):
                print(f"[MIL-ResNet] mil_mode={mil_mode} ignored in advanced FPN mode; using attention pooling.")
            weights = None
            if ResNet50_Weights is not None:
                weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            self.resnet = _tv_resnet50(weights=weights)
            self.resnet.fc = nn.Identity()
            self._adapt_resnet_stem_1ch()
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
            self.instance_head = _MILInstanceHead(
                in_channels=fpn_channels,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )
            self.att_pool = _MILAttentionPool(in_dim=fpn_channels, num_classes=num_classes)
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

    def _adapt_resnet_stem_1ch(self) -> None:
        stem = self.resnet.conv1
        if stem.in_channels == 1:
            return
        new_conv = nn.Conv2d(
            1, stem.out_channels,
            kernel_size=stem.kernel_size, stride=stem.stride,
            padding=stem.padding, bias=stem.bias is not None,
        )
        with torch.no_grad():
            avg = stem.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg)
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

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not self._use_advanced_fpn and self._freeze_bn_for_training:
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
        if not self._use_advanced_fpn:
            cls_logits = self.mil(x)
            if return_segmentation:
                seg = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)
                return cls_logits, seg
            return cls_logits

        B, N, C, H, W = x.shape
        flat = x.reshape(B * N, C, H, W)
        feats = self._forward_features(flat)
        fused, _ = self.fpn(feats)
        feat_vec, seg_logits, box_pred = self.instance_head(fused, flat.shape)
        feat_vec = feat_vec.view(B, N, -1)
        cls_logits, att = self.att_pool(feat_vec)
        self._last_attention = att
        if return_segmentation:
            if seg_logits is None:
                seg_logits = torch.zeros((B * N, 1, H, W), device=x.device)
            seg_logits = seg_logits.view(B, N, 1, H, W)
            if box_pred is None:
                box_pred = torch.zeros((B, N, 4), device=x.device)
            else:
                box_pred = box_pred.view(B, N, 4)
            return cls_logits, seg_logits, box_pred
        return cls_logits

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-instance attention weights, shape ``(B, N)``.

        Only valid for ``mil_mode in {"att", "att_trans"}``. Useful for
        thesis figures and sanity checks (does attention land on
        tumor-bearing slices in BigLunge where we happen to have a tumor
        mask available for overlay?).
        """
        if self._use_advanced_fpn:
            if self._last_attention is not None:
                return self._last_attention
            raise RuntimeError("attention_weights requires a forward pass in advanced FPN mode.")
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


class MILSwinTinyClassifier(nn.Module):
    """BigLunge MIL classifier: attention pooling over whole-slice bags,
    Swin-Tiny backbone (counterpart to ``MILResNet50Classifier``).

    Wraps ``monai.networks.nets.MILModel`` with a timm
    ``swin_tiny_patch4_window7_224`` backbone built as a feature extractor
    (``num_classes=0`` → final ``head`` is ``Identity``, output is the 768-d
    pooled feature vector). The backbone is initialized from RadImageNet
    (the same checkpoint used by ``SwinTiny2DClassifier``) and the
    3-channel patch_embed stem is averaged to a 1-channel conv.

    **No BN to freeze.** Unlike the ResNet-50 MIL variant, Swin-Tiny uses
    LayerNorm everywhere — there is no batch-statistic leakage across the
    flattened ``(B*N, C, H, W)`` MIL minibatch. We do not need (or run) the
    ``train()`` BN-eval override that ``MILResNet50Classifier`` provides.

    **AMP / softmax stability.** The same caveat as the ResNet-50 MIL
    variant applies: a randomly-initialized attention head on top of a
    strong pretrained backbone can NaN under AMP fp16 in early steps.
    Use a low FT learning rate, long warmup, and ``--disable-amp`` for the
    first run to rule out the fp16 path.

    **Loading a DAPT backbone.** Call ``load_backbone_from_dapt(state_dict)``
    with a checkpoint produced by training ``SwinTiny2DClassifier``. Keys
    of the form ``swin.<timm_key>`` are stripped to ``<timm_key>`` and
    loaded into ``self.mil.net``; the DAPT 3-class head ``swin.head.*`` is
    dropped (MIL has its own attention + ``myfc``).
    """

    DEFAULT_RADIMAGENET_CKPT = "/home/hansstem/RadImageNet_swin/rin_swintf.pth"
    BACKBONE_NUM_FEATURES = 768  # swin_tiny_patch4_window7_224

    def __init__(
        self,
        num_classes: int = 3,
        mil_mode: str = "att",
        radimagenet_ckpt: str = "",
        trans_blocks: int = 4,
        trans_dropout: float = 0.0,
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
        self._last_attention = None

        if not self._use_advanced_fpn:
            backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                num_classes=0,
                in_chans=1,
            )

            self.mil = MILModel(
                num_classes=num_classes,
                mil_mode=mil_mode,
                pretrained=False,
                backbone=backbone,
                backbone_num_features=self.BACKBONE_NUM_FEATURES,
                trans_blocks=trans_blocks,
                trans_dropout=trans_dropout,
            )

            rin_path = radimagenet_ckpt or self.DEFAULT_RADIMAGENET_CKPT
            if rin_path and os.path.exists(rin_path):
                self._load_radimagenet_into_backbone(rin_path)
            else:
                print(
                    f"[!] MILSwinTinyClassifier: RadImageNet checkpoint not found at "
                    f"{rin_path}; backbone is randomly initialized."
                )
        else:
            if mil_mode not in ("att", "att_trans"):
                print(f"[MIL-Swin] mil_mode={mil_mode} ignored in advanced FPN mode; using attention pooling.")
            self.backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=1,
            )
            rin_path = radimagenet_ckpt or self.DEFAULT_RADIMAGENET_CKPT
            if rin_path and os.path.exists(rin_path):
                self._load_rin_into_backbone(self.backbone, rin_path)
            else:
                print(
                    f"[!] MILSwinTinyClassifier: RadImageNet checkpoint not found at "
                    f"{rin_path}; backbone is randomly initialized."
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
            self.instance_head = _MILInstanceHead(
                in_channels=fpn_channels,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )
            self.att_pool = _MILAttentionPool(in_dim=fpn_channels, num_classes=num_classes)

    def _load_radimagenet_into_backbone(self, path: str) -> None:
        """Load RadImageNet weights into ``self.mil.net``.

        Reuses ``SwinTiny2DClassifier``'s MS->timm key remapping (downsample
        index +1) and the 3-ch->1-ch stem averaging. The RadImageNet 165-class
        head is dropped; the MIL head/attention stay at their fresh init.
        """
        print(f"[*] MILSwinTinyClassifier: loading RadImageNet weights from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            sd = ck["model"]
        elif isinstance(ck, dict) and "state_dict" in ck:
            sd = ck["state_dict"]
        else:
            sd = ck

        # Drop buffers timm recomputes at construction.
        sd = {
            k: v for k, v in sd.items()
            if not (k.endswith(".relative_position_index") or k.endswith(".attn_mask"))
        }
        # Drop the RadImageNet 165-class head; MIL's myfc replaces it.
        sd = {k: v for k, v in sd.items() if k not in ("head.weight", "head.bias")}
        # MS-style downsample keys -> timm-style (+1 stage).
        sd = SwinTiny2DClassifier._remap_ms_to_timm(sd)
        # Average 3-channel stem -> 1-channel.
        stem_key = "patch_embed.proj.weight"
        if stem_key in sd and sd[stem_key].shape[1] == 3:
            sd[stem_key] = sd[stem_key].mean(dim=1, keepdim=True)

        missing, unexpected = self.mil.net.load_state_dict(sd, strict=False)
        matched = len(sd) - len(unexpected)
        print(
            f"[*] RadImageNet weights into MIL backbone: matched {matched}/{len(sd)} keys "
            f"(unexpected={len(unexpected)}, missing={len(missing)})"
        )

    def _load_rin_into_backbone(self, backbone: nn.Module, path: str) -> None:
        print(f"[*] MILSwinTinyClassifier: loading RadImageNet weights from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            sd = ck["model"]
        elif isinstance(ck, dict) and "state_dict" in ck:
            sd = ck["state_dict"]
        else:
            sd = ck

        sd = {
            k: v for k, v in sd.items()
            if not (k.endswith(".relative_position_index") or k.endswith(".attn_mask"))
        }
        sd = {k: v for k, v in sd.items() if k not in ("head.weight", "head.bias")}
        sd = SwinTiny2DClassifier._remap_ms_to_timm(sd)
        stem_key = "patch_embed.proj.weight"
        if stem_key in sd and sd[stem_key].shape[1] == 3:
            sd[stem_key] = sd[stem_key].mean(dim=1, keepdim=True)

        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        matched = len(sd) - len(unexpected)
        print(
            f"[*] RadImageNet weights into MIL-Swin backbone: matched {matched}/{len(sd)} keys "
            f"(unexpected={len(unexpected)}, missing={len(missing)})"
        )

    def load_backbone_from_dapt(self, state_dict, logger=None):
        """Load a DAPT ``SwinTiny2DClassifier`` state_dict into ``self.mil.net``.

        DAPT keys look like ``swin.<timm_key>`` (the 2D wrapper stores the
        timm Swin under ``self.swin``). Strip the ``swin.`` prefix and load
        non-strict. The DAPT 3-class ``swin.head.*`` is dropped — MIL has
        ``attention`` + ``myfc`` which stay at their current init.
        """
        src_prefix = "swin."
        target_sd = self.mil.net.state_dict()
        mapped: dict = {}
        for k, v in state_dict.items():
            if not k.startswith(src_prefix):
                continue
            new_k = k[len(src_prefix):]
            if new_k in ("head.weight", "head.bias"):
                continue  # drop DAPT classification head
            if new_k in target_sd:
                mapped[new_k] = v
        missing, unexpected = self.mil.net.load_state_dict(mapped, strict=False)
        matched = len(target_sd) - len(missing)
        if logger is not None:
            logger.info(
                f"[MIL-Swin] Loaded DAPT backbone: matched {matched}/{len(target_sd)} target keys "
                f"(source contributed {len(mapped)} entries, missing={len(missing)}, "
                f"unexpected={len(unexpected)})"
            )
            if 0 < len(missing) <= 10:
                logger.info(f"[MIL-Swin] Missing keys: {missing}")
        return matched, missing, unexpected

    def forward(self, x, return_segmentation: bool = False):
        if x.ndim != 5:
            raise ValueError(
                f"MILSwinTinyClassifier expects (B, N, C, H, W); got shape {tuple(x.shape)}"
            )
        if not self._use_advanced_fpn:
            cls_logits = self.mil(x)
            if return_segmentation:
                seg = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)
                return cls_logits, seg
            return cls_logits

        B, N, C, H, W = x.shape
        flat = x.reshape(B * N, C, H, W)
        feats = self.backbone(flat)
        # timm Swin features_only returns NHWC (B,H,W,C); FPN expects NCHW
        feats_nchw = []
        for f, info in zip(feats, self.backbone.feature_info):
            expected_c = info["num_chs"]
            if f.ndim == 4 and f.shape[-1] == expected_c and f.shape[1] != expected_c:
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        fused, _ = self.fpn(feats_nchw)
        feat_vec, seg_logits, box_pred = self.instance_head(fused, flat.shape)
        feat_vec = feat_vec.view(B, N, -1)
        cls_logits, att = self.att_pool(feat_vec)
        self._last_attention = att
        if return_segmentation:
            if seg_logits is None:
                seg_logits = torch.zeros((B * N, 1, H, W), device=x.device)
            seg_logits = seg_logits.view(B, N, 1, H, W)
            if box_pred is None:
                box_pred = torch.zeros((B, N, 4), device=x.device)
            else:
                box_pred = box_pred.view(B, N, 4)
            return cls_logits, seg_logits, box_pred
        return cls_logits

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-instance attention weights, shape ``(B, N)``.

        Mirrors ``MILResNet50Classifier.attention_weights`` for downstream
        attention-overlay figures and entropy logging.
        """
        if self._use_advanced_fpn:
            if self._last_attention is not None:
                return self._last_attention
            raise RuntimeError("attention_weights requires a forward pass in advanced FPN mode.")
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
        a = torch.softmax(a, dim=1)
        return a.squeeze(-1)
