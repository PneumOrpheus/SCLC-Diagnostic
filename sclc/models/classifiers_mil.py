import timm
import torch
import torch.nn as nn
from monai.networks.nets import MILModel

from .advanced_fpn import AdvancedFPNNeck


class _MILAttentionPool(nn.Module):
    """Gated attention pooling for MIL bags (FPN path).

    Architecture: per-instance Linear→Tanh→Dropout→Linear attention scores,
    softmax over bag, weighted sum, Dropout on the pooled representation,
    then a linear classifier.  Dropout improves regularisation over the small
    BigLunge training set (≤342 bags/fold) and reduces FPN-branch variance.
    """
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(64, in_dim // 2)
        self.att = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )
        self.drop = nn.Dropout(p=dropout)
        self.cls = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor):
        att = self.att(feats)
        weights = torch.softmax(att, dim=1)
        pooled = (weights * feats).sum(dim=1)
        pooled = self.drop(pooled)
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


class MILSwinV2TinyClassifier(nn.Module):
    """MIL bag classifier with a SwinV2-Tiny backbone (swinv2_tiny_window8_256).

    Input must be 256x256 (window8 constraint).
    Expected forward input: ``(B, N, 1, H, W)``.
    """

    BACKBONE_NUM_FEATURES = 768

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
            swin = timm.create_model(
                "swinv2_tiny_window8_256.ms_in1k",
                pretrained=pretrained_backbone,
                num_classes=0,
                in_chans=1,
            )
            self.mil = MILModel(
                num_classes=num_classes,
                mil_mode=mil_mode,
                pretrained=False,
                backbone=swin,
                backbone_num_features=self.BACKBONE_NUM_FEATURES,
                trans_blocks=trans_blocks,
                trans_dropout=trans_dropout,
            )
        else:
            if mil_mode not in ("att", "att_trans"):
                print(f"[MIL-SwinV2Tiny] mil_mode={mil_mode} ignored in advanced FPN mode; using attention pooling.")
            self.backbone = timm.create_model(
                "swinv2_tiny_window8_256.ms_in1k",
                pretrained=pretrained_backbone,
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
            self.instance_head = _MILInstanceHead(
                in_channels=fpn_channels,
                use_seg=bool(use_det_seg),
                use_det=bool(use_det_seg),
            )
            self.att_pool = _MILAttentionPool(
                in_dim=fpn_channels, num_classes=num_classes, dropout=trans_dropout
            )

    def load_backbone_from_dapt(self, state_dict, logger=None):
        """Load a DAPT `SwinV2Tiny2DClassifier` state dict (keys `swin.<...>`)
        into the backbone, dropping the 3-class head."""
        src_prefix = "swin."
        target = self.backbone if self._use_advanced_fpn else self.mil.net
        target_sd = target.state_dict()

        mapped: dict = {}
        for k, v in state_dict.items():
            if not k.startswith(src_prefix):
                continue
            new_k = k[len(src_prefix):]
            if new_k in ("head.weight", "head.bias"):
                continue
            if new_k in target_sd:
                mapped[new_k] = v

        missing, unexpected = target.load_state_dict(mapped, strict=False)
        matched = len(target_sd) - len(missing)
        if logger is not None:
            logger.info(
                f"[MIL-SwinV2Tiny] Loaded DAPT backbone: matched {matched}/{len(target_sd)} target keys "
                f"(source contributed {len(mapped)} entries, missing={len(missing)}, "
                f"unexpected={len(unexpected)})"
            )
        return matched, missing, unexpected

    def load_backbone_from_checkpoint(self, state_dict, logger=None):
        """Transfer backbone weights from any supported checkpoint format.

        Tries prefixes `swin.` (2D DAPT), `backbone.` (FPN MIL), `mil.net.`
        (non-FPN MIL); the prefix with the most matching keys wins.
        """
        target = self.backbone if self._use_advanced_fpn else self.mil.net
        target_sd = target.state_dict()
        best_prefix, best_mapped = "", {}
        for prefix in ("swin.", "backbone.", "mil.net."):
            mapped: dict = {}
            for k, v in state_dict.items():
                if not k.startswith(prefix):
                    continue
                new_k = k[len(prefix):]
                if new_k in ("head.weight", "head.bias"):
                    continue
                if new_k in target_sd:
                    mapped[new_k] = v
            if len(mapped) > len(best_mapped):
                best_prefix, best_mapped = prefix, mapped
        if not best_mapped:
            if logger is not None:
                logger.warning("[MIL-SwinV2Tiny] load_backbone_from_checkpoint: no matching keys found.")
            return 0, list(target_sd.keys()), []
        missing, unexpected = target.load_state_dict(best_mapped, strict=False)
        matched = len(target_sd) - len(missing)
        if logger is not None:
            logger.info(
                f"[MIL-SwinV2Tiny] Transferred backbone (prefix='{best_prefix}'): "
                f"matched {matched}/{len(target_sd)} target keys "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        return matched, missing, unexpected

    def forward(self, x, return_segmentation: bool = False):
        if x.ndim != 5:
            raise ValueError(
                f"MILSwinV2TinyClassifier expects (B, N, C, H, W); got shape {tuple(x.shape)}"
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
        # timm SwinV2 features_only returns NHWC; FPN expects NCHW
        feats_nchw = []
        for f, info in zip(feats, self.backbone.feature_info):
            expected_c = info["num_chs"]
            if f.ndim == 4 and f.shape[-1] == expected_c and f.shape[1] != expected_c:
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        _, all_fused = self.fpn(feats_nchw)
        feat_vec, seg_logits, box_pred = self.instance_head(all_fused[-1], flat.shape)
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
