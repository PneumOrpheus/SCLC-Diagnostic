import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from .advanced_fpn import AdvancedFPNNeck, MultiTaskHead


class SwinUNETRClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        use_advanced_fpn: bool = False,
        use_det_seg: bool = False,
        fpn_channels: int = 256,
        tfpn_enabled: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: tuple = (8, 8, 8),
        tfpn_levels: int = 1,
    ):
        super().__init__()
        # We initialize the base SwinUNETR
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
        # Single linear head
        self.classification_head = nn.Linear(bottleneck_channels, num_classes)
        self._use_advanced_fpn = bool(use_advanced_fpn)
        self._use_det_seg = bool(use_det_seg)
        if self._use_advanced_fpn:
            self.fpn = AdvancedFPNNeck(
                num_levels=4,
                out_channels=fpn_channels,
                spatial_dims=3,
                use_tfpn=bool(tfpn_enabled),
                tfpn_heads=tfpn_heads,
                tfpn_layers=tfpn_layers,
                tfpn_pool=tfpn_pool,
                tfpn_levels=tfpn_levels,
            )
            self.fpn_head = MultiTaskHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                spatial_dims=3,
                use_seg=False,
                use_det=bool(use_det_seg),
            )
            self.box_head = None
        else:
            self.fpn = None
            self.fpn_head = None
            self.box_head = nn.Linear(bottleneck_channels, 6) if use_det_seg else None
        
        # Hook captures the deepest swinViT feature map during the full forward pass
        self.deepest_features = None
        self._capture_deepest = False
        def vit_hook(module, input, output):
            if self._capture_deepest:
                self.deepest_features = output[-1]
        self.swin_unetr.swinViT.register_forward_hook(vit_hook)

    def forward(self, x, return_segmentation=False, return_detection: bool = False):
        self.deepest_features = None
        seg_logits = None
        if return_segmentation:
            # populates self.deepest_features with the final encoder bottleneck before the decoder.
            self._capture_deepest = True
            try:
                seg_logits = self.swin_unetr(x)
            finally:
                self._capture_deepest = False

        # Always compute encoder features for classification/detection.
        hidden_states = self.swin_unetr.swinViT(x.contiguous())
        if self._use_advanced_fpn:
            feats = hidden_states[-4:] if isinstance(hidden_states, (list, tuple)) and len(hidden_states) >= 4 else [hidden_states[-1]]
            fused, _ = self.fpn(list(feats))
            cls_logits, _seg_unused, box_pred = self.fpn_head(fused, x.shape)
        else:
            bottleneck = hidden_states[-1]
            pooled = self.global_pool(bottleneck).flatten(1)
            cls_logits = self.classification_head(pooled)
            box_pred = torch.sigmoid(self.box_head(pooled)) if self.box_head is not None else None

        if return_segmentation or return_detection:
            if return_segmentation and seg_logits is None:
                seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            if return_detection:
                if box_pred is None:
                    box_pred = torch.zeros((x.shape[0], 6), device=x.device)
                return cls_logits, seg_logits, box_pred
            return cls_logits, seg_logits
        return cls_logits



# Pipeline selection by model-type suffix.
#   _2d  -> single axial slice containing tumor, 2D CNN (in_channels=1)
#   mil_ -> attention-MIL bag of axial slices
#   else -> full 3D volume
