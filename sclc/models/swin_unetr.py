import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


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
