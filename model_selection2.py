import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import os

class SwinUNETRClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # We initialize the base SwinUNETR. 
        # (out_channels is required but ignored since we bypass the decoder for now)
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=1, 
            feature_size=48,
            spatial_dims=3,
            use_checkpoint=True, 
            use_v2=True
        )
        
        # In MONAI's SwinUNETR, the deepest feature channel size is feature_size * 16 (i.e., 768)
        bottleneck_channels = 48 * 16  
        
        self.global_pool = nn.AdaptiveMaxPool3d(1)
        self.classification_head = nn.Sequential(
            nn.Linear(bottleneck_channels, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Register a hook to capture the deepest features during full forward pass
        self.deepest_features = None
        def vit_hook(module, input, output):
            self.deepest_features = output[-1]
        self.swin_unetr.swinViT.register_forward_hook(vit_hook)

    def forward(self, x, return_segmentation=False):
        # populates self.deepest_features with the final encoder bottleneck before the decoder.
        seg_logits = self.swin_unetr(x)
        
        # Pool spatially to [B, C, 1, 1, 1] then flatten to [B, C]
        pooled = self.global_pool(self.deepest_features).flatten(1)
        
        # Get class logits [B, num_classes]
        cls_logits = self.classification_head(pooled)
        
        if return_segmentation:
            return cls_logits, seg_logits
        else:
            return cls_logits


def get_sclc_model(checkpoint_path: str = "") -> nn.Module:
    model = SwinUNETRClassifier(in_channels=1, num_classes=3)
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
            
            model.swin_unetr.load_state_dict(state_dict, strict=False)
            print("[*] Pretrained weights loaded successfully.")
        else:
            print(f"[!] Warning: Checkpoint path {checkpoint_path} does not exist. Initializing from scratch.")
    return model
