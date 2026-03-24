import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR, resnet50
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


class ResNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Initialize MONAI's 3D ResNet50. Setting feed_forward=False allows loading MedicalNet weights
        self.resnet = resnet50(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=in_channels,
            feed_forward=False,
            shortcut_type="B",
            bias_downsample=False
        )
        
        # Adding a custom classification head
        self.classification_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_segmentation=False):
        features = self.resnet(x)
        
        # Squeeze in case MONAI returns spatial features [B, C, D, H, W]
        if features.dim() > 2:
            features = nn.functional.adaptive_avg_pool3d(features, 1).flatten(1)
            
        cls_logits = self.classification_head(features)
        
        if return_segmentation:
            # ResNet doesn't naturally output a segmentation mask, so we return a dummy zero mask
            # This enables pipeline compatibility without breaking the loss functions
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        else:
            return cls_logits


def get_sclc_model(checkpoint_path: str = "", model_type: str = "swin_unetr") -> nn.Module:
    if model_type.lower() == "resnet50":
        model = ResNetClassifier(in_channels=1, num_classes=3)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading pretrained ResNet weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.resnet.load_state_dict(state_dict, strict=False)
            print("[*] Pretrained weights loaded successfully.")
    else:
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
