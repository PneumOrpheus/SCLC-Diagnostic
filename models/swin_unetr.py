from monai.networks.nets import SwinUNETR
from torch import nn
import torch
import torch.nn.functional as F
from typing import List, Optional, Any
import os

class SwinUNETRFeatureExtractor(nn.Module):
    """
    Wrapper for MONAI SwinUNETR that extracts multi-scale encoder features
    and collapses the depth dimension to produce 2D feature maps for FPN.
    """
    num_features: List[int]

    def __init__(self, swin_unetr_model: nn.Module):
        super().__init__()
        self.encoder = swin_unetr_model.swinViT  # Trains a new SwinTransformer encoder from scratch
        self.normalize = getattr(swin_unetr_model, 'normalize', True)

        embed_dim = int(self.encoder.embed_dim)
        num_layers = int(self.encoder.num_layers)  # 4 (not .layers which doesn't exist)
        # SwinTransformer.forward() returns [x0_out, x1_out, x2_out, x3_out, x4_out]
        # We skip x0_out (patch embed, embed_dim channels) and use the 4 stage outputs.
        # Each stage includes PatchMerging downsample, so output channels double:
        #   x1_out: embed_dim*2, x2_out: embed_dim*4, x3_out: embed_dim*8, x4_out: embed_dim*16
        self.num_features = [embed_dim * (2 ** (i + 1)) for i in range(num_layers)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features, collapsing depth to 2D for FPN.

        Args:
            x: (B, C, D, H, W) 5D volume tensor.

        Returns:
            List of (B, C_i, H_i, W_i) 2D feature maps per stage.
        """
        hidden_states = self.encoder(x, self.normalize)
        # hidden_states: [x0_out(patch_embed), x1_out, x2_out, x3_out, x4_out]
        # Skip patch embed, use stage outputs which are already (B, C, D, H, W)
        features = []
        for hs in hidden_states[1:]:
            if hs.ndim == 5:
                feat_2d = F.adaptive_max_pool3d(hs, (1, hs.shape[3], hs.shape[4])).squeeze(2)
            elif hs.ndim == 4:
                feat_2d = hs
            else:
                raise ValueError(f"Unexpected hidden state ndim={hs.ndim}")
            features.append(feat_2d)
        return features

def get_swin_unetr_model(checkpoint_path: str) -> nn.Module:
    model = SwinUNETR(in_channels=1,
                    patch_size=2,
                    depths=[2, 2, 4, 2],
                    out_channels=14,
                    feature_size=48,
                    spatial_dims=3,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    dropout_path_rate=0.0,
                    use_checkpoint=True)
    
    # Load the pretrained weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading pretrained SwinUNETR weights from {checkpoint_path}")
    else: 
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # MONAI checkpoints sometimes wrap the weights in a 'state_dict' key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
            
        # load_state_dict with strict=False is required!
    model.load_state_dict(state_dict, strict=False)
    print("SwinUNETR weights loaded successfully.")
    return SwinUNETRFeatureExtractor(model)

