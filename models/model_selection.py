import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict

"""
SCLC Diagnostic System - Model Architecture Module
==================================================
Implements a Dual-Head architecture with flexible backbone selection.
Supports loading local.pth checkpoints for transfer learning.

Key Components:
- Backbone Factory: Selects SwinV2, ResNet, or DenseNet.
- FPN: dynamically adapts to backbone output channels.
- Dual-Head: Simultaneous Detection and Global Classification.
"""

class FlexibleBackbone(nn.Module):
    """
    Wraps timm models to be compatible with torchvision's detection models.
    Supports loading local checkpoints for fine-tuning.
    """
    def __init__(self, model_name: str, checkpoint_path: str = "", out_channels: int = 256):
        super(FlexibleBackbone, self).__init__()
        
        print(f"Initializing Backbone: {model_name}")
        
        # Create backbone using timm, features_only=True extracts the feature maps
        self.body = timm.create_model(
            model_name,
            pretrained=(checkpoint_path == ""),
            features_only=True,
            out_indices=(0, 1, 2, 3) # Select features from 4 stages
        )
        
        # Load local checkpoint
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print(f"Loading local weights from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle potential key mismatches
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "").replace("backbone.body.", "")
                    new_state_dict[name] = v
                
                # Use strict=False to ignore heads if they exist in checkpoint but not in features_only model
                msg = self.body.load_state_dict(new_state_dict, strict=False)
                print(f"Weights loaded. Missing keys (expected for headless): {len(msg.missing_keys)}")
            else:
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Dynamic FPN Configuration, get channel counts from the backbone automatically
        feature_info = self.body.feature_info
        self.in_channels_list = list(feature_info.channels())  # type: ignore[union-attr]
        
        # Create FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=out_channels
        )
        
        # Required attribute for torchvision detection models
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict:
        # Get features from backbone
        xs = self.body(x)
        
        # Prepare dictionary for FPN
        x_dict = OrderedDict()
        for i, feature in enumerate(xs):
            # Ensure NCHW format
            if feature.ndim == 4 and feature.shape[1] != self.in_channels_list[i]:
                 feature = feature.permute(0, 3, 1, 2)
            x_dict[f"{i}"] = feature
            
        # Pass through FPN
        fpn_out = self.fpn(x_dict)
        return fpn_out
