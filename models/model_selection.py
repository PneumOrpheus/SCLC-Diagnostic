import os
from typing import Dict, List, Optional, Tuple, Union
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
    
class GlobalClassificationHead(nn.Module):
    """
    Global Classification Head for image-level classification.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super(GlobalClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class DualHeadSCLCModel(nn.Module):
    """
    Composite model wrapping Faster R-CNN and Global Classifier.
    """
    def __init__(self, backbone_type: str, checkpoint_path: str = "", 
                 num_detection_classes: int = 2, num_global_classes: int = 2):
        super(DualHeadSCLCModel, self).__init__()
        
        # Map simple names to timm model names
        backbone_map = {
            "swin": "swin_base_patch4_window7_224",
            "swinv2": "swinv2_base_window12to24_192to384",
            "resnet": "resnet50",
            "densenet": "densenet121"
        }
        
        if backbone_type not in backbone_map:
            raise ValueError(f"Unsupported backbone type: {backbone_type}. Choose from {list(backbone_map.keys())}")
        
        model_name = backbone_map[backbone_type]
        fpn_out_channels = 256
        
        # Initialize Flexible Backbone
        self.backbone = FlexibleBackbone(model_name, checkpoint_path, out_channels=fpn_out_channels)
        
        # RPN Anchor Generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=[f"{i}" for i in range(len(self.backbone.in_channels_list))],
            output_size=7,
            sampling_ratio=2
        )
        
        # Initialize Detection Head
        self.detector = torchvision.models.detection.FasterRCNN(
            backbone=self.backbone,
            num_classes=num_detection_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        
        # Initialize Global Classification Head
        self.global_classifier = GlobalClassificationHead(
            in_channels=fpn_out_channels,
            num_classes=num_global_classes
        )
        
    def forward(self, scans: torch.Tensor, targets: Optional[List[Dict]] = None) -> Union[Dict[str, torch.Tensor], Tuple[List[Dict], torch.Tensor]]:
        # Internal transform for normalization
        original_scan_sizes = [scan.shape[-2:] for scan in scans]
        scans_transformed, targets_transformed = self.detector.transform(scans, targets)
        
        # Bacbone forward pass
        features = self.backbone(scans_transformed.tensors)
        
        # Detection head
        if self.training:
            proposals, proposals_losses = self.detector.rpn(
                scans_transformed, features, targets_transformed
            )
            detections, detector_losses = self.detector.roi_heads(
                features, proposals, scans_transformed.image_sizes, targets_transformed
            )
            
            losses = {}
            losses.update(proposals_losses)
            losses.update(detector_losses)
        else:
            proposals, _ = self.detector.rpn(
                scans_transformed, features, targets_transformed
            )
            detections, _ = self.detector.roi_heads(
                features, proposals, scans_transformed.image_sizes, targets_transformed
            )
            losses = {}
        
        # Global classification head
        global_features = features[f"{len(self.backbone.in_channels_list) - 1}"]
        global_logits = self.global_classifier(global_features)
        
        if self.training:
            gt_labels = torch.stack([t["scan_label"] for t in targets_transformed])
            global_loss = F.cross_entropy(global_logits, gt_labels)
            losses['global_classification_loss'] = global_loss
            return losses
        else:
            global_probabilites = F.softmax(global_logits, dim=1)
            return detections, global_probabilites
        
def get_sclc_model(backbone_type="swinv2", checkpoint_path="", 
                   num_detection_classes=2, num_global_classes=2) -> DualHeadSCLCModel:
    """
    Factory function to create DualHeadSCLCModel with specified backbone.
    """
    model = DualHeadSCLCModel(
        backbone_type=backbone_type,
        checkpoint_path=checkpoint_path,
        num_detection_classes=num_detection_classes,
        num_global_classes=num_global_classes
    )
    return model