import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict

from models.build import build_model
from models.utils import load_pretrained

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

class SwinFeatureExtractor(nn.Module):
    """
    Wrapper for custom SwinTransformer/SwinTransformerV2 models that extracts
    intermediate features from each stage for use with FPN.
    """
    patch_embed: nn.Module
    pos_drop: nn.Module
    layers: nn.ModuleList
    ape: bool
    absolute_pos_embed: torch.Tensor
    num_features: List[int]
    patches_resolution: Tuple[int, int]
    
    def __init__(self, swin_model: nn.Module):
        super(SwinFeatureExtractor, self).__init__()
        self.patch_embed = swin_model.patch_embed  # type: ignore[attr-defined]
        self.pos_drop = swin_model.pos_drop  # type: ignore[attr-defined]
        self.layers = swin_model.layers  # type: ignore[attr-defined]
        self.ape = swin_model.ape  # type: ignore[attr-defined]
        if self.ape:
            self.absolute_pos_embed = swin_model.absolute_pos_embed  # type: ignore[attr-defined]
        
        # Layer dim gives channel dimensions
        self.num_features = [layer.dim for layer in self.layers]
        self.patches_resolution = swin_model.patches_resolution  # type: ignore[attr-defined]
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        features = []
        for i, layer in enumerate(self.layers):
            # Process blocks and extract features before downsample
            for block in layer.blocks:
                x = block(x)
            
            B, L, C = x.shape
            # Use the BasicLayer's known spatial resolution (supports non-square feature maps)
            H, W = layer.input_resolution
            feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            features.append(feat)
            
            # Apply downsample after extracting features
            if layer.downsample is not None:
                x = layer.downsample(x)
        
        return features


class FlexibleBackbone(nn.Module):
    """
    Wraps timm models to be compatible with torchvision's detection models.
    Supports loading local checkpoints for fine-tuning.
    """
    def __init__(self, model_name: str, checkpoint_path: str = "", config: Optional[Any] = None,  out_channels: int = 256, logger: Optional[logging.Logger] = None):
        super(FlexibleBackbone, self).__init__()
        
        print(f"Initializing Backbone: {model_name}")
        
        self._use_custom_swin = False

        if config is not None and checkpoint_path != "":
            if logger is not None:
                logger.info(f"Creating model from config file:{config.MODEL.TYPE}/{config.MODEL.NAME}")
                logger.info(f"=> Path to pretrained weights: '{config.MODEL.PRETRAINED}'")
            swin_model = build_model(config)
            load_pretrained(config, swin_model, logger)
            
            # Wrap custom Swin model to extract multi-scale features
            self.body = SwinFeatureExtractor(swin_model)
            self._use_custom_swin = True
        else:
            import timm
            self.body = timm.create_model(model_name, pretrained=True, features_only=True)

        # Dynamic FPN Configuration, get channel counts from the backbone automatically
        if hasattr(self.body, "num_features"):
            # Custom SwinFeatureExtractor exposes num_features directly
            self.in_channels_list = list(self.body.num_features)  # type: ignore[arg-type]
        elif hasattr(self.body, "feature_info"):
            # timm models with features_only=True expose channel info via feature_info
            self.in_channels_list = list(self.body.feature_info.channels())
        else:
            raise AttributeError("Backbone model does not expose 'num_features' or 'feature_info.channels()'.")
        
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
            nn.LayerNorm(512),
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
    def __init__(self, backbone_type: str, checkpoint_path: str = "", config: Optional[Any] = None, 
                 num_detection_classes: int = 2, num_global_classes: int = 2, logger: Optional[logging.Logger] = None):
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
        self.backbone = FlexibleBackbone(model_name, checkpoint_path, config, out_channels=fpn_out_channels, logger=logger)
        
        # RPN Anchor Generator
        num_feature_levels = len(self.backbone.in_channels_list)
        base_anchor_sizes = (32, 64, 128, 256, 512)
        # Use one scale per feature level, slicing from the predefined base sizes
        anchor_sizes = tuple((base_anchor_sizes[i],) for i in range(min(num_feature_levels, len(base_anchor_sizes))))
        anchor_aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_aspect_ratios
        )
        
        # Determine the feature map names for ROI pooling based on the backbone's FPN outputs
        featmap_names = [str(i) for i in getattr(self.backbone, "out_indices", range(len(self.backbone.in_channels_list)))]
        
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=7,
            sampling_ratio=2
        )
        
        # TODO: Adjust min_size and max_size based on backbone requirements
        # Initialize Detection Head
        self.detector = torchvision.models.detection.FasterRCNN(
            backbone=self.backbone,
            num_classes=num_detection_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            # Set min/max size to match Swin input requirements
            min_size=224,
            max_size=224,
        )
        
        # Initialize Global Classification Head
        self.global_classifier = GlobalClassificationHead(
            in_channels=fpn_out_channels,
            num_classes=num_global_classes
        )
        
    def forward(self, scans: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Union[Dict[str, torch.Tensor], Tuple[List[Dict], torch.Tensor]]:
        # Internal transform for normalization

        scans_transformed, targets_transformed = self.detector.transform(scans, targets)
        
        # Backbone forward pass
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
                scans_transformed, features, None
            )
            detections, _ = self.detector.roi_heads(
                features, proposals, scans_transformed.image_sizes, None
            )
            losses = {}
        
        # Global classification head
        global_features = list(features.values())[-1]
        global_logits = self.global_classifier(global_features)
        
        if self.training:
            if targets_transformed is None:
                raise ValueError(
                    "DualHeadSCLCModel.forward expected 'targets' with 'scan_label' "
                    "for each sample during training, but got None."
                )
            missing_labels = [idx for idx, t in enumerate(targets_transformed) if "scan_label" not in t]
            if missing_labels:
                raise ValueError(
                    "DualHeadSCLCModel.forward expected each target to contain a "
                    "'scan_label' key during training, but it is missing for indices: "
                    f"{missing_labels}"
                )
            gt_labels = torch.stack([t["scan_label"] for t in targets_transformed])
            global_loss = F.cross_entropy(global_logits, gt_labels)
            losses['global_classification_loss'] = global_loss
            return losses
        else:
            global_probabilities = F.softmax(global_logits, dim=1)
            return detections, global_probabilities
        
def get_sclc_model(
    backbone_type: str = "swinv2",
    checkpoint_path: str = "",
    num_detection_classes: int = 2,
    num_global_classes: int = 2,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
) -> DualHeadSCLCModel:
    """
    Factory function to create DualHeadSCLCModel with specified backbone.
    
    Args:
        backbone_type: One of 'swin', 'swinv2', 'resnet', 'densenet'
        checkpoint_path: Path to pretrained weights (.pth file)
        num_detection_classes: Number of detection classes (including background)
        num_global_classes: Number of global classification classes
        config: Optional Microsoft-style yacs config object. If provided,
                will use config.MODEL.PRETRAINED for checkpoint_path and
                config.MODEL.NUM_CLASSES for num_global_classes.
                
    Returns:
        Initialized DualHeadSCLCModel
    """
    model = DualHeadSCLCModel(
        backbone_type=backbone_type,
        checkpoint_path=checkpoint_path,
        config=config,
        num_detection_classes=num_detection_classes,
        num_global_classes=num_global_classes,
        logger=logger
    )
    return model
