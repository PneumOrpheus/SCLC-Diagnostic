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
        self.num_features = [int(getattr(layer, "dim")) for layer in self.layers]
        self.patches_resolution = swin_model.patches_resolution  # type: ignore[attr-defined]
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        features = []
        for i, layer in enumerate(self.layers):
            # Process blocks and extract features before downsample
            blocks: nn.ModuleList = layer.blocks  # type: ignore[assignment]
            for block in blocks:
                x = block(x)
            
            B, L, C = x.shape
            # Use BasicLayer's known spatial resolution
            H, W = layer.input_resolution  # type: ignore[union-attr]
            feat = x.view(B, int(H), int(W), int(C)).permute(0, 3, 1, 2).contiguous()
            features.append(feat)
            
            # Apply downsample after extracting features
            downsample: Optional[nn.Module] = layer.downsample  # type: ignore[assignment]
            if downsample is not None:
                x = downsample(x)
        
        return features


class SwinFeatureExtractor3D(nn.Module):
    """
    Wrapper for 3D SwinTransformer models that extracts intermediate features
    from each stage and collapses the depth dimension via adaptive average pooling
    to produce 2D feature maps (NCHW) compatible with FPN.
    """
    patch_embed: nn.Module
    pos_drop: nn.Module
    layers: nn.ModuleList
    ape: bool
    absolute_pos_embed: torch.Tensor
    num_features: List[int]
    patches_resolution: Tuple[int, int, int]

    def __init__(self, swin_model: nn.Module):
        super(SwinFeatureExtractor3D, self).__init__()
        self.patch_embed = swin_model.patch_embed  # type: ignore[attr-defined]
        self.pos_drop = swin_model.pos_drop  # type: ignore[attr-defined]
        self.layers = swin_model.layers  # type: ignore[attr-defined]
        self.ape = swin_model.ape  # type: ignore[attr-defined]
        if self.ape:
            self.absolute_pos_embed = swin_model.absolute_pos_embed  # type: ignore[attr-defined]

        self.num_features = [int(getattr(layer, "dim")) for layer in self.layers]
        self.patches_resolution = swin_model.patches_resolution  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features, collapsing depth to 2D for FPN.

        Args:
            x: (B, C, D, H, W) 5D volume tensor.

        Returns:
            List of (B, C_i, H_i, W_i) 2D feature maps per stage.
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for i, layer in enumerate(self.layers):
            blocks: nn.ModuleList = layer.blocks  # type: ignore[assignment]
            for block in blocks:
                x = block(x)

            B, L, C = x.shape
            D, H, W = layer.input_resolution  # type: ignore[union-attr]
            # Reshape to 5D: (B, D, H, W, C) -> (B, C, D, H, W)
            feat_5d = x.view(B, int(D), int(H), int(W), int(C)).permute(0, 4, 1, 2, 3).contiguous()
            # Collapse depth via adaptive average pooling -> (B, C, 1, H, W) -> (B, C, H, W)
            feat_2d = F.adaptive_avg_pool3d(feat_5d, (1, int(H), int(W))).squeeze(2)
            features.append(feat_2d)

            downsample: Optional[nn.Module] = layer.downsample  # type: ignore[assignment]
            if downsample is not None:
                x = downsample(x)

        return features


class FlexibleBackbone(nn.Module):
    """
    Pure backbone wrapper for timm/custom Swin models.
    Extracts multi-scale features without FPN - use BackboneWithFPN for detection tasks.
    Supports loading local checkpoints for fine-tuning.
    """
    def __init__(self, model_name: str, checkpoint_path: str = "", config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        super(FlexibleBackbone, self).__init__()
        
        print(f"Initializing Backbone: {model_name}")
        
        self._use_custom_swin = False
        self._is_3d = False

        if config is not None:
            if logger is not None:
                logger.info(f"Creating model from config file:{config.MODEL.TYPE}/{config.MODEL.NAME}")
            swin_model = build_model(config)
            if checkpoint_path != "":
                if logger is not None:
                    logger.info(f"=> Path to pretrained weights: '{config.MODEL.PRETRAINED}'")
                load_pretrained(config, swin_model, logger)
            
            # Wrap custom Swin model to extract multi-scale features
            if config.MODEL.TYPE in ('swin3d', 'swinv2_3d'):
                self.body = SwinFeatureExtractor3D(swin_model)
                self._is_3d = True
            else:
                self.body = SwinFeatureExtractor(swin_model)
            self._use_custom_swin = True
        else:
            import timm
            self.body = timm.create_model(model_name, pretrained=True, features_only=True)

        # Get channel counts from the backbone automatically
        if hasattr(self.body, "num_features"):
            # Case SwinFeatureExtractor
            self.in_channels_list = list(self.body.num_features)  # type: ignore[arg-type]
        elif hasattr(self.body, "feature_info"):
            # For timm models
            self.in_channels_list = list(self.body.feature_info.channels())  # type: ignore[union-attr]
        else:
            raise AttributeError("Backbone model does not give 'num_features' or 'feature_info.channels()'.")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of multi-scale feature maps from backbone stages."""
        xs = self.body(x)
        
        # Ensure NCHW format for all features
        features = []
        for i, feature in enumerate(xs):
            if feature.ndim == 4 and feature.shape[1] != self.in_channels_list[i]:
                feature = feature.permute(0, 3, 1, 2)
            features.append(feature)
        
        return features


class BackboneWithFPN(nn.Module):
    """
    Combines FlexibleBackbone with Feature Pyramid Network.
    Compatible with torchvision's detection models.
    """
    def __init__(self, backbone: FlexibleBackbone, out_channels: int = 256):
        super(BackboneWithFPN, self).__init__()
        
        self.backbone = backbone
        self.in_channels_list = backbone.in_channels_list
        
        # Create FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=out_channels
        )
        
        # Required attribute for torchvision detection models
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get features from backbone
        xs = self.backbone(x)
        
        # Prepare dictionary for FPN
        x_dict = OrderedDict()
        for i, feature in enumerate(xs):
            x_dict[f"{i}"] = feature
            
        # Pass through FPN
        fpn_out = self.fpn(x_dict)
        return fpn_out


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Focal loss for multi-class classification with optional label smoothing.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        logits: Raw predictions [B, C].
        targets: Integer class labels [B].
        alpha: Per-class weights [C] (optional).
        gamma: Focusing parameter to down-weight easy examples.
        label_smoothing: Label smoothing factor.
        reduction: 'mean' or 'sum'.
    """
    num_classes = logits.shape[1]
    
    with torch.no_grad():
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
    
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    
    focal_weight = (1 - probs) ** gamma
    
    if alpha is not None:
        focal_weight = focal_weight * alpha.unsqueeze(0)
    
    loss = -focal_weight * smooth_targets * log_probs
    loss = loss.sum(dim=1)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

    
class GlobalClassificationHead(nn.Module):
    """
    Global Classification Head with multi-scale feature aggregation.
    Concatenates pooled features from all FPN levels for richer representation.
    """
    def __init__(self, in_channels: int, num_classes: int, num_levels: int = 4):
        super(GlobalClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        concat_channels = in_channels * num_levels
        self.fc = nn.Sequential(
            nn.Linear(concat_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, features_dict: Dict[str, torch.Tensor], attention_masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        pooled = []
        for key, feat in features_dict.items():
            if attention_masks is not None and key in attention_masks:
                feat = feat * attention_masks[key]
            p = self.avgpool(feat)
            pooled.append(torch.flatten(p, 1))
        x = torch.cat(pooled, dim=1)
        x = self.fc(x)
        return x
    
class DualHeadSCLCModel(nn.Module):
    """
    Composite model wrapping Faster R-CNN and Global Classifier.
    
    Args:
        train_backbone_only: If True, freezes FPN, detection head, and global classifier
                            to train only the backbone as a baseline.
    """
    def __init__(self, backbone_type: str, checkpoint_path: str = "", config: Optional[Any] = None, 
                 num_detection_classes: int = 4, num_global_classes: int = 3, 
                 train_backbone_only: bool = False, logger: Optional[logging.Logger] = None):
        super(DualHeadSCLCModel, self).__init__()
        
        self._train_backbone_only = train_backbone_only
        self._is_3d = backbone_type in ("swin3d", "swinv2_3d")
        
        # Class weights for weighted CE loss
        self.register_buffer('class_weights', None)
        
        # Map simple names to timm model names
        backbone_map = {
            "swin": "swin_base_patch4_window7_224",
            "swinv2": "swinv2_base_window12to24_192to384",
            "swin3d": "swin3d_base_patch4_window7_224",
            "swinv2_3d": "swinv2_3d_base_patch4_window7_224",
            "resnet": "resnet50",
            "densenet": "densenet121"
        }
        
        if backbone_type not in backbone_map:
            raise ValueError(f"Unsupported backbone type: {backbone_type}. Choose from {list(backbone_map.keys())}")


        model_name = backbone_map[backbone_type]
        fpn_out_channels = 256
        
        # Initialize backbone (without FPN)
        self._backbone_core = FlexibleBackbone(model_name, checkpoint_path, config, logger=logger)
        
        # Initialize BackboneWithFPN wrapper for detection
        self.backbone = BackboneWithFPN(self._backbone_core, out_channels=fpn_out_channels)
        
        # RPN Anchor Generator
        num_feature_levels = len(self.backbone.in_channels_list)
        base_anchor_sizes = (8, 16, 32, 64, 128)
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
        num_feature_levels = len(self.backbone.in_channels_list)
        self.global_classifier = GlobalClassificationHead(
            in_channels=fpn_out_channels,
            num_classes=num_global_classes,
            num_levels=num_feature_levels
        )
        
        # Apply backbone-only training mode if requested
        if train_backbone_only:
            self._freeze_non_backbone_params()
    
    def _freeze_non_backbone_params(self) -> None:
        """Freeze all parameters except the backbone for baseline training."""
        # Freeze FPN
        for param in self.backbone.fpn.parameters():
            param.requires_grad = False
        
        # Freeze detection head (RPN and ROI heads)
        for param in self.detector.rpn.parameters():
            param.requires_grad = False
        for param in self.detector.roi_heads.parameters():
            param.requires_grad = False
        
        # Freeze global classifier
        for param in self.global_classifier.parameters():
            param.requires_grad = False
    
    def _unfreeze_all_params(self) -> None:
        """Unfreeze all parameters for full model training."""
        for param in self.parameters():
            param.requires_grad = True
    
    def set_train_backbone_only(self, enable: bool) -> None:
        """
        Toggle backbone-only training mode.
        
        Args:
            enable: If True, freezes non-backbone params. If False, unfreezes all.
        """
        self._train_backbone_only = enable
        if enable:
            self._freeze_non_backbone_params()
        else:
            self._unfreeze_all_params()

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        """Set class weights for weighted cross-entropy loss.
        
        Args:
            class_weights: Tensor of shape (num_classes,) with per-class weights.
                          Higher weight = more emphasis on that class.
        """
        self.class_weights = class_weights

    def _build_proposal_attention(
        self,
        proposals: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Build spatial attention masks from RPN proposals.
        
        Proposals indicate regions of interest detected by the RPN.
        These are converted to soft attention masks that guide the global
        classifier to focus on diagnostically relevant regions.
        """
        attention_masks = {}
        for key, feat in features.items():
            B, C, H, W = feat.shape
            masks = torch.ones(B, 1, H, W, device=feat.device) * 0.5
            
            for i, boxes in enumerate(proposals):
                if len(boxes) == 0:
                    masks[i] = 1.0
                    continue
                
                ih, iw = image_sizes[i]
                scale_h = H / ih
                scale_w = W / iw
                
                top_k = min(16, len(boxes))
                for box in boxes[:top_k]:
                    x1, y1, x2, y2 = box.detach()
                    fx1 = max(0, int(x1.item() * scale_w))
                    fy1 = max(0, int(y1.item() * scale_h))
                    fx2 = min(W, max(fx1 + 1, int(x2.item() * scale_w)))
                    fy2 = min(H, max(fy1 + 1, int(y2.item() * scale_h)))
                    masks[i, :, fy1:fy2, fx1:fx2] = 1.0
            
            attention_masks[key] = masks
        return attention_masks
        
    def forward(self, scans: List[torch.Tensor], targets: Optional[List[Dict]] = None,
                return_logits: bool = False) -> Union[Dict[str, torch.Tensor], Tuple[List[Dict], torch.Tensor]]:
        if self._is_3d:
            # 3D path: scans are (C, H, W, D) tensors from MONAI 3D transforms.
            # Stack into (B, C, H, W, D) then permute to (B, C, D, H, W) for Conv3d.
            batch_5d = torch.stack(scans, dim=0)  # (B, C, H, W, D)
            batch_5d = batch_5d.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, D, H, W)
            features = self.backbone(batch_5d)
            # Build ImageList with 2D spatial sizes (H, W) for RPN/ROI heads
            _, _, _, sH, sW = batch_5d.shape
            image_sizes = [(sH, sW)] * len(scans)
            # Create a dummy 2D tensor view for ImageList (RPN needs .tensors attribute)
            # Use any 2D FPN feature to infer the batched tensor shape expected
            first_feat = next(iter(features.values()))
            B = first_feat.shape[0]
            scans_transformed = ImageList(
                torch.zeros(B, 1, sH, sW, device=batch_5d.device), image_sizes
            )
            targets_transformed = targets
        else:
            scans_transformed, targets_transformed = self.detector.transform(scans, targets)
            features = self.backbone(scans_transformed.tensors)
        
        # Detection head - generates proposals and computes detection losses
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
        
        # Proposal-guided attention for classification
        attention_masks = self._build_proposal_attention(
            proposals, scans_transformed.image_sizes, features
        )
        global_logits = self.global_classifier(features, attention_masks=attention_masks)
        
        if self.training:
            if targets_transformed is None:
                raise ValueError(
                    "DualHeadSCLCModel.forward expected 'targets' with 'scan_label' "
                    "for each sample during training, but got None."
                )
            gt_labels = torch.stack([t["scan_label"] for t in targets_transformed])
            global_loss = focal_loss(
                global_logits, gt_labels, alpha=self.class_weights
            )
            losses['global_classification_loss'] = global_loss
            if return_logits:
                losses['global_logits'] = global_logits
            return losses
        else:
            global_probabilities = F.softmax(global_logits, dim=1)
            return detections, global_probabilities
        
def get_sclc_model(
    backbone_type: str = "swinv2",
    checkpoint_path: str = "",
    num_detection_classes: int = 4,
    num_global_classes: int = 3,
    train_backbone_only: bool = False,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
) -> DualHeadSCLCModel:
    """
    Factory function to create DualHeadSCLCModel with specified backbone.
    
    Args:
        backbone_type: One of 'swin', 'swinv2', 'resnet', 'densenet'
        checkpoint_path: Path to pretrained weights (.pth file)
        num_detection_classes: Number of detection classes (including background).
            Default 4: background + A(Adenocarcinoma) + B(Small Cell) + G(Squamous Cell)
        num_global_classes: Number of global classification classes.
            Default 3: A(Adenocarcinoma), B(Small Cell), G(Squamous Cell)
        train_backbone_only: If True, freezes FPN and heads to train only backbone
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
        train_backbone_only=train_backbone_only,
        logger=logger
    )
    return model
