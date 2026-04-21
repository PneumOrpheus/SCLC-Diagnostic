import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR, resnet50, resnet18, ViT, DenseNet121, EfficientNetBN
import os
import sys

# Add ModelsGenesis/pytorch to sys.path to allow importing unet3d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ModelsGenesis/pytorch")))
from unet3d import UNet3D

class ModelsGenesisClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.base_model = UNet3D(n_class=num_classes)
        
        if in_channels != 1:
            conv1 = self.base_model.down_tr64.ops[0].conv1
            new_conv = nn.Conv3d(in_channels, conv1.out_channels, kernel_size=conv1.kernel_size, padding=conv1.padding)
            with torch.no_grad():
                new_conv.weight[:, 0:1] = conv1.weight
                if in_channels > 1:
                    new_conv.weight[:, 1:] = conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels-1, 1, 1, 1)
                new_conv.bias = conv1.bias
            self.base_model.down_tr64.ops[0].conv1 = new_conv

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x, return_segmentation=False):
        out64, skip_out64 = self.base_model.down_tr64(x)
        out128, skip_out128 = self.base_model.down_tr128(out64)
        out256, skip_out256 = self.base_model.down_tr256(out128)
        out512, skip_out512 = self.base_model.down_tr512(out256)
        
        pooled = self.pool(out512).flatten(1)
        
        out = torch.nn.functional.relu(self.dense_1(pooled))
        cls_logits = self.dense_2(out)
        
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        else:
            return cls_logits

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
        # Single linear head: with ~300 training volumes and 3 classes, a 2-layer
        # head (197K params) overfits. The pretrained encoder features should be
        # linearly separable if they encode the right information.
        self.classification_head = nn.Linear(bottleneck_channels, num_classes)
        
        # Register a hook to capture the deepest features during full forward pass
        self.deepest_features = None
        def vit_hook(module, input, output):
            self.deepest_features = output[-1]
        self.swin_unetr.swinViT.register_forward_hook(vit_hook)

    def forward(self, x, return_segmentation=False):
        self.deepest_features = None
        if return_segmentation:
            # populates self.deepest_features with the final encoder bottleneck before the decoder.
            seg_logits = self.swin_unetr(x)
            
            # Pool spatially to [B, C, 1, 1, 1] then flatten to [B, C]
            pooled = self.global_pool(self.deepest_features).flatten(1)
            
            # Get class logits [B, num_classes]
            cls_logits = self.classification_head(pooled)
            return cls_logits, seg_logits
        else:
            # Bypass the heavy decoder entirely for classification-only runs
            hidden_states = self.swin_unetr.swinViT(x.contiguous())
            bottleneck = hidden_states[-1]
            pooled = self.global_pool(bottleneck).flatten(1)
            cls_logits = self.classification_head(pooled)
            return cls_logits


class ResNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Initialize MONAI's 3D ResNet50. Setting feed_forward=False allows loading MedicalNet weights
        self.resnet = resnet50(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
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


class ResNet18Classifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Initialize MONAI's 3D ResNet18.
        self.resnet = resnet18(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True
        )
        
        # Adding a custom classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
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


class ViTClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, img_size=(224, 224, 128)):
        super().__init__()
        # Initialize MONAI's Vision Transformer
        # patch_size should be a divisor of the image dimensions.
        # We use a smaller patch size and fewer layers for a "tiny" ViT
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=(32, 32, 32),
            spatial_dims=3,
            classification=False, # We'll pool the hidden states manually or use their default
            hidden_size=384,
            mlp_dim=1536,
            num_layers=6,
            num_heads=6,
            dropout_rate=0.1
        )
        
        # Adding a custom classification head
        self.classification_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_segmentation=False):
        hidden_states, _ = self.vit(x)
        # hidden_states is [B, Num_Tokens, Hidden_Size] -> we need [B, Hidden_Size]
        pooled = hidden_states.mean(dim=1)
            
        cls_logits = self.classification_head(pooled)
        
        if return_segmentation:
            # ViT purely for classification doesn't naturally output a segmentation mask, return a dummy mask
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        else:
            return cls_logits


class DenseNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Initialize MONAI's 3D DenseNet121
        self.densenet = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes
        )

    def forward(self, x, return_segmentation=False):
        cls_logits = self.densenet(x)
        
        if return_segmentation:
            # DenseNet purely for classification doesn't naturally output a segmentation mask, return a dummy zero mask
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        else:
            return cls_logits


# Pipeline selection by model-type suffix.
#   _2d   -> single axial slice containing tumor, 2D CNN (in_channels=1)
#   _2p5d -> N adjacent axial slices as channels, 2D CNN
#   else  -> full 3D volume
TWO_P_FIVE_D_MODEL_TYPES = ("efficientnet_b0_2p5d",)
TWO_D_MODEL_TYPES = ("efficientnet_b0_2d",)


def is_2p5d_model_type(model_type: str) -> bool:
    return model_type.lower() in TWO_P_FIVE_D_MODEL_TYPES


def is_2d_model_type(model_type: str) -> bool:
    return model_type.lower() in TWO_D_MODEL_TYPES


def get_pipeline(model_type: str) -> str:
    if is_2d_model_type(model_type):
        return "2d"
    if is_2p5d_model_type(model_type):
        return "2p5d"
    return "3d"


class EfficientNet2DClassifier(nn.Module):
    """2D classifier: one axial slice -> EfficientNet-B0 (ImageNet-pretrained).

    The slice is fed single-channel; MONAI re-initializes the stem conv for
    ``in_channels != 3`` and keeps the rest pretrained. Pure 2D baseline for
    comparison against 2.5D and 3D.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.efficientnet = EfficientNetBN(
            "efficientnet-b0",
            pretrained=True,
            spatial_dims=2,
            in_channels=1,
            num_classes=num_classes,
        )

    def forward(self, x, return_segmentation=False):
        # Expected (B, 1, H, W). Collapse an accidental trailing depth dim.
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        cls_logits = self.efficientnet(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits


class EfficientNet2p5DClassifier(nn.Module):
    """2.5D classifier: MONAI EfficientNet-B0 (2D, ImageNet-pretrained) with
    ``num_slices`` axial slices fed in as input channels. The first conv is
    re-initialized for ``in_channels != 3`` but the rest of the network keeps
    its pretrained weights, which is the standard 2.5D transfer setup.
    """

    def __init__(self, num_slices: int = 5, num_classes: int = 3):
        super().__init__()
        self.num_slices = int(num_slices)
        self.efficientnet = EfficientNetBN(
            "efficientnet-b0",
            pretrained=True,
            spatial_dims=2,
            in_channels=self.num_slices,
            num_classes=num_classes,
        )

    def forward(self, x, return_segmentation=False):
        # x is (B, num_slices, H, W). If 3D upstream code accidentally hands us
        # (B, C, D, H, W) with C=1, collapse D into the channel axis.
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        cls_logits = self.efficientnet(x)
        if return_segmentation:
            # Spatial dims of x are (H, W) in 2.5D; mirror the 3D API by
            # returning a zero seg tensor so the shared training loop works.
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits


def get_sclc_model(checkpoint_path: str = "", model_type: str = "swin_unetr", in_channels: int = 1, depth_size: int = 128, num_slices: int = 5) -> nn.Module:
    if model_type.lower() == "efficientnet_b0_2d":
        model = EfficientNet2DClassifier(num_classes=3)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading 2D checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "efficientnet_b0_2p5d":
        model = EfficientNet2p5DClassifier(num_slices=num_slices, num_classes=3)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading 2.5D checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Matched {matched}/{len(state_dict)} keys.")
        return model
    if model_type.lower() == "resnet50":
        model = ResNetClassifier(in_channels=in_channels, num_classes=3)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading pretrained ResNet weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            missing, unexpected = model.resnet.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Pretrained weights loaded. Matched {matched}/{len(state_dict)} keys.")
            if matched == 0:
                print(f"[!] Warning: 0 keys matched! Checkpoint {checkpoint_path} is likely for a different architecture.")
    elif model_type.lower() == "resnet18":
        model = ResNet18Classifier(in_channels=in_channels, num_classes=3)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading pretrained ResNet18 weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            missing, unexpected = model.resnet.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Pretrained weights loaded. Matched {matched}/{len(state_dict)} keys.")
            if matched == 0:
                print(f"[!] Warning: 0 keys matched! Checkpoint {checkpoint_path} is likely for a different architecture.")
    elif model_type.lower() == "vit":
        raise ValueError(
            "model_type='vit' is disabled: a 27M-param transformer cannot learn a "
            "3-way label from ~300 CT volumes from scratch, and no 3D-ViT pretrained "
            "checkpoint is wired up. Use swin_unetr or resnet18/50 instead."
        )
    elif model_type.lower() == "densenet121":
        model = DenseNetClassifier(in_channels=in_channels, num_classes=3)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading pretrained DenseNet121 weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.densenet.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(unexpected)
            print(f"[*] Pretrained weights loaded. Matched {matched}/{len(state_dict)} keys.")
            if matched == 0:
                print(f"[!] Warning: 0 keys matched! Checkpoint {checkpoint_path} is likely for a different architecture.")
    elif model_type.lower() == "models_genesis":
        model = ModelsGenesisClassifier(in_channels=in_channels, num_classes=3)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading pretrained Models Genesis weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            unParalled_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("module.", "")
                unParalled_state_dict[new_key] = state_dict[key]
                
            # Pop mismatched keys
            if 'out_tr.final_conv.weight' in unParalled_state_dict:
                unParalled_state_dict.pop('out_tr.final_conv.weight')
            if 'out_tr.final_conv.bias' in unParalled_state_dict:
                unParalled_state_dict.pop('out_tr.final_conv.bias')
                
            missing, unexpected = model.base_model.load_state_dict(unParalled_state_dict, strict=False)
            matched = len(unParalled_state_dict) - len(unexpected)
            print(f"[*] Pretrained weights loaded. Matched {matched}/{len(unParalled_state_dict)} keys.")
            if matched == 0:
                print(f"[!] Warning: 0 keys matched! Checkpoint {checkpoint_path} is likely for a different architecture.")
    else:
        model = SwinUNETRClassifier(in_channels=in_channels, num_classes=3)
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
                        
                missing, unexpected = model.swin_unetr.load_state_dict(state_dict, strict=False)
                matched = len(state_dict) - len(unexpected)
                print(f"[*] Pretrained weights loaded. Matched {matched}/{len(state_dict)} keys.")
                if matched == 0:
                    print(f"[!] Warning: 0 keys matched! Checkpoint {checkpoint_path} is likely for a different architecture.")
            else:
                print(f"[!] Warning: Checkpoint path {checkpoint_path} does not exist. Initializing from scratch.")
    return model
