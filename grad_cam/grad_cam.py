from matplotlib import pyplot as plt
from monai.networks.nets import TorchVisionFCModel
from monai.visualize import GradCAMpp
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureType,
    NormalizeIntensity,
    ToTensor,
    RepeatChannel,
    Resize,
)
import numpy as np
import torch
import torch.nn as nn

import cv2
from monai.visualize import GradCAM, GradCAMpp
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureType,
    ScaleIntensity,
    Resize,
    ToTensor,
    RepeatChannel,
    EnsureChannelFirst
)
import os
import sys

from models.model_selection import get_sclc_model

class SCLCModelWrapper(nn.Module):
    """
    Wraps the SCLC model to make it compatible with MONAI GradCAM.
    1. Accepts a Tensor input (B, C, H, W) instead of list[Tensor].
    2. Returns only the probability tensor instead of (detections, probs).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The SCLC model likely expects a list of tensors (one per image)
        # MONAI GradCAM passes a batch tensor.
        if isinstance(x, torch.Tensor):
            # Convert batch tensor to list of 3D tensors (C, H, W)
            input_list = [x[i] for i in range(x.shape[0])]
        else:
            input_list = x
            
        outputs = self.model(input_list)
        
        # Return only the classification probabilities/logits
        # Assuming output is (detections, probabilities)
        if isinstance(outputs, tuple):
            return outputs[1]
        return outputs

def get_target_layer(model, backbone_type):
    """
    Find the target layer name (string) for MONAI GradCAM.
    Targets the last FPN layer block, which produces the spatial feature map
    used by the GlobalClassificationHead.
    """
    try:
        # For DualHeadSCLCModel wrapped in SCLCModelWrapper:
        # model.backbone.fpn.layer_blocks is a ModuleList
        inner = model.model if hasattr(model, 'model') else model
        if hasattr(inner, 'backbone') and hasattr(inner.backbone, 'fpn'):
            # Find the string name matching the last FPN layer block
            target_module = inner.backbone.fpn.layer_blocks[-1]
            for name, mod in model.named_modules():
                if mod is target_module:
                    return name
                
        # Default fallback
        print(f"Warning: Could not automatically identify target layer for {backbone_type}.")
        print("Available top-level modules:", list(model._modules.keys()))
        return None
        
    except (AttributeError, IndexError):
        print("Error traversing model to find target layer.")
        return None

def preprocess_image(image_path, img_size=224, device="cpu"):
    """
    Load and preprocess image. For 3D volumes (NIfTI), extracts the middle
    axial slice to produce a 2D image compatible with the model.
    """
    loader = LoadImage(image_only=True)
    img = loader(image_path)
    img_np = np.array(img)
    
    # Handle 3D volumes: extract middle axial slice
    if img_np.ndim == 3:
        mid = img_np.shape[-1] // 2
        img_np = img_np[:, :, mid]
    
    transforms = Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensity(),
        Resize((img_size, img_size)),
        ToTensor(),
        RepeatChannel(repeats=3),
        EnsureType()
    ])
    
    img_tensor = transforms(img_np)
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor

def _detect_class_counts(state_dict):
    """
    Auto-detect num_detection_classes and num_global_classes from checkpoint state dict.
    """
    num_det = 5  # default
    num_global = 4  # default
    
    det_key = "detector.roi_heads.box_predictor.cls_score.weight"
    if det_key in state_dict:
        num_det = state_dict[det_key].shape[0]
    
    global_key = "global_classifier.fc.4.weight"
    if global_key in state_dict:
        num_global = state_dict[global_key].shape[0]
    
    return num_det, num_global


def use_grad_cam(
    model_path: str, 
    image_path: str, 
    backbone_type: str = "swinv2", 
    output_dir: str = "grad_cam/gradcam_output",
    target_class_idx: int = None,
    config=None
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_path}")
    
    # Load weights first to auto-detect architecture parameters
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    num_det, num_global = _detect_class_counts(state_dict)
    print(f"Detected num_detection_classes={num_det}, num_global_classes={num_global}")
    
    # Initialize model with matching architecture
    original_model = get_sclc_model(
        backbone_type=backbone_type, 
        checkpoint_path="", 
        config=config, 
        num_detection_classes=num_det,
        num_global_classes=num_global,
        train_backbone_only=False
    )
    
    original_model.load_state_dict(state_dict, strict=False)
    original_model.to(device)
    original_model.eval()
    
    # Wrap model first, then find target layer name within the wrapper
    model_wrapper = SCLCModelWrapper(original_model)
    
    # Identify target layer (returns string name relative to model_wrapper)
    target_layer_name = get_target_layer(model_wrapper, backbone_type)
    if target_layer_name is None:
        raise ValueError("Could not define target layer. Please check model structure.")
    print(f"Target layer found: {target_layer_name}")
    
    # Initialize MONAI GradCAM
    cam = GradCAMpp(nn_module=model_wrapper, target_layers=target_layer_name)
    
    # Load Image
    print(f"Processing image: {image_path}")
    input_tensor = preprocess_image(image_path, device=device)
    
    # Run Inference to get prediction first
    with torch.no_grad():
        probs = model_wrapper(input_tensor)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    class_idx = target_class_idx if target_class_idx is not None else pred_idx
    print(f"Generating GradCAM for class index: {class_idx}")

    # Generate CAM
    # GradCAM result shape: (B, 1, H, W)
    result = cam(x=input_tensor, class_idx=class_idx)
    
    # Visualization
    img_np = input_tensor.cpu().numpy()[0, 0, :, :] # Get first channel of original image
    heatmap = result.cpu().numpy()[0, 0, :, :]
    
    plt.figure(figsize=(10, 5))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("GradCAM Heatmap")
    plt.axis("off")
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img_np, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"Overlay (Class {class_idx})")
    plt.axis("off")
    
    save_path = os.path.join(output_dir, f"gradcam_{os.path.basename(image_path)}.png")
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example Usage
    # You can call this script directly to test
    import argparse
    from models.config import get_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth model file")
    parser.add_argument("--image", required=True, help="Path to image file (nii/png/npy)")
    parser.add_argument("--backbone", default="swinv2", help="Backbone type")
    parser.add_argument("--config", default="", help="Path to model config YAML (required for custom Swin models)")
    args = parser.parse_args()
    
    config = None
    if args.config and os.path.exists(args.config):
        config = get_config(args)
    
    use_grad_cam(args.model, args.image, args.backbone, config=config)


# python -m grad_cam.grad_cam --model /home/data/trained_models/finetune_swinv2_best.pth --image /home/data/Lung-PET-CT-Dx/Lung_Dx-A0263_1.3.6.1.4.1.14519.5.2.1.6655.2359.594718314657441527730748498440.nii.gz --config /home/data/RadImageNet/RadImageNet_swin/rin_config.yaml
