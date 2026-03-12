import sys
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler
from monai.data.dataloader import DataLoader
import numpy as np
import os
import argparse
from typing import Dict, List, Optional, Any, Tuple
import json
from collections import Counter

from models.model_selection import get_sclc_model
from models.config import get_config
from training.train import (
    sclc_collate_fn,
    train_epoch,
    validate_epoch
)
from data.biglunge_loader import (
    create_biglunge_dataset,
    CLASS_NAMES
)
from data.lung_pet_ct_dx_loader import (
    create_lung_pet_ct_dataset,
)
from logger import create_logger

"""
SCLC Diagnostic System - Main Pipeline
--------------------------------------

Implements a complete pipeline for:
1. Domain-Adaptive Pre-Training (DAPT) of backbone on Lung-PET-CT-Dx dataset
2. Fine-tuning the full model on the BigLunge dataset
3. Inference on the test set

The training flow:
    RadImageNet/timm model weights -> DAPT on Lung-PET-CT-Dx -> Fine-tune on BigLunge -> Test

Usage:
    # Full pipeline (DAPT + fine-tune + test)
    python main.py --mode full
    
    # DAPT only (pre-train backbone on Lung-PET-CT-Dx)
    python main.py --mode dapt
    
    # Fine-tune only (requires pre-trained checkpoint)
    python main.py --mode finetune --pretrained-checkpoint <path>
    
    # Inference only
    python main.py --mode inference --model-checkpoint <path>
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCLC Main Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        
Examples:
  # Full pipeline (DAPT + fine-tune + test)
  python main.py --mode full
  
  # DAPT only (pre-train backbone)
  python main.py --mode dapt --dapt-epochs 20
  
  # Fine-tune only (load pre-trained checkpoint)
  python main.py --mode finetune \\
      --pretrained-checkpoint checkpoints/dapt_swinv2_best.pth
  
  # Inference only
  python main.py --mode inference \\
      --model-checkpoint checkpoints/finetune_swinv2_best.pth
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "dapt", "finetune", "inference"],
        help="Pipeline mode: full (DAPT+fine-tune+test), dapt (backbone pre-training only), "
             "finetune (requires pre-trained checkpoint), or inference"
    )

    # Model configuration
    parser.add_argument(
        "--backbone",
        type=str,
        default="swinv2",
        choices=["swin", "swinv2", "swin3d", "swinv2_3d", "resnet", "densenet"],
        help="Backbone model architecture"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/data/RadImageNet/RadImageNet_swin/rin_config.yaml",
        help="Path to model config file"
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=str,
        default="/home/data/RadImageNet/RadImageNet_swin/rin_swintf.pth",
        help="Path to initial backbone checkpoint (RadImageNet weights)"
    )

    # Dataset paths - DAPT (Domain-Adaptive Pre-Training)
    parser.add_argument(
        "--dapt-backbone-dataset",
        type=str,
        default="/home/data/Lung-PET-CT-Dx",
        help="Path to DAPT dataset for backbone pre-training (Lung-PET-CT-Dx)"
    )

    # Dataset paths - Fine-tuning
    parser.add_argument(
        "--fine-tuning-dataset",
        type=str,
        default="/home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1",
        help="Path to fine-tuning dataset (BigLunge)"
    )
    parser.add_argument(
        "--fine-tuning-csv",
        type=str,
        default="/home/data/BigLunge/patients_parameters.csv",
        help="Path to BigLunge patient labels CSV file"
    )

    # Checkpoint paths for resume/inference
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="",
        help="Path to pre-trained checkpoint (for finetune mode)"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="",
        help="Path to final model checkpoint (for inference mode)"
    )

    # Training hyperparameters - DAPT phase
    parser.add_argument("--dapt-epochs", type=int, default=50,
                        help="Number of epochs for DAPT (backbone pre-training)")
    parser.add_argument("--dapt-lr", type=float, default=1e-4,
                        help="Learning rate for DAPT phase")

    # Training hyperparameters - Fine-tuning phase
    parser.add_argument("--finetune-epochs", type=int, default=100,
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--finetune-lr", type=float, default=3e-5,
                        help="Learning rate for fine-tuning")

    # Common hyperparameters
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")

    # Output directories
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for logs")
    parser.add_argument("--checkpoint-dir", type=str, default="/home/data/trained_models", help="Checkpoint directory")

    # Early stopping
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without improvement)")

    # Annotation directory for bounding boxes (Lung-PET-CT-Dx)
    parser.add_argument("--annotation-dir", type=str, default="/home/data/Annotation",
                        help="Path to annotation directory with per-patient XML bounding boxes")

    # Performance / acceleration
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--disable-amp", action="store_true",
                        help="Disable automatic mixed precision (AMP) training")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch-size * accumulation-steps)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0.0 = disabled)")
    parser.add_argument("--clip-grad", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled)")

    # 3D backbone options
    parser.add_argument("--depth-size", type=int, default=16,
                        help="Number of depth slices for 3D backbone models")

    return parser.parse_args()


def compute_class_weights(data_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataloader.
    
    Returns:
        Tensor of shape (num_classes,) with weights inversely proportional to class frequency.
    """
    class_counts = Counter()
    for _, targets in data_loader:
        for t in targets:
            label = t["scan_label"].item() if isinstance(t["scan_label"], torch.Tensor) else t["scan_label"]
            class_counts[label] += 1
    
    total = sum(class_counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 1)  # avoid division by zero
        weights[cls_idx] = total / (num_classes * count)
    
    return weights.to(device)

def create_dataloaders(
    data_path: str,
    batch_size: int,
    device: torch.device,
    dataset_type: str = "lung_pet_ct",
    csv_path: str = "",
    convert_to_rgb: bool = True,
    num_workers: int = 4,
    annotation_dir: str = "",
    use_multichannel_windowing: bool = False,
    use_3d: bool = False,
    depth_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders for specified dataset."""

    if dataset_type == "biglunge":
        train_dataset = create_biglunge_dataset(
            data_path=data_path,
            csv_path=csv_path,
            split="train",
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing,
            num_workers=num_workers,
            use_3d=use_3d,
            depth_size=depth_size,
        )
        val_dataset = create_biglunge_dataset(
            data_path=data_path,
            csv_path=csv_path,
            split="val",
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing,
            num_workers=num_workers,
            use_3d=use_3d,
            depth_size=depth_size,
        )
        test_dataset = create_biglunge_dataset(
            data_path=data_path,
            csv_path=csv_path,
            split="test",
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing,
            num_workers=num_workers,
            use_3d=use_3d,
            depth_size=depth_size,
        )
    else:  # lung_pet_ct
        train_dataset = create_lung_pet_ct_dataset(
            data_path=data_path,
            split="train",
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing,
            num_workers=num_workers,
            annotation_dir=annotation_dir,
            use_3d=use_3d,
            depth_size=depth_size,
        )
        val_dataset = create_lung_pet_ct_dataset(
            data_path=data_path,
            split="val",
            convert_to_rgb=convert_to_rgb,
            use_multichannel_windowing=use_multichannel_windowing,
            num_workers=num_workers,
            annotation_dir=annotation_dir,
            use_3d=use_3d,
            depth_size=depth_size,
        )
        test_dataset = create_lung_pet_ct_dataset(
            data_path=data_path,
            split="test",
            convert_to_rgb=convert_to_rgb,
            num_workers=num_workers,
            annotation_dir=annotation_dir,
            use_3d=use_3d,
            depth_size=depth_size,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=sclc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sclc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sclc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader, test_loader


def run_dapt_phase(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_dir: str,
    logger,
    backbone_type: str,
    patience: int = 10,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1,
    clip_grad: float = 1.0,
    label_smoothing: float = 0.0,
) -> str:
    """
    Phase 1: Domain-Adaptive Pre-Training (DAPT)
    
    Pre-trains the backbone, FPN, and detection head on Lung-PET-CT-Dx dataset
    using bounding box annotations. Only the global classifier is frozen.
    This helps the backbone learn lung CT-specific spatial features.
    
    Returns:
        Path to best checkpoint.
    """
    logger.info("-" * 70)
    logger.info("PHASE 1: Domain-Adaptive Pre-Training (DAPT)")
    logger.info("-" * 70)
    logger.info(f"Epochs: {epochs}, Learning Rate: {lr}, Patience: {patience}")
    logger.info("Mode: Backbone-only training (FPN, detection, classifier frozen)")

    # Freeze everything except backbone for stable pre-training
    model.set_train_backbone_only(True)

    # Optimize backbone parameters only
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in params):,}")

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    # LR schedule - linear warmup for 5 epochs then cosine decay
    warmup_epochs = min(5, epochs // 4)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_checkpoint_path = ""
    epochs_without_improvement = 0

    for epoch in range(epochs):
        logger.info(f"\n--- DAPT Epoch {epoch+1}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.2e}) ---")

        # Train
        train_metrics = train_epoch(
            model, optimizer, train_loader, device, epoch + 1,
            scaler=scaler, accumulation_steps=accumulation_steps,
            clip_grad=clip_grad, label_smoothing=label_smoothing,
        )

        # Validation
        val_metrics = validate_epoch(model, val_loader, device, phase="val")

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Grad Norm: {train_metrics.get('grad_norm', 0):.4f}"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(
                checkpoint_dir, f"dapt_{backbone_type}_best.pth"
            )
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"New best DAPT model saved: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs (patience={patience})")
                break

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"dapt_{backbone_type}_epoch_{epoch+1}.pth"
            )
            save_dict = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "phase": "dapt"
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, checkpoint_path)
            logger.info(f"Saved DAPT checkpoint: {checkpoint_path}")

    logger.info(f"\nDAPT Phase complete. Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


def run_finetune_phase(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_dir: str,
    logger,
    backbone_type: str,
    patience: int = 10,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1,
    clip_grad: float = 1.0,
    label_smoothing: float = 0.0,
) -> str:
    """
    Phase 2: Fine-tuning on BigLunge Dataset
    
    Fine-tunes the complete model (backbone + FPN + heads) on the target dataset.
    Uses differential learning rates: lower for backbone, higher for heads.
    
    Returns:
        Path to best checkpoint.
    """
    logger.info("-" * 70)
    logger.info("PHASE 2: Fine-tuning on BigLunge Dataset")
    logger.info("-" * 70)
    logger.info(f"Epochs: {epochs}, Base Learning Rate: {lr}, Patience: {patience}")
    logger.info("Mode: Full model training (all layers unfrozen)")
    logger.info("Using differential LR: backbone=0.2x, FPN/heads=1x")

    # Unfreeze all layers for fine-tuning
    model.set_train_backbone_only(False)

    # Backbone gets 5x lower LR than heads
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name and "fpn" not in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    param_groups = [
        {"params": backbone_params, "lr": lr * 0.2},
        {"params": head_params, "lr": lr},
    ]
    
    logger.info(f"Backbone params: {sum(p.numel() for p in backbone_params):,} (LR: {lr * 0.2:.2e})")
    logger.info(f"Head params: {sum(p.numel() for p in head_params):,} (LR: {lr:.2e})")
    logger.info(f"Total trainable: {sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params):,}")

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # LR schedule - linear warmup for 5 epochs then cosine decay
    warmup_epochs = min(5, epochs // 4)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_checkpoint_path = ""
    epochs_without_improvement = 0

    for epoch in range(epochs):
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[1]['lr']
        logger.info(f"\n--- Fine-tune Epoch {epoch+1}/{epochs} (LR backbone: {current_lr_backbone:.2e}, head: {current_lr_head:.2e}) ---")

        # Train with mixup augmentation
        train_metrics = train_epoch(
            model, optimizer, train_loader, device, epoch + 1,
            use_mixup=True, mixup_alpha=0.2,
            scaler=scaler, accumulation_steps=accumulation_steps,
            clip_grad=clip_grad, label_smoothing=label_smoothing,
        )

        # Validation
        val_metrics = validate_epoch(model, val_loader, device, phase="val")

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Det Loss: {val_metrics['det_loss']:.4f}, "
            f"Global Loss: {val_metrics['global_loss']:.4f}, "
            f"Grad Norm: {train_metrics.get('grad_norm', 0):.4f}"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(
                checkpoint_dir, f"finetune_{backbone_type}_best.pth"
            )
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"New best fine-tuned model saved: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs (patience={patience})")
                break

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"finetune_{backbone_type}_epoch_{epoch+1}.pth"
            )
            save_dict = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "phase": "finetune"
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, checkpoint_path)
            logger.info(f"Saved fine-tune checkpoint: {checkpoint_path}")

    logger.info(f"\nFine-tuning complete. Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    logger,
    output_dir: str
) -> Dict[str, Any]:
    """
    Phase 3: Inference on Test Set
    
    Evaluates the model on the test set and computes classification metrics.
    
    Returns:
        Dictionary containing predictions and metrics.
    """
    logger.info("-" * 70)
    logger.info("PHASE 3: Inference on BigLunge Test Set")
    logger.info("-" * 70)

    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []
    results = []

    for batch_idx, (scans, targets) in enumerate(test_loader):
        scans = [scan.to(device) for scan in scans]

        # Forward pass (model in eval mode returns (detections, global_probs))
        outputs = model(scans)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            detections, probs = outputs
        else:
            probs = outputs
            detections = [{}] * len(scans)

        preds = torch.argmax(probs, dim=1)

        for i, (pred, prob) in enumerate(zip(preds, probs)):
            target = targets[i]
            gt_label = target["scan_label"].item() if isinstance(target["scan_label"], torch.Tensor) else target["scan_label"]

            all_predictions.append(pred.item())
            all_labels.append(gt_label)
            all_probs.append(prob.cpu().numpy().tolist())

            results.append({
                "batch_idx": batch_idx,
                "sample_idx": i,
                "prediction": pred.item(),
                "prediction_class": CLASS_NAMES[pred.item()] if pred.item() < len(CLASS_NAMES) else "Unknown",
                "ground_truth": gt_label,
                "ground_truth_class": CLASS_NAMES[gt_label] if gt_label < len(CLASS_NAMES) else "Unknown",
                "probabilities": prob.cpu().numpy().tolist(),
                "correct": pred.item() == gt_label
            })

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_predictions == all_labels)

    # Per-class metrics
    class_metrics = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_mask = all_labels == class_idx

        tp = np.sum((all_predictions == class_idx) & (all_labels == class_idx))
        fp = np.sum((all_predictions == class_idx) & (all_labels != class_idx))
        fn = np.sum((all_predictions != class_idx) & (all_labels == class_idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(class_mask))
        }

    # Confusion matrix
    num_classes = len(CLASS_NAMES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_predictions, all_labels):
        if pred < num_classes and label < num_classes:
            confusion_matrix[label, pred] += 1

    metrics = {
        "accuracy": float(accuracy),
        "num_samples": len(all_predictions),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion_matrix.tolist()
    }

    # Log results
    logger.info(f"\n{'-'*50}")
    logger.info("TEST RESULTS")
    logger.info(f"{'-'*50}")
    logger.info(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(all_predictions))}/{len(all_predictions)})")
    logger.info(f"\nPer-class metrics:")
    for class_name, m in class_metrics.items():
        logger.info(
            f"  {class_name}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, Support={m['support']}"
        )

    logger.info(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    header = "            " + " ".join([f"{name[:8]:>8}" for name in CLASS_NAMES])
    logger.info(header)
    for i, class_name in enumerate(CLASS_NAMES):
        row = f"{class_name[:12]:<12}" + " ".join([f"{confusion_matrix[i,j]:>8}" for j in range(num_classes)])
        logger.info(row)

    # Save results to JSON
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "predictions": results
        }, f, indent=2)
    logger.info(f"\nDetailed results saved to: {results_path}")

    return metrics


def main():
    args = parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Reproducibility seeding
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # cuDNN auto-tuner
    cudnn.benchmark = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AMP setup
    amp_enabled = (not args.disable_amp) and (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    # Setup logger
    logger = create_logger(
        output_dir=args.output_dir,
        dist_rank=-1,
        name=f"sclc_{args.backbone}"
    )

    logger.info("-" * 70)
    logger.info("SCLC Diagnostic System - Main Pipeline")
    logger.info("-" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"AMP (mixed precision): {amp_enabled}")
    logger.info(f"Gradient accumulation steps: {args.accumulation_steps}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info(f"Gradient clipping max norm: {args.clip_grad}")
    logger.info(f"DAPT Dataset: {args.dapt_backbone_dataset}")
    logger.info(f"Fine-tuning Dataset: {args.fine_tuning_dataset}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Checkpoint Directory: {args.checkpoint_dir}")

    # Determine if using timm model (for RGB conversion)
    uses_timm_model = not (args.initial_checkpoint and args.config)

    # --------------
    # Inference Mode
    # --------------
    if args.mode == "inference":
        if not args.model_checkpoint or not os.path.exists(args.model_checkpoint):
            raise ValueError(f"Model checkpoint required for inference: {args.model_checkpoint}")

        logger.info(f"Loading model from: {args.model_checkpoint}")
        model = get_sclc_model(
            backbone_type=args.backbone,
            checkpoint_path="",
            config=None,
            train_backbone_only=False,
            logger=logger
        )
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        model.to(device)

        # Create test dataloader for BigLunge
        use_3d = args.backbone in ("swin3d", "swinv2_3d")
        _, _, test_loader = create_dataloaders(
            data_path=args.fine_tuning_dataset,
            batch_size=args.batch_size,
            device=device,
            dataset_type="biglunge",
            csv_path=args.fine_tuning_csv,
            convert_to_rgb=uses_timm_model,
            num_workers=args.num_workers,
            use_3d=use_3d,
            depth_size=args.depth_size
        )

        metrics = run_inference(model, test_loader, device, logger, args.output_dir)
        logger.info("Inference complete!")
        return

    # -------------------------------------------------------------
    # Full or Dapt Mode - Initialize model with RadImageNet weights
    # -------------------------------------------------------------
    if args.mode in ["full", "dapt"]:
        logger.info(f"\nInitializing model with RadImageNet weights: {args.initial_checkpoint}")
        
        config = None
        if args.config and os.path.exists(args.config):
            config = get_config(args)

        model = get_sclc_model(
            backbone_type=args.backbone,
            checkpoint_path=args.initial_checkpoint,
            config=config,
            train_backbone_only=True,
            logger=logger
        )
        model.to(device)

        # Create DAPT dataloaders (with annotation bounding boxes)
        logger.info(f"\nLoading DAPT dataset from: {args.dapt_backbone_dataset}")
        logger.info(f"Annotation directory: {args.annotation_dir}")
        use_3d = args.backbone in ("swin3d", "swinv2_3d")
        dapt_train_loader, dapt_val_loader, _ = create_dataloaders(
            data_path=args.dapt_backbone_dataset,
            batch_size=args.batch_size,
            device=device,
            dataset_type="lung_pet_ct",
            convert_to_rgb=uses_timm_model,
            num_workers=args.num_workers,
            annotation_dir=args.annotation_dir,
            use_multichannel_windowing=True,
            use_3d=use_3d,
            depth_size=args.depth_size
        )

        # No class weights for DAPT - uniform weighting for stable backbone pre-training
        model.set_class_weights(None)
        logger.info("DAPT: Using uniform class weights (no weighting)")

        # Run DAPT phase
        dapt_checkpoint = run_dapt_phase(
            model=model,
            train_loader=dapt_train_loader,
            val_loader=dapt_val_loader,
            device=device,
            epochs=args.dapt_epochs,
            lr=args.dapt_lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            logger=logger,
            backbone_type=args.backbone,
            patience=args.patience,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            clip_grad=args.clip_grad,
            label_smoothing=args.label_smoothing,
        )

        if args.mode == "dapt":
            logger.info("\nDAPT-only mode complete. Exiting.")
            logger.info(f"Use --pretrained-checkpoint {dapt_checkpoint} for fine-tuning.")
            return

        # Continue to fine-tuning
        args.pretrained_checkpoint = dapt_checkpoint

    # ---------------------------------------------
    # Full or Finetune Mode - Fine-tune on BigLunge
    # ---------------------------------------------
    if args.mode in ["full", "finetune"]:
        if args.mode == "finetune":
            # Load from pre-trained checkpoint
            if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
                logger.info(f"\nLoading pre-trained checkpoint: {args.pretrained_checkpoint}")
                model = get_sclc_model(
                    backbone_type=args.backbone,
                    checkpoint_path="",
                    config=None,
                    train_backbone_only=False,
                    logger=logger
                )
                model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=device))
                model.to(device)
            else:
                # Initialize fresh model (no DAPT)
                logger.warning("No pre-trained checkpoint provided. Initializing with RadImageNet weights.")
                config = None
                if args.config and os.path.exists(args.config):
                    config = get_config(args)
                    
                model = get_sclc_model(
                    backbone_type=args.backbone,
                    checkpoint_path=args.initial_checkpoint,
                    config=config,
                    train_backbone_only=False,
                    logger=logger
                )
                model.to(device)

        # Create BigLunge dataloaders
        logger.info(f"\nLoading fine-tuning dataset from: {args.fine_tuning_dataset}")
        use_3d = args.backbone in ("swin3d", "swinv2_3d")
        finetune_train_loader, finetune_val_loader, test_loader = create_dataloaders(
            data_path=args.fine_tuning_dataset,
            batch_size=args.batch_size,
            device=device,
            dataset_type="biglunge",
            csv_path=args.fine_tuning_csv,
            convert_to_rgb=uses_timm_model,
            num_workers=args.num_workers,
            use_multichannel_windowing=True,
            use_3d=use_3d,
            depth_size=args.depth_size
        )

        # Compute class weights for BigLunge training set
        finetune_class_weights = compute_class_weights(finetune_train_loader, num_classes=3, device=device)
        model.set_class_weights(finetune_class_weights)
        logger.info(f"Fine-tuning class weights: {finetune_class_weights.cpu().numpy()}")

        # Run fine-tuning phase
        finetune_checkpoint = run_finetune_phase(
            model=model,
            train_loader=finetune_train_loader,
            val_loader=finetune_val_loader,
            device=device,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            logger=logger,
            backbone_type=args.backbone,
            patience=args.patience,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            clip_grad=args.clip_grad,
            label_smoothing=args.label_smoothing,
        )

        # Load best model for testing
        logger.info(f"\nLoading best fine-tuned model for testing: {finetune_checkpoint}")
        model.load_state_dict(torch.load(finetune_checkpoint, map_location=device))

        # Run inference on test set
        metrics = run_inference(model, test_loader, device, logger, args.output_dir)

    # --------
    # COMPLETE
    # --------
    logger.info("\n" + "-" * 70)
    logger.info("Pipeline Complete!")
    logger.info("-" * 70)
    logger.info(f"Checkpoints saved in: {args.checkpoint_dir}")
    logger.info(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
