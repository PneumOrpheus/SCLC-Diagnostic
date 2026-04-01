import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Any, Tuple
import argparse
import time
from collections import Counter

from model_selection2 import get_sclc_model
from training.train2 import simple_collate_fn, train_epoch, validate_epoch
from data.data_loader import create_dataset
from logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description="SCLC Simplified 3D Classification Pipeline")
    
    # Mode selection
    parser.add_argument("--model-type", type=str, default="swin_unetr", choices=["swin_unetr", "resnet50"],
                        help="Model architecture to use")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "dapt", "finetune", "inference"],
                        help="Pipeline mode")
    
    # Datasets
    parser.add_argument("--dapt-dataset", type=str, default="/home/data/Lung-PET-CT-Dx-Clean")
    parser.add_argument("--pet-dir", type=str, default="/home/data/Lung-PET-CT-Dx_PET", help="Directory containing the PET NIfTI files")
    parser.add_argument("--use-pet", action="store_true", default=False, help="Include PET as the second input channel")
    parser.add_argument("--finetune-dataset", type=str, default="/home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1")
    parser.add_argument("--finetune-csv", type=str, default="/home/data/BigLunge/patients_parameters.csv")
    
    # Checkpoints
    parser.add_argument("--initial-checkpoint", type=str, default="/home/data/temp/model_swin_unetr_btcv_segmentation_v1.pt")
    parser.add_argument("--pretrained-checkpoint", type=str, default="")
    parser.add_argument("--model-checkpoint", type=str, default="")
    
    # Hyperparameters
    parser.add_argument("--dapt-epochs", type=int, default=30)
    parser.add_argument("--dapt-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps to increase effective batch size")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--checkpoint-dir", type=str, default="trained_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--testing", default=False, action="store_true", help="Run with a tiny subset for testing")
    parser.add_argument("--anno", default=True, action="store_true", help="Use annotations for multi-task segmentation learning")
    parser.add_argument("--depth-size", type=int, default=128, help="Depth size for the 3D images")

    return parser.parse_args()


def create_dataloaders(args, dataset_type, data_path, csv_path="", depth_size=64):
    train_ds, val_ds, test_ds = create_dataset(
        dataset_type=dataset_type,
        data_path=data_path,
        csv_path=csv_path,
        img_size=224,
        depth_size=depth_size,
        convert_to_rgb=False,
        use_multichannel_windowing=False,
        num_workers=args.num_workers,
        pet_dir=args.pet_dir if dataset_type == "lung_pet_ct_dx" else "",
        use_pet=args.use_pet if dataset_type == "lung_pet_ct_dx" else False,
        use_3d=True,
        testing=args.testing,
        warm_cache=False,
    )
    
    if dataset_type == "lung_pet_ct_dx" and hasattr(train_ds, "data") and len(train_ds.data) > 0:
        print(f"Applying WeightedRandomSampler for highly imbalanced dataset: {dataset_type}")
        train_labels = [item["scan_label"] for item in train_ds.data]
        class_counts = Counter(train_labels)
        num_samples = len(train_labels)
        
        # Invert class frequencies to create weights
        class_weights_dict = {cls: num_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights_dict[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        # Note: shuffle must be False when using a sampler
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=simple_collate_fn, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=simple_collate_fn, num_workers=args.num_workers)
        
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=simple_collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=simple_collate_fn, num_workers=args.num_workers)
    
    return train_loader, val_loader, test_loader


def run_training_phase(
    model, train_loader, val_loader, device, epochs, lr, weight_decay,
    checkpoint_dir, logger, phase_name, patience=20, scaler=None, use_segmentation=False, accumulation_steps=4,
    model_type="swin_unetr"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(checkpoint_dir, f"best_{phase_name}_phase.pth")
    
    num_classes = 3 
    
    print(f"\nComputing class distributions for the {phase_name} dataset...")
    logger.info(f"Computing class distributions for the {phase_name} dataset...")
    class_weights = compute_class_weights(train_loader, num_classes=num_classes, device=device)
    print(f"[{phase_name}] Automatically assigned Class Weights: {class_weights.cpu().numpy()}")
    logger.info(f"Class Weights: {class_weights.cpu().numpy()}")

    for epoch in range(1, epochs + 1):
        print(f"\n--- {phase_name} Epoch {epoch}/{epochs} ---")
        
        train_loss = train_epoch(
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            epoch=epoch, 
            device=device, 
            logger=logger, 
            scaler=scaler,
            use_segmentation=use_segmentation,
            class_weights=class_weights,
            accumulation_steps=accumulation_steps # Pass down here
        )
        val_metrics = validate_epoch(model, val_loader, device, logger)
        
        scheduler.step()
        
        acc = val_metrics["accuracy"]
        
        # Clean up phase name for saving (e.g. 'DAPT Phase' -> 'dapt')
        phase_prefix = phase_name.lower().replace(' ', '_').replace('_phase', '')
        
        if acc > best_acc:
            best_acc = acc
            best_ckpt = os.path.join(checkpoint_dir, f"best_{model_type}_{phase_prefix}.pth")
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"[*] New best model saved! Accuracy: {best_acc:.4f}")
            epochs_no_improve = 0  # reset counter
        else:
            epochs_no_improve += 1
            
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_ckpt = os.path.join(checkpoint_dir, f"{phase_prefix}_{model_type}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), periodic_ckpt)
            logger.info(f"[*] Periodic checkpoint saved at epoch {epoch}: {periodic_ckpt}")
            
        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered. No improvement for {patience} epochs.")
            break
            
    return best_ckpt


def compute_class_weights(data_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataloader."""
    class_counts = Counter()

    dataset = getattr(data_loader, "dataset", None)
    metadata = getattr(dataset, "data", None)
    
    # Fast path: check dataset metadata directly if possible
    if isinstance(metadata, list) and len(metadata) > 0 and "scan_label" in metadata[0]:
        for item in metadata:
            class_counts[int(item["scan_label"])] += 1
    # Fallback to iterating dataloader
    else:
        for batch_data in data_loader:
            if len(batch_data) == 3:
                _, targets, _ = batch_data
            else:
                _, targets = batch_data
                
            for t in targets:
                label = t.item() if isinstance(t, torch.Tensor) else t
                class_counts[label] += 1
    
    total = sum(class_counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 1)  # avoid division by zero
        weights[cls_idx] = total / (num_classes * count)
    
    return weights.to(device)


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=not args.disable_amp and device.type == "cuda")
    
    logger = create_logger(output_dir=args.output_dir, dist_rank=-1, name=f"sclc_{args.model_type}")
    logger.info(f"Running {args.model_type} 3D Classification Pipeline")
    logger.info(f"Mode: {args.mode} | Testing: {args.testing} | Device: {device} | AMP: {not args.disable_amp}")
    
    in_channels = 2 if getattr(args, "use_pet", False) else 1
    model = get_sclc_model(args.initial_checkpoint, model_type=args.model_type, in_channels=in_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialized {args.model_type} Classifier. Total Params: {num_params:,}")
    print(f"Initialized {args.model_type} Classifier. Total Params: {num_params:,}")
    
    current_checkpoint = args.initial_checkpoint
    
    # --- PHASE 1: DAPT ---
    if args.mode in ["full", "dapt"]:
        logger.info(f"Setting up DAPT Datasets from: {args.dapt_dataset}")
        train_loader, val_loader, test_loader = create_dataloaders(args, "lung_pet_ct_dx", args.dapt_dataset, depth_size=args.depth_size)
        
        best_dapt_ckpt = run_training_phase(
            model, train_loader, val_loader, device, 
            args.dapt_epochs, args.dapt_lr, args.weight_decay, args.checkpoint_dir, logger, 
            "dapt", scaler=scaler,
            use_segmentation=args.anno, 
            accumulation_steps=args.accumulation_steps,
            model_type=args.model_type
        )
        current_checkpoint = best_dapt_ckpt
        
    # --- PHASE 2: FINETUNE ---
    if args.mode in ["full", "finetune"]:
        if args.mode == "finetune" and args.pretrained_checkpoint:
            model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=device))
            logger.info(f"Loaded pretrained checkpoint from: {args.pretrained_checkpoint}")
        elif current_checkpoint:
            model.load_state_dict(torch.load(current_checkpoint, map_location=device))
            logger.info(f"Loaded DAPT checkpoint for fine-tuning.")
            
        logger.info(f"Setting up FineTuning Datasets from: {args.finetune_dataset}")
        train_loader, val_loader, test_loader = create_dataloaders(args, "big_lunge", args.finetune_dataset, args.finetune_csv)
        
        best_finetune_ckpt = run_training_phase(
            model, train_loader, val_loader, device, 
            args.finetune_epochs, args.finetune_lr, args.weight_decay, args.checkpoint_dir, logger, 
            "finetune", scaler=scaler,
            use_segmentation=args.anno, 
            accumulation_steps=args.accumulation_steps,
            model_type=args.model_type
        )
        
    # --- PHASE 3: INFERENCE ---
    if args.mode in ["full", "inference"]:
        logger.info(f"\n{'='*60}\nStarting Inference Phase\n{'='*60}")
        
        if args.mode == "inference":
            if args.model_checkpoint:
                model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
                logger.info(f"Loaded model checkpoint from: {args.model_checkpoint}")
            else:
                logger.info("No --model-checkpoint provided for inference mode. Using initial weights.")
            
            logger.info("Setting up Test Datasets...")
            if 'test_loader' not in locals() or test_loader is None:
                _, _, test_loader = create_dataloaders(args, "big_lunge", args.finetune_dataset, args.finetune_csv)
            
        elif args.mode == "full":
            if 'best_finetune_ckpt' in locals() and best_finetune_ckpt:
                model.load_state_dict(torch.load(best_finetune_ckpt, map_location=device))
                logger.info("Loaded best FineTune checkpoint for final inference.")
            elif 'best_dapt_ckpt' in locals() and best_dapt_ckpt:
                model.load_state_dict(torch.load(best_dapt_ckpt, map_location=device))
                logger.info("Loaded best DAPT checkpoint for final inference.")
                
        if 'test_loader' in locals():
            logger.info("Running evaluation on the Test Set...")
            test_metrics = validate_epoch(model, test_loader, device, logger)
            logger.info(f"Final Test Set Accuracy: {test_metrics['accuracy']:.4f}")
        else:
            logger.error("Test loader not available. Skipping inference.")
            
    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
