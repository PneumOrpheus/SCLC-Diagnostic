import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import argparse
import time

from model_selection2 import get_sclc_model
from training.train2 import simple_collate_fn, train_epoch, validate_epoch
from data.data_loader import create_dataset
from logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description="SCLC Simplified 3D Classification Pipeline")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="full", choices=["full", "dapt", "finetune", "inference"],
                        help="Pipeline mode")
    
    # Datasets
    parser.add_argument("--dapt-dataset", type=str, default="/home/data/Lung-PET-CT-Dx")
    parser.add_argument("--annotation-dir", type=str, default="/home/data/Annotation_ZMapped")
    parser.add_argument("--finetune-dataset", type=str, default="/home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1")
    parser.add_argument("--finetune-csv", type=str, default="/home/data/BigLunge/patients_parameters.csv")
    
    # Checkpoints
    parser.add_argument("--initial-checkpoint", type=str, default="/home/data/temp/model_swin_unetr_btcv_segmentation_v1.pt")
    parser.add_argument("--pretrained-checkpoint", type=str, default="")
    parser.add_argument("--model-checkpoint", type=str, default="")
    
    # Hyperparameters
    parser.add_argument("--dapt-epochs", type=int, default=20)
    parser.add_argument("--dapt-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--checkpoint-dir", type=str, default="trained_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--testing", default=False, action="store_true", help="Run with a tiny subset for testing")

    return parser.parse_args()


def create_dataloaders(args, dataset_type, data_path, csv_path=""):
    train_ds, val_ds, test_ds = create_dataset(
        dataset_type=dataset_type,
        data_path=data_path,
        csv_path=csv_path,
        img_size=224,
        depth_size=64,
        convert_to_rgb=False,
        use_multichannel_windowing=False,
        num_workers=args.num_workers,
        annotation_dir=args.annotation_dir if dataset_type == "lung_pet_ct" else "",
        use_3d=True,
        testing=args.testing,
        warm_cache=False,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=simple_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=simple_collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=simple_collate_fn, num_workers=args.num_workers)
    
    return train_loader, val_loader, test_loader


def run_training_phase(phase_name, model, train_loader, val_loader, device, epochs, lr, weight_decay, checkpoint_dir, logger, scaler):
    logger.info(f"\n{'='*60}\nStarting {phase_name}\n{'='*60}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_ckpt = ""
    
    for epoch in range(1, epochs + 1):
        logger.info(f"--- {phase_name} Epoch {epoch}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.2e}) ---")
        
        train_epoch(model, train_loader, optimizer, epoch, device, logger, scaler)
        val_metrics = validate_epoch(model, val_loader, device, logger)
        
        scheduler.step()
        
        acc = val_metrics["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_ckpt = os.path.join(checkpoint_dir, f"best_{phase_name.lower().replace(' ', '_')}.pth")
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"[*] New best model saved! Accuracy: {best_acc:.4f}")
            
    return best_ckpt


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=not args.disable_amp and device.type == "cuda")
    
    logger = create_logger(output_dir=args.output_dir, dist_rank=-1, name="sclc_swinunetr")
    logger.info(f"Running SwinUNETR 3D Classification Pipeline")
    logger.info(f"Mode: {args.mode} | Testing: {args.testing} | Device: {device} | AMP: {not args.disable_amp}")
    
    model = get_sclc_model(args.initial_checkpoint).to(device)
    logger.info(f"Initialized SwinUNETR Classifier. Total Params: {sum(p.numel() for p in model.parameters()):,}")
    
    current_checkpoint = args.initial_checkpoint
    
    # --- PHASE 1: DAPT ---
    if args.mode in ["full", "dapt"]:
        logger.info(f"Setting up DAPT Datasets from: {args.dapt_dataset}")
        train_loader, val_loader, test_loader = create_dataloaders(args, "lung_pet_ct_dx", args.dapt_dataset)
        
        best_dapt_ckpt = run_training_phase(
            "DAPT Phase", model, train_loader, val_loader, device, 
            args.dapt_epochs, args.dapt_lr, args.weight_decay, args.checkpoint_dir, logger, scaler
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
            "FineTune Phase", model, train_loader, val_loader, device, 
            args.finetune_epochs, args.finetune_lr, args.weight_decay, args.checkpoint_dir, logger, scaler
        )
        
    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
