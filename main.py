import sys
import torch
import torch.optim as optim
from monai.data import DataLoader 
import numpy as np
import os
import argparse
from models.model_selection import get_sclc_model
from training.train import create_train_dataset, create_val_dataset, create_test_dataset, sclc_collate_fn, train_epoch


from logger import create_logger
from models.config import get_config


def parse_options():
    parser = argparse.ArgumentParser(description="SCLC Diagnostic System Training")
    parser.add_argument("--backbone", type=str, default="swinv2", choices=["swin", "swinv2", "resnet50", "densenet121"], help="Which backbone model to use")
    parser.add_argument("--data-path", type=str,default="/home/data/Lung-PET-CT-Dx", help="Path to the SCLC training data")
    parser.add_argument("--checkpoint", type=str, default="/home/hansstem/RadImageNet_swin/rin_swintf.pth", help="Path to .pth model file from which to resume checkpoint")
    parser.add_argument("--config", type=str, default="/home/hansstem/RadImageNet_swin/rin_config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--train-backbone-only", action="store_true", help="Train only the backbone (freeze FPN and heads)")
    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCLC Diagnostic System Training")
    args = parse_options()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.backbone}")

    # Use RGB conversion only for ImageNet-pretrained timm models
    uses_timm_model = not (args.checkpoint and args.config)
    
    train_dataset = create_train_dataset(
        data_path=args.data_path,
        convert_to_rgb=uses_timm_model,
        cache_rate=1.0,
        num_workers=4,
    )
    
    val_dataset = create_val_dataset(
        data_path=args.data_path,
        convert_to_rgb=uses_timm_model,
        cache_rate=1.0,
        num_workers=4,
    )

    test_dataset = create_test_dataset(
        data_path=args.data_path,
        convert_to_rgb=uses_timm_model,
        cache_rate=0.0,  # Typically don't need to cache test data during training
        num_workers=4,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=sclc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=sclc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=sclc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    config = get_config(args)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=-1, name=f"{config.MODEL.NAME}")
    logger.info(f"Using backbone: {args.backbone}")
    if args.train_backbone_only:
        logger.info("Training mode: backbone-only (FPN and heads frozen)")

    model = get_sclc_model(
        backbone_type=args.backbone, 
        checkpoint_path=args.checkpoint, 
        config=config, 
        train_backbone_only=args.train_backbone_only,
        logger=logger
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create checkpoint directories
    os.makedirs("checkpoint_weights", exist_ok=True)
    os.makedirs("full_checkpoints", exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        train_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            # Save model weights only
            checkpoint_path = f"checkpoint_weights/sclc_{args.backbone}_model_weights_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint (weights only): {checkpoint_path}")

            # Save full training state for proper resume
            full_checkpoint_path = f"full_checkpoints/sclc_{args.backbone}_full_checkpoint_epoch_{epoch+1}.pth"
            full_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            }
            torch.save(full_checkpoint, full_checkpoint_path)
            logger.info(f"Saved full training checkpoint: {full_checkpoint_path}")

