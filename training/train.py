import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from models.model_selection import get_sclc_model

"""
SCLC Diagnostic System Training
===============================
Implements the training pipeline for the SCLC diagnostic system with support for:
- Backbone model selection
- Resuming from checkpoints for fine-tuning
- Multi-task loss aggregation
"""

def detection_collate_fn(batch):
    """Custom collate function for DataLoader that unzips (scan, target) pairs."""
    return tuple(zip(*batch))

# Backward compatibility alias; prefer using `detection_collate_fn` directly.
batch_fn = detection_collate_fn
class SCLCTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Initialize dataset
        self.data_path = data_path

        # Validate that the data path exists and is a directory
        if not os.path.isdir(self.data_path):
            raise ValueError(f"Data path '{self.data_path}' does not exist or is not a directory.")

        try:
            all_files = os.listdir(self.data_path)
        except OSError as e:
            raise ValueError(f"Unable to list contents of data path '{self.data_path}': {e}") from e

        # Collect all valid samples
        self.samples = [file for file in all_files if file.endswith('.nii.gz')]

        if not self.samples:
            raise ValueError(
                f"No '.nii.gz' files found in data path '{self.data_path}'. "
                "Please provide a directory containing valid data files."
            )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.samples[idx])
        data_dict = np.load(path, allow_pickle=True).item()
        
        scan = torch.tensor(data_dict['scan'], dtype=torch.float32)
        targets = {
            'boxes': torch.tensor(data_dict['boxes'], dtype=torch.float32),
            'labels': torch.tensor(data_dict['labels'], dtype=torch.int64),
            'scan_label': torch.tensor(data_dict['scan_label'], dtype=torch.int64),

        }
        return scan, targets       


def train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    
    for i, (scans, targets) in enumerate(data_loader):
        scans = list(scan.to(device) for scan in scans)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # forward pass
        loss_dict = model(scans, targets)
        
        # loss aggregation
        global_loss = loss_dict.pop("global_classification_loss")

        # ensure detection loss is always a tensor on the correct device,
        # and handle the case where there are no detection losses explicitly
        if loss_dict:
            detection_losses = []
            for loss in loss_dict.values():
                if isinstance(loss, torch.Tensor):
                    detection_losses.append(loss.to(device))
                else:
                    detection_losses.append(torch.as_tensor(loss, device=device))
            loss_detection = sum(detection_losses)
        else:
            # no detection losses; use a zero scalar tensor on the target device
            loss_detection = torch.zeros((), device=device)
        
        # weighted sum
        total_loss = loss_detection + 0.5 * global_loss
        
        # backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Iteration [{i}/{len(data_loader)}], "
                  f"Total loss: {total_loss.item():.4f}, "
                  f"Detection loss: {loss_detection.item():.4f}, "
                  f"Global loss: {global_loss.item():.4f}")
            
def main(args):
    parser = argparse.ArgumentParser(description="SCLC Diagnostic System Training")
    parser.add_argument("--backbone", type=str, default="swinv2", choices=["swinv2", "resnet50", "densenet121"], help="Which backbone model to use")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the SCLC training data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth model file from which to resume checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    
    args = parser.parse_args(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.backbone}")
    
    train_dataset = SCLCTrainDataset(args.data_path)
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=batch_fn,
        pin_memory=(device.type == "cuda"),
    )
    
    # Model initialization
    model = get_sclc_model(backbone_type=args.backbone, checkpoint_path=args.checkpoint)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    for epoch in range(args.epochs):
        train_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            # Save model weights only (backward-compatible with existing usage)
            checkpoint_path = f"sclc_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint (weights only): {checkpoint_path}")

            # Save full training state for proper resume
            full_checkpoint_path = f"sclc_full_checkpoint_epoch_{epoch+1}.pth"
            full_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            }
            torch.save(full_checkpoint, full_checkpoint_path)
            print(f"Saved full training checkpoint: {full_checkpoint_path}")
if __name__ == "__main__":
    main(sys.argv[1:])