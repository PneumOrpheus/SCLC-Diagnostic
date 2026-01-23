import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
# from model_selection import get_model

"""
SCLC Diagnostic System Training
===============================
Implements the training pipeline for the SCLC diagnostic system with support for:
- Backbone model selection
- Resuming from checkpoints for fine-tuning
- Multi-task loss aggregation
"""

def batch_fn(batch):
    return tuple(zip(*batch))

class SCLCTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Initialize dataset
        self.data_path = data_path
        self.samples = [file for file in os.listdir(data_path) if file.endswith('.nii.gz')]
    
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
            'scan_id': torch.tensor(data_dict['scan_id'], dtype=torch.int64)
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
        global_loss = loss_dict.pop("global_loss_classifier")
        loss_detection = sum(loss for loss in loss_dict.values())
        if isinstance(loss_detection, int):
             loss_detection = torch.tensor(loss_detection).to(device)
        
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