import time
import torch
import torch.nn as nn
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def simple_collate_fn(batch):
    # Just extract the volume and the single target class logic
    scans = torch.stack([item["image"] for item in batch], dim=0)
    
    # We want a 1D tensor of class indices for CrossEntropyLoss
    labels = torch.tensor([item["scan_label"] for item in batch], dtype=torch.long)
    return scans, labels


def train_epoch(model, loader, optimizer, epoch, device, logger, scaler=None):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    for idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(enabled=(scaler is not None), device_type='cuda'):
            logits = model(data)
            loss = criterion(logits, target)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        run_loss.update(loss.item(), n=data.size(0))
        
        # Log every 10 steps and on the last step
        if (idx + 1) % 10 == 0 or (idx + 1) == len(loader):
            msg = (f"Epoch {epoch} [{idx + 1}/{len(loader)}] "
                   f"Loss: {run_loss.avg:.4f} "
                   f"Time: {time.time() - start_time:.2f}s")
            print(msg)
            logger.info(msg)
            start_time = time.time()
            
    return run_loss.avg


@torch.no_grad()
def validate_epoch(model, loader, device, logger):
    model.eval()
    run_loss = AverageMeter()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_targets = []
    
    print("\nStarting validation...")
    logger.info("Starting validation...")

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = criterion(logits, target)
        
        run_loss.update(loss.item(), n=data.size(0))
        
        # Calculate Classification Accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    accuracy = correct / total if total > 0 else 0.0
    val_msg = f"Validation Complete => Loss: {run_loss.avg:.4f}, Accuracy: {accuracy:.4f}"
    print(val_msg)
    logger.info(val_msg)
    
    # Generate and print confusion matrix
    if len(all_targets) > 0:
        names = ["Adenocarcinoma", "Small Cell", "Squamous"]
        
        num_classes = max(max(all_targets), max(all_preds)) + 1
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_targets, all_preds):
            conf_matrix[t, p] += 1
            
        print("\nConfusion Matrix (X: Actual Class, Y: Predicted Class):")
        logger.info("\nConfusion Matrix (X: Actual Class, Y: Predicted Class):")
        
        # Format names safely based on indices 
        display_names = [names[i] if i < len(names) else f"Class{i}" for i in range(num_classes)]
        
        # Header row 
        header = f"{'Guess':<18}" + "".join([f"{display_names[i]:<16}" for i in range(num_classes)])
        print(header)
        logger.info(header)
        
        # Rows
        for j in range(num_classes):
            row_str = f"Y:{display_names[j]:<16}"
            for i in range(num_classes):
                count = conf_matrix[i, j]
                row_str += f"{count:<18}"
            print(row_str)
            logger.info(row_str)
        print("\n")
        print(f"Accuracy: {accuracy:.4f}")
    
    # Return metrics as a dict (matching what main.py expects)
    return {"loss": run_loss.avg, "accuracy": accuracy}
