import os
import sys
import shutil
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Any, Tuple
import argparse
import time
from datetime import datetime
from collections import Counter, deque

from model_selection import get_sclc_model
from training.train import simple_collate_fn, train_epoch, validate_epoch
from data.data_loader import create_dataset
from logger import create_logger


def _save_inference_probabilities(output_dir: str, model_type: str, payload: Dict[str, Any], logger) -> str:
    """Persist inference softmax probabilities to disk for post-hoc analysis."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(output_dir, f"{model_type}_{timestamp}_inference_probabilities.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    mean_probs = payload.get("mean_probability_per_class", {})
    if mean_probs:
        msg = ", ".join([f"mean P({k})={float(v):.4f}" for k, v in mean_probs.items()])
        logger.info(f"Inference probability means: {msg}")
    logger.info(f"Saved inference probabilities to: {out_path}")
    return out_path

def parse_args():
    parser = argparse.ArgumentParser(description="SCLC Simplified 3D Classification Pipeline")
    
    # Mode selection
    parser.add_argument("--model-type", type=str, default="swin_unetr", choices=["swin_unetr", "resnet50", "resnet18", "densenet121", "models_genesis"],
                        help="Model architecture to use")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "dapt", "finetune", "inference"],
                        help="Pipeline mode")
    
    # Datasets
    parser.add_argument("--dapt-dataset", type=str, default="/home/data/Lung-PET-CT-Dx-Clean")
    parser.add_argument("--pet-dir", type=str, default="/home/data/Lung-PET-CT-Dx_PET", help="Directory containing the PET NIfTI files")
    parser.add_argument("--use-pet", action="store_true", default=False, help="Include PET as the second input channel")
    parser.add_argument("--finetune-dataset", type=str, default="/home/data/TrainingData")
    parser.add_argument("--finetune-csv", type=str, default="/home/data/TrainingData/patients_parameters.csv")
    
    # Checkpoints
    parser.add_argument("--initial-checkpoint", type=str, default="", help="Path to initial checkpoint. Defaults to Swin UNETR BTCV if model-type is swin_unetr")
    parser.add_argument("--pretrained-checkpoint", type=str, default="")
    parser.add_argument("--model-checkpoint", type=str, default="")
    
    # Hyperparameters
    parser.add_argument("--dapt-epochs", type=int, default=30)
    parser.add_argument("--dapt-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument("--finetune-backbone-lr-scale", type=float, default=0.1,
                        help="Backbone LR multiplier for fine-tune differential LR (backbone_lr = finetune_lr * scale).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps to increase effective batch size")
    parser.add_argument("--seg-loss-weight", type=float, default=0.1,
                        help="Weight for auxiliary segmentation loss when --anno is enabled and masks exist.")
    parser.add_argument("--weight-decay", type=float, default=1e-3) # Fine-tune weight decay
    parser.add_argument("--dapt-weight-decay", type=float, default=3e-3,
                        help="Weight decay for DAPT. Higher than fine-tune to combat scan-diversity overfitting. "
                             "Round 3 used 1e-2 and under-fit; Round 4 dials back to 3e-3.")
    parser.add_argument(
        "--monitor-rolling-window",
        type=int,
        default=3,
        help="Use rolling mean over last k validation epochs for checkpoint selection and early stopping. 1 disables smoothing.",
    )
    
    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--checkpoint-dir", type=str, default="/home/data/trained_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--testing", default=False, action="store_true", help="Run with a tiny subset for testing")
    parser.add_argument("--clear-cache", default=False, action="store_true",
                        help="Delete the MONAI PersistentDataset cache before building datasets")
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

    # Sanity: every class must be present in the training split.
    # This catches the "stale --testing cache froze training on 12 non-SCLC samples" bug loudly.
    if hasattr(train_ds, "data") and len(train_ds.data) > 0 and "scan_label" in train_ds.data[0]:
        train_counts = Counter(int(item["scan_label"]) for item in train_ds.data)
        missing = [c for c in range(3) if train_counts.get(c, 0) == 0]
        if missing:
            raise RuntimeError(
                f"[{dataset_type}] Training split is missing classes {missing}. "
                f"Counts: {dict(train_counts)}. "
                f"Refusing to train — clear the cache (--clear-cache) and check the data directory."
            )
        print(f"[{dataset_type}] Train class distribution: {dict(train_counts)}")

    return train_loader, val_loader, test_loader


_HEAD_PREFIXES = ("classification_head", "dense_1", "dense_2")


def _set_backbone_frozen(model, frozen: bool, logger=None) -> int:
    """Freeze/unfreeze every parameter that is NOT part of a classification head.

    Returns the number of parameters affected (backbone params). The head is
    identified by attribute-name prefix — see _HEAD_PREFIXES. DenseNet121's
    head lives inside densenet.class_layers, so it is also caught here.
    """
    n_backbone = 0
    n_head = 0
    for name, param in model.named_parameters():
        is_head = (
            name.startswith(_HEAD_PREFIXES)
            or "class_layers" in name  # DenseNet121 classifier
        )
        if is_head:
            param.requires_grad = True
            n_head += 1
        else:
            param.requires_grad = not frozen
            n_backbone += 1
    if logger is not None:
        state = "FROZEN" if frozen else "TRAINABLE"
        logger.info(f"[freeze] backbone {state}: {n_backbone} backbone params, {n_head} head params")
    return n_backbone


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int, warmup_start_lr: float, base_lr: float):
    """Cosine schedule with an optional linear warmup prepended.

    LinearLR uses start_factor=warmup_start_lr/base_lr so the effective LR
    ramps from warmup_start_lr on epoch 1 to base_lr at the end of warmup.
    After that, CosineAnnealingLR runs over the remaining epochs.
    """
    if warmup_epochs <= 0 or warmup_epochs >= epochs:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_factor = max(warmup_start_lr / base_lr, 1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def run_training_phase(
    model, train_loader, val_loader, device, epochs, lr, weight_decay,
    checkpoint_dir, logger, phase_name, patience=10, scaler=None, use_segmentation=False, accumulation_steps=4,
    model_type="swin_unetr",
    warmup_epochs: int = 0, warmup_start_lr: float = 5e-6, freeze_backbone_epochs: int = 0,
    monitor_window: int = 3,
    differential_lr: bool = False,
    backbone_lr_scale: float = 0.1,
    seg_loss_weight: float = 0.1,
):
    seg_loss_weight = max(0.0, float(seg_loss_weight))

    # Differential LR for fine-tune stability:
    # - backbone updates are gentle (pretrained features)
    # - head updates are faster (fresh classifier layers)
    diff_lr_active = bool(differential_lr)
    if diff_lr_active:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith(_HEAD_PREFIXES) or ("class_layers" in name)
            if is_head:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if len(backbone_params) == 0 or len(head_params) == 0:
            logger.warning(
                f"[{phase_name}] differential_lr requested but parameter split failed "
                f"(backbone={len(backbone_params)}, head={len(head_params)}). Falling back to single LR."
            )
            diff_lr_active = False
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            backbone_lr = lr * max(backbone_lr_scale, 1e-6)
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
                    {"params": head_params, "lr": lr, "weight_decay": weight_decay},
                ]
            )
            logger.info(
                f"[{phase_name}] Differential LR enabled: backbone_lr={backbone_lr:.2e}, "
                f"head_lr={lr:.2e}, backbone_tensors={len(backbone_params)}, head_tensors={len(head_params)}"
            )
    else:
        # All params go into the optimizer up front. Frozen params have
        # requires_grad=False, so AdamW skips their step entirely (no weight
        # decay leak), and unfreezing is a pure requires_grad flip — no optimizer
        # rebuild, no state loss on the head.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if diff_lr_active:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if warmup_epochs > 0 or freeze_backbone_epochs > 0:
            logger.info(
                f"[{phase_name}] Ignoring warmup/freeze settings because differential_lr=True "
                f"(warmup_epochs={warmup_epochs}, freeze_backbone_epochs={freeze_backbone_epochs})."
            )
    else:
        scheduler = _build_scheduler(optimizer, epochs, warmup_epochs, warmup_start_lr, lr)

    if freeze_backbone_epochs > 0 and not diff_lr_active:
        n_backbone = _set_backbone_frozen(model, frozen=True, logger=logger)
        if n_backbone == 0:
            logger.warning(
                f"[{phase_name}] freeze_backbone_epochs={freeze_backbone_epochs} requested but "
                f"no backbone parameters were identified — check _HEAD_PREFIXES for this model type."
            )

    monitor_window = max(1, int(monitor_window))
    phase_prefix = phase_name.lower().replace(' ', '_').replace('_phase', '')
    stamp_day_month = datetime.now().strftime("%d_%m")
    # Keep phase in the model tag so DAPT/finetune checkpoints do not overwrite each other.
    model_tag = f"{model_type}_{phase_prefix}"

    rolling_history = {
        "loss": deque(maxlen=monitor_window),
        "accuracy": deque(maxlen=monitor_window),
        "balanced_accuracy": deque(maxlen=monitor_window),
        "macro_precision": deque(maxlen=monitor_window),
        "macro_recall": deque(maxlen=monitor_window),
        "macro_f1": deque(maxlen=monitor_window),
    }

    best_monitor_macro_f1 = -1.0  # first validated epoch always eligible to save
    best_ckpt = None
    epochs_no_improve = 0

    if use_segmentation:
        logger.info(f"[{phase_name}] Segmentation auxiliary loss enabled with seg_loss_weight={seg_loss_weight:.3f}")

    for epoch in range(1, epochs + 1):
        # Unfreeze exactly once, at the boundary.
        if freeze_backbone_epochs > 0 and not diff_lr_active and epoch == freeze_backbone_epochs + 1:
            _set_backbone_frozen(model, frozen=False, logger=logger)
            logger.info(f"[{phase_name}] Backbone unfrozen at epoch {epoch}.")

        if diff_lr_active and len(optimizer.param_groups) >= 2:
            backbone_lr = optimizer.param_groups[0]["lr"]
            head_lr = optimizer.param_groups[1]["lr"]
            lr_msg = f"[{phase_name}] Epoch {epoch}/{epochs} | lr(backbone/head)={backbone_lr:.2e}/{head_lr:.2e}"
            logger.info(lr_msg)
            print(f"\n--- {phase_name} Epoch {epoch}/{epochs} | lr(backbone/head)={backbone_lr:.2e}/{head_lr:.2e} ---")
        else:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"[{phase_name}] Epoch {epoch}/{epochs} | lr={current_lr:.2e}")
            print(f"\n--- {phase_name} Epoch {epoch}/{epochs} | lr={current_lr:.2e} ---")
        
        train_loss, train_macro_f1 = train_epoch(
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            epoch=epoch, 
            device=device, 
            logger=logger, 
            scaler=scaler,
            use_segmentation=use_segmentation,
            accumulation_steps=accumulation_steps, # Pass down here
            seg_loss_weight=seg_loss_weight,
        )
        val_metrics = validate_epoch(model, val_loader, device, logger)

        for key in rolling_history:
            rolling_history[key].append(float(val_metrics[key]))
        rolling_metrics = {k: float(np.mean(v)) for k, v in rolling_history.items()}

        raw_macro_f1 = float(val_metrics["macro_f1"])
        rolling_macro_f1 = float(rolling_metrics["macro_f1"])

        train_val_msg = (
            f"[{phase_name}] Epoch {epoch} Summary => "
            f"TrainLoss: {train_loss:.4f}, TrainMacroF1: {train_macro_f1:.4f}, "
            f"ValMacroF1: {raw_macro_f1:.4f}/{rolling_macro_f1:.4f} (cur/roll{monitor_window})"
        )
        print(train_val_msg)
        logger.info(train_val_msg)

        rolling_msg = (
            f"[{phase_name}] Val current vs rolling-{monitor_window}: "
            f"accuracy {val_metrics['accuracy']:.4f}/{rolling_metrics['accuracy']:.4f}, "
            f"balanced_accuracy {val_metrics['balanced_accuracy']:.4f}/{rolling_metrics['balanced_accuracy']:.4f}, "
            f"macro_precision {val_metrics['macro_precision']:.4f}/{rolling_metrics['macro_precision']:.4f}, "
            f"macro_recall {val_metrics['macro_recall']:.4f}/{rolling_metrics['macro_recall']:.4f}, "
            f"macro_f1 {raw_macro_f1:.4f}/{rolling_macro_f1:.4f}"
        )
        print(rolling_msg)
        logger.info(rolling_msg)
        
        scheduler.step()

        if rolling_macro_f1 > best_monitor_macro_f1:
            best_monitor_macro_f1 = rolling_macro_f1
            best_ckpt = os.path.join(checkpoint_dir, f"{stamp_day_month}_{model_tag}_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            logger.info(
                f"[*] New best model saved! RollingMacroF1({monitor_window}): {best_monitor_macro_f1:.4f} "
                f"| CurrentMacroF1: {raw_macro_f1:.4f}"
            )
            epochs_no_improve = 0  # reset counter
        else:
            epochs_no_improve += 1
            
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_ckpt = os.path.join(
                checkpoint_dir,
                f"{stamp_day_month}_{model_tag}_epoch_{epoch}.pth",
            )
            torch.save(model.state_dict(), periodic_ckpt)
            logger.info(f"[*] Periodic checkpoint saved at epoch {epoch}: {periodic_ckpt}")
            
        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping triggered. No rolling macro-F1 improvement for {patience} epochs "
                f"(window={monitor_window})."
            )
            break

    if best_ckpt is None:
        logger.warning(f"[{phase_name}] No checkpoint saved — val macro-F1 never improved.")
    return best_ckpt


def main():
    args = parse_args()
    
    if not args.initial_checkpoint and args.model_type == "swin_unetr":
        args.initial_checkpoint = "/home/data/pre_trained_models/model_swin_unetr_btcv_segmentation_v1.pt"
        
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.clear_cache:
        cache_root = os.path.join(os.path.expanduser("~"), ".cache")
        for name in ("monai_lung_pet_ct_clean", "monai_biglunge"):
            path = os.path.join(cache_root, name)
            if os.path.isdir(path):
                print(f"[--clear-cache] Removing {path}")
                shutil.rmtree(path)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=not args.disable_amp and device.type == "cuda")
    
    logger = create_logger(output_dir=args.output_dir, dist_rank=-1, name=f"{args.model_type}")
    logger.info(f"Running {args.model_type} 3D Classification Pipeline")
    logger.info(f"Mode: {args.mode} | Testing: {args.testing} | Device: {device} | AMP: {not args.disable_amp}")
    
    # Segmentation aux loss only flows gradient through SwinUNETR — the other
    # wrappers return a zero-tensor seg head with no graph connection, so the
    # --anno flag would silently do nothing. Warn loudly.
    if args.anno and args.model_type != "swin_unetr":
        logger.warning(
            f"--anno is set but model_type={args.model_type} has no real segmentation "
            f"decoder; mask supervision will have NO effect on training. Either switch "
            f"to swin_unetr or drop --anno."
        )

    in_channels = 2 if getattr(args, "use_pet", False) else 1
    model = get_sclc_model(args.initial_checkpoint, model_type=args.model_type, in_channels=in_channels, depth_size=args.depth_size).to(device)
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
            args.dapt_epochs, args.dapt_lr, args.dapt_weight_decay, args.checkpoint_dir, logger,
            "dapt", scaler=scaler,
            use_segmentation=args.anno, 
            accumulation_steps=args.accumulation_steps,
            model_type=args.model_type,
            monitor_window=args.monitor_rolling_window,
            seg_loss_weight=args.seg_loss_weight,
        )
        current_checkpoint = best_dapt_ckpt
        
    # --- PHASE 2: FINETUNE ---
    if args.mode in ["full", "finetune"]:
        if args.mode == "finetune" and args.pretrained_checkpoint:
            model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=device))
            logger.info(f"Loaded pretrained checkpoint from: {args.pretrained_checkpoint}")
        elif current_checkpoint and current_checkpoint != args.initial_checkpoint and os.path.isfile(current_checkpoint):
            # DAPT produced a checkpoint in full mode — load it.
            model.load_state_dict(torch.load(current_checkpoint, map_location=device))
            logger.info(f"Loaded DAPT checkpoint for fine-tuning: {current_checkpoint}")
        else:
            # No DAPT checkpoint — fine-tune from in-memory weights.
            # In --mode finetune without --pretrained-checkpoint this means the
            # model still has the initial backbone weights (e.g. BTCV) loaded by
            # get_sclc_model(), which is a valid starting point.
            logger.info("Fine-tuning from initial in-memory weights (no DAPT checkpoint).")
            
        logger.info(f"Setting up FineTuning Datasets from: {args.finetune_dataset}")
        train_loader, val_loader, test_loader = create_dataloaders(args, "big_lunge", args.finetune_dataset, args.finetune_csv, depth_size=args.depth_size)
        
        best_finetune_ckpt = run_training_phase(
            model, train_loader, val_loader, device,
            args.finetune_epochs, args.finetune_lr, args.weight_decay, args.checkpoint_dir, logger,
            "finetune", scaler=scaler,
            use_segmentation=args.anno,
            accumulation_steps=args.accumulation_steps,
            model_type=args.model_type,
            monitor_window=args.monitor_rolling_window,
            differential_lr=True,
            backbone_lr_scale=args.finetune_backbone_lr_scale,
            seg_loss_weight=args.seg_loss_weight,
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
                _, _, test_loader = create_dataloaders(args, "big_lunge", args.finetune_dataset, args.finetune_csv, depth_size=args.depth_size)
            
        elif args.mode == "full":
            if 'best_finetune_ckpt' in locals() and best_finetune_ckpt:
                model.load_state_dict(torch.load(best_finetune_ckpt, map_location=device))
                logger.info("Loaded best FineTune checkpoint for final inference.")
            elif 'best_dapt_ckpt' in locals() and best_dapt_ckpt:
                model.load_state_dict(torch.load(best_dapt_ckpt, map_location=device))
                logger.info("Loaded best DAPT checkpoint for final inference.")
                
        if 'test_loader' in locals():
            logger.info("Running evaluation on the Test Set...")
            test_metrics = validate_epoch(model, test_loader, device, logger, return_probabilities=True)
            logger.info(f"Final Test Set Accuracy: {test_metrics['accuracy']:.4f}")

            prob_payload = test_metrics.get("inference_probabilities")
            if prob_payload is not None:
                probs_file = _save_inference_probabilities(args.output_dir, args.model_type, prob_payload, logger)
                print(f"Saved inference probabilities to: {probs_file}")
        else:
            logger.error("Test loader not available. Skipping inference.")
            
    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
