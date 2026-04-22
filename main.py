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

try:
    import yaml
except ImportError:  # PyYAML is only needed when --config is used
    yaml = None

from model_selection import get_sclc_model, get_pipeline
from training.train import simple_collate_fn, train_epoch, validate_epoch
from training.train_2d import simple_collate_fn_2d, train_epoch_2d, validate_epoch_2d
from data.data_loader import create_dataset, create_dataset_2p5d
from data.dataset_2d import create_dataset_2d
from logger import create_logger


def _dump_effective_config(output_dir: str, args: argparse.Namespace, logger) -> str:
    """Persist the fully-resolved argparse namespace (config + CLI merged) to
    ``output_dir/effective_config.yaml``. Written at startup so even a crashed
    run leaves behind a reproducible record of what it was about to do.
    """
    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    cfg["_meta"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_file": getattr(args, "config", "") or None,
        "cwd": os.getcwd(),
    }
    out_path = os.path.join(output_dir, "effective_config.yaml")
    try:
        if yaml is not None:
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
        else:
            out_path = os.path.splitext(out_path)[0] + ".json"
            with open(out_path, "w") as f:
                json.dump(cfg, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to write effective config: {e}")
        return ""
    logger.info(f"Effective config written to: {out_path}")
    return out_path


def _append_metrics_row(metrics_path: str, row: Dict[str, Any]) -> None:
    """Append one JSON object per line to ``metrics_path``. Flushing is
    best-effort — a mid-run crash still leaves completed epochs on disk.
    """
    if not metrics_path:
        return
    try:
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        # Metrics logging must never take down training.
        pass


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

def _flatten_config(cfg: Any, out: Dict[str, Any] = None) -> Dict[str, Any]:
    """Flatten a nested YAML config into {dest_name: value}.

    Nested sections are walked for their leaves; section names themselves are
    discarded (grouping is cosmetic). Hyphenated keys are converted to
    underscores so they match argparse ``dest`` names.
    """
    if out is None:
        out = {}
    if not isinstance(cfg, dict):
        return out
    for k, v in cfg.items():
        if isinstance(v, dict):
            _flatten_config(v, out)
        else:
            out[str(k).replace("-", "_")] = v
    return out


def _apply_config_to_parser(parser: argparse.ArgumentParser, config_path: str) -> Dict[str, Any]:
    """Load YAML at ``config_path`` and push its leaves as argparse defaults.

    Returns the flat dict that was applied so main() can log what came from
    the config file. Keys that don't match any argparse dest are logged as a
    warning (typos surface loud instead of silently doing nothing).
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required to use --config. Install with: pip install pyyaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"--config path does not exist: {config_path}")
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    flat = _flatten_config(raw)
    known_dests = {a.dest for a in parser._actions}
    unknown = sorted(k for k in flat if k not in known_dests and k != "name" and k != "notes")
    if unknown:
        print(f"[config] WARNING: ignoring unknown keys in {config_path}: {unknown}", file=sys.stderr)
    applied = {k: v for k, v in flat.items() if k in known_dests}
    parser.set_defaults(**applied)
    return applied


def parse_args():
    parser = argparse.ArgumentParser(description="SCLC Simplified 3D Classification Pipeline")

    # Config file. Loaded before the main parse so CLI flags still override.
    parser.add_argument("--config", type=str, default="",
                        help="Path to a YAML experiment config. Values are applied as argparse defaults; "
                             "any CLI flag given on the command line overrides the config.")

    # Mode selection
    parser.add_argument("--model-type", type=str, default="swin_unetr",
                        choices=["swin_unetr", "resnet50", "resnet18", "densenet121", "models_genesis",
                                 "efficientnet_b0_2p5d", "densenet121_2p5d",
                                 "efficientnet_b0_2d", "densenet121_2d", "resnet50_2d"],
                        help="Model architecture to use. '_2d' uses the per-slice 2D pipeline; "
                             "'_2p5d' uses the 2.5D tumor-cropped pipeline; others use the full 3D pipeline.")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "dapt", "finetune", "inference"],
                        help="Pipeline mode")
    
    # Datasets
    parser.add_argument("--dapt-dataset", type=str, default="/home/data/Lung-PET-CT-Dx-Clean")
    parser.add_argument("--finetune-dataset", type=str, default="/home/data/TrainingData")
    parser.add_argument("--finetune-csv", type=str, default="/home/data/TrainingData/patients_parameters.csv")
    
    # Checkpoints
    parser.add_argument(
        "--initial-checkpoint",
        type=str,
        default="",
        help="Base initialization checkpoint. Defaults to Swin UNETR BTCV when model-type is swin_unetr.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="",
        help="Checkpoint to load in --mode finetune or --mode inference (e.g., best DAPT checkpoint).",
    )
    
    # Hyperparameters
    parser.add_argument("--dapt-epochs", type=int, default=30)
    parser.add_argument("--dapt-lr", type=float, default=1e-4)
    parser.add_argument("--dapt-warmup-epochs", type=int, default=3,
                        help="Linear warmup length for the DAPT phase. 0 disables.")
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument("--finetune-warmup-epochs", type=int, default=2,
                        help="Linear warmup length for the fine-tune phase. 0 disables.")
    parser.add_argument("--warmup-start-lr", type=float, default=1e-6,
                        help="Starting LR for linear warmup (ramps to the phase LR).")
    parser.add_argument("--finetune-backbone-lr-scale", type=float, default=0.1,
                        help="Backbone LR multiplier for fine-tune differential LR (backbone_lr = finetune_lr * scale).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps to increase effective batch size")
    parser.add_argument("--seg-loss-weight", type=float, default=0.1,
                        help="Weight for the auxiliary segmentation loss (only active for SwinUNETR; ignored otherwise).")
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
    parser.add_argument("--depth-size", type=int, default=128, help="Depth size for the 3D images (must be divisible by 32 for SwinUNETR)")

    # 2.5D pipeline knobs
    parser.add_argument("--num-slices", type=int, default=5,
                        help="Number of axial slices stacked as channels for the 2.5D pipeline.")
    parser.add_argument("--img-size-2p5d", type=int, default=96,
                        help="In-plane size for the 2.5D tumor-centered crop.")
    parser.add_argument("--tumor-mask-suffix", type=str, default="_label_tumor.nii.gz",
                        help="Per-patient tumor mask suffix expected under the BigLunge patient folder.")

    # 2D pipeline knobs
    parser.add_argument("--img-size-2d", type=int, default=224,
                        help="In-plane size for the 2D tumor-centered slice crop.")
    parser.add_argument("--max-slices-per-volume", type=int, default=8,
                        help="Cap tumor slices sampled per volume in the 2D pipeline (0 = no cap). "
                             "Each uncapped slice triggers a full volume reload during cache build, so keep this small.")
    parser.add_argument("--cache-workers", type=int, default=4,
                        help="Parallel workers used when building the 2D/2.5D/3D PersistentDataset cache for the first time.")

    # Two-pass parse: peek at --config first, apply it as defaults, then the
    # real parse lets CLI flags override config values. This keeps a single
    # source of truth for argument definitions (here), and makes YAML a thin
    # "set a bunch of defaults" layer.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()
    applied: Dict[str, Any] = {}
    if pre_args.config:
        applied = _apply_config_to_parser(parser, pre_args.config)

    args = parser.parse_args()
    args._config_applied = applied  # stashed so main() can log what came from the file
    return args


def create_dataloaders(args, dataset_type, data_path, csv_path="", depth_size=64):
    pipeline = get_pipeline(args.model_type)
    if pipeline == "2d":
        max_slices = args.max_slices_per_volume if args.max_slices_per_volume and args.max_slices_per_volume > 0 else None
        train_ds, val_ds, test_ds = create_dataset_2d(
            data_path=data_path,
            csv_path=csv_path,
            dataset_type=dataset_type,
            img_size=args.img_size_2d,
            tumor_mask_suffix=args.tumor_mask_suffix,
            max_slices_per_volume=max_slices,
            testing=args.testing,
            cache_workers=args.cache_workers,
        )
        collate_fn = simple_collate_fn_2d
    elif pipeline == "2p5d":
        train_ds, val_ds, test_ds = create_dataset_2p5d(
            data_path=data_path,
            csv_path=csv_path,
            dataset_type=dataset_type,
            img_size=args.img_size_2p5d,
            num_slices=args.num_slices,
            tumor_mask_suffix=args.tumor_mask_suffix,
            testing=args.testing,
        )
        collate_fn = simple_collate_fn
    else:
        train_ds, val_ds, test_ds = create_dataset(
            dataset_type=dataset_type,
            data_path=data_path,
            csv_path=csv_path,
            img_size=224,
            depth_size=depth_size,
            convert_to_rgb=False,
            use_multichannel_windowing=False,
            num_workers=args.num_workers,
            use_3d=True,
            testing=args.testing,
            warm_cache=False,
        )
        collate_fn = simple_collate_fn

    if hasattr(train_ds, "data") and len(train_ds.data) > 0:
        print(f"Applying WeightedRandomSampler for {dataset_type}")
        train_labels = [item["scan_label"] for item in train_ds.data]
        class_counts = Counter(train_labels)
        num_samples = len(train_labels)

        # Invert class frequencies to create weights
        class_weights_dict = {cls: num_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights_dict[label] for label in train_labels]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        # Note: shuffle must be False when using a sampler
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Sanity: every class must be present in the training split.
    # This catches the "stale --testing cache froze training on 12 non-SCLC samples" bug loudly.
    if hasattr(train_ds, "data") and len(train_ds.data) > 0 and "scan_label" in train_ds.data[0]:
        train_counts = Counter(int(item["scan_label"]) for item in train_ds.data)
        missing = [c for c in range(3) if train_counts.get(c, 0) == 0]
        if missing:
            msg = (
                f"[{dataset_type}] Training split is missing classes {missing}. "
                f"Counts: {dict(train_counts)}."
            )
            if args.testing:
                print(f"WARNING: {msg} Proceeding because --testing is enabled.")
            else:
                raise RuntimeError(
                    msg + " Refusing to train — clear the cache (--clear-cache) and check the data directory."
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
    train_fn=train_epoch,
    validate_fn=validate_epoch,
    metrics_path: str = "",
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
    stamp_day_month = datetime.now().strftime("%h_%d_%m")
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
        
        train_loss, train_macro_f1 = train_fn(
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
        val_metrics = validate_fn(model, val_loader, device, logger)

        patient_metrics = val_metrics.get("patient_level")
        has_patient_metrics = isinstance(patient_metrics, dict)
        monitor_source = patient_metrics if has_patient_metrics else val_metrics
        monitor_level = "patient" if has_patient_metrics else "volume"

        for key in rolling_history:
            if key == "loss":
                rolling_history[key].append(float(val_metrics[key]))
            else:
                rolling_history[key].append(float(monitor_source.get(key, val_metrics[key])))
        rolling_metrics = {k: float(np.mean(v)) for k, v in rolling_history.items()}

        raw_accuracy = float(monitor_source.get("accuracy", val_metrics["accuracy"]))
        raw_balanced_accuracy = float(monitor_source.get("balanced_accuracy", val_metrics["balanced_accuracy"]))
        raw_macro_precision = float(monitor_source.get("macro_precision", val_metrics["macro_precision"]))
        raw_macro_recall = float(monitor_source.get("macro_recall", val_metrics["macro_recall"]))
        raw_macro_f1 = float(monitor_source.get("macro_f1", val_metrics["macro_f1"]))
        rolling_macro_f1 = float(rolling_metrics["macro_f1"])

        train_val_msg = (
            f"[{phase_name}] Epoch {epoch} Summary => "
            f"TrainLoss: {train_loss:.4f}, TrainMacroF1: {train_macro_f1:.4f}, "
            f"ValMacroF1({monitor_level}): {raw_macro_f1:.4f}/{rolling_macro_f1:.4f} (cur/roll{monitor_window})"
        )
        print(train_val_msg)
        logger.info(train_val_msg)

        rolling_msg = (
            f"[{phase_name}] {monitor_level.capitalize()} current vs rolling-{monitor_window}: "
            f"accuracy {raw_accuracy:.4f}/{rolling_metrics['accuracy']:.4f}, "
            f"balanced_accuracy {raw_balanced_accuracy:.4f}/{rolling_metrics['balanced_accuracy']:.4f}, "
            f"macro_precision {raw_macro_precision:.4f}/{rolling_metrics['macro_precision']:.4f}, "
            f"macro_recall {raw_macro_recall:.4f}/{rolling_metrics['macro_recall']:.4f}, "
            f"macro_f1 {raw_macro_f1:.4f}/{rolling_macro_f1:.4f}"
        )
        print(rolling_msg)
        logger.info(rolling_msg)

        # Per-epoch metrics row for post-hoc plotting. Schema stays stable:
        # one object per epoch, new pipelines append their extras inside
        # val_patient / val_slice sub-objects rather than polluting the root.
        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "phase": phase_name,
            "epoch": epoch,
            "model_type": model_type,
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else None,
            "train_loss": float(train_loss),
            "train_macro_f1": float(train_macro_f1),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_precision": float(val_metrics["macro_precision"]),
            "val_macro_recall": float(val_metrics["macro_recall"]),
            "val_macro_f1": raw_macro_f1,
            "val_macro_f1_rolling": rolling_macro_f1,
            "monitor_level": monitor_level,
            "monitor_window": int(monitor_window),
            "epochs_no_improve": int(epochs_no_improve),
        }
        if "slice_level" in val_metrics:
            row["val_slice"] = val_metrics["slice_level"]
        if "patient_level" in val_metrics:
            row["val_patient"] = val_metrics["patient_level"]
        _append_metrics_row(metrics_path, row)

        scheduler.step()

        if rolling_macro_f1 > best_monitor_macro_f1:
            best_monitor_macro_f1 = rolling_macro_f1
            best_ckpt = os.path.join(checkpoint_dir, f"{stamp_day_month}_{model_tag}_pbest.pth")
            torch.save(model.state_dict(), best_ckpt)
            logger.info(
                f"[*] New best model saved! Rolling{monitor_level.capitalize()}MacroF1({monitor_window}): {best_monitor_macro_f1:.4f} "
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
        
    # Organize outputs as: {output_dir}/{pipeline}/{model_type}/
    pipeline_dir = get_pipeline(args.model_type)
    args.output_dir = os.path.join(args.output_dir, pipeline_dir, args.model_type)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.clear_cache:
        cache_root = os.path.join(os.path.expanduser("~"), ".cache")
        for name in (
            "monai_lung_pet_ct_clean", "monai_biglunge",
            "monai_lung_pet_ct_clean_2p5d", "monai_biglunge_2p5d",
            "monai_lung_pet_ct_clean_2d", "monai_biglunge_2d",
        ):
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
    if getattr(args, "config", ""):
        applied = getattr(args, "_config_applied", {})
        logger.info(f"Loaded config: {args.config} ({len(applied)} values applied as defaults)")

    _dump_effective_config(args.output_dir, args, logger)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    if args.model_checkpoint and args.mode in ["full", "dapt"]:
        logger.warning(
            "--model-checkpoint is ignored in modes 'full' and 'dapt'. "
            "It is only used in modes 'finetune' and 'inference'."
        )
    
    # Segmentation auxiliary loss is only meaningful for SwinUNETR (its decoder
    # actually consumes the gradient); all other wrappers return a zero-tensor
    # seg output. The 2.5D/2D EfficientNet paths have no seg output at all.
    use_segmentation_loss = (args.model_type == "swin_unetr")

    # Pipeline dispatch: pick train/validate fns once for the whole run.
    pipeline = get_pipeline(args.model_type)
    if pipeline == "2d":
        train_fn, validate_fn = train_epoch_2d, validate_epoch_2d
    else:
        train_fn, validate_fn = train_epoch, validate_epoch
    logger.info(f"Pipeline: {pipeline} (model_type={args.model_type})")

    model = get_sclc_model(
        args.initial_checkpoint,
        model_type=args.model_type,
        in_channels=1,
        depth_size=args.depth_size,
        num_slices=args.num_slices,
    ).to(device)
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
            use_segmentation=use_segmentation_loss,
            accumulation_steps=args.accumulation_steps,
            model_type=args.model_type,
            monitor_window=args.monitor_rolling_window,
            seg_loss_weight=args.seg_loss_weight,
            warmup_epochs=args.dapt_warmup_epochs,
            warmup_start_lr=args.warmup_start_lr,
            train_fn=train_fn, validate_fn=validate_fn,
            metrics_path=metrics_path,
        )
        current_checkpoint = best_dapt_ckpt
        
    # --- PHASE 2: FINETUNE ---
    if args.mode in ["full", "finetune"]:
        if args.mode == "finetune" and args.model_checkpoint:
            model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
            logger.info(f"Loaded model checkpoint for fine-tuning: {args.model_checkpoint}")
        elif current_checkpoint and current_checkpoint != args.initial_checkpoint and os.path.isfile(current_checkpoint):
            # DAPT produced a checkpoint in full mode — load it.
            model.load_state_dict(torch.load(current_checkpoint, map_location=device))
            logger.info(f"Loaded DAPT checkpoint for fine-tuning: {current_checkpoint}")
        else:
            # No DAPT checkpoint — fine-tune from in-memory weights.
            # In --mode finetune without --model-checkpoint this means the
            # model still has the initial backbone weights (e.g. BTCV) loaded by
            # get_sclc_model(), which is a valid starting point.
            logger.info("Fine-tuning from initial in-memory weights (no DAPT checkpoint).")
            
        logger.info(f"Setting up FineTuning Datasets from: {args.finetune_dataset}")
        train_loader, val_loader, test_loader = create_dataloaders(args, "big_lunge", args.finetune_dataset, args.finetune_csv, depth_size=args.depth_size)
        
        best_finetune_ckpt = run_training_phase(
            model, train_loader, val_loader, device,
            args.finetune_epochs, args.finetune_lr, args.weight_decay, args.checkpoint_dir, logger,
            "finetune", scaler=scaler,
            use_segmentation=use_segmentation_loss,
            accumulation_steps=args.accumulation_steps,
            model_type=args.model_type,
            monitor_window=args.monitor_rolling_window,
            differential_lr=True,
            backbone_lr_scale=args.finetune_backbone_lr_scale,
            seg_loss_weight=args.seg_loss_weight,
            warmup_epochs=args.finetune_warmup_epochs,
            warmup_start_lr=args.warmup_start_lr,
            train_fn=train_fn, validate_fn=validate_fn,
            metrics_path=metrics_path,
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
            test_metrics = validate_fn(model, test_loader, device, logger, return_probabilities=True)
            logger.info(f"Final Test Set Accuracy: {test_metrics['accuracy']:.4f}")

            test_row: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "phase": "test",
                "epoch": None,
                "model_type": args.model_type,
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "test_macro_precision": float(test_metrics["macro_precision"]),
                "test_macro_recall": float(test_metrics["macro_recall"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
            }
            if "slice_level" in test_metrics:
                test_row["test_slice"] = test_metrics["slice_level"]
            if "patient_level" in test_metrics:
                test_row["test_patient"] = test_metrics["patient_level"]
            _append_metrics_row(metrics_path, test_row)

            prob_payload = test_metrics.get("inference_probabilities")
            if prob_payload is not None:
                probs_file = _save_inference_probabilities(args.output_dir, args.model_type, prob_payload, logger)
                print(f"Saved inference probabilities to: {probs_file}")
        else:
            logger.error("Test loader not available. Skipping inference.")
            
    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
