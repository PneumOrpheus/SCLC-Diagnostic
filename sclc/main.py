import os
import sys
import csv
import shutil
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Any, List, Tuple
import argparse
import time
from datetime import datetime
from collections import Counter, deque

try:
    import yaml
except ImportError:  # PyYAML is only needed when --config is used
    yaml = None

from sclc.models import (
    get_sclc_model, get_pipeline, MILResNet50Classifier, MILSwinTinyClassifier, MILSwinV2BaseClassifier,
    MILSwinV2TinyClassifier,
)
from sclc.training.train_3d import simple_collate_fn, train_epoch, validate_epoch
from sclc.training.train_2d import simple_collate_fn_2d, train_epoch_2d, validate_epoch_2d
from sclc.training.train_mil import simple_collate_fn_mil, train_epoch_mil, validate_epoch_mil
from sclc.data.loaders import create_dataset
from sclc.data.dataset_2d import create_dataset_2d
from sclc.data.dataset_mil import create_dataset_mil_bag, create_dataset_mil_bag_dapt, create_dataset_whole_slice
from sclc.logger import create_logger


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


def _save_inference_probabilities(output_dir: str, model_type: str, payload: Dict[str, Any], logger, suffix: str = "") -> str:
    """Persist inference softmax probabilities to disk for post-hoc analysis.

    ``suffix`` (e.g. "dapt") is inserted into the filename so multiple test
    evaluations in the same run (DAPT test + finetune test) don't overwrite
    each other.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"_{suffix}" if suffix else ""
    out_path = os.path.join(output_dir, f"{model_type}_{timestamp}{tag}_inference_probabilities.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    mean_probs = payload.get("mean_probability_per_class", {})
    if mean_probs:
        msg = ", ".join([f"mean P({k})={float(v):.4f}" for k, v in mean_probs.items()])
        logger.info(f"Inference probability means: {msg}")
    logger.info(f"Saved inference probabilities to: {out_path}")
    return out_path


def _save_misclassifications_csv(
    output_dir: str, model_type: str, payload: Dict[str, Any], logger,
    phase: str, suffix: str = "",
) -> str:
    """Filter the inference payload to misclassified samples and write a CSV.

    One row per misclassified sample with patient/volume identifiers, true
    vs predicted labels, model confidence (max softmax) and per-class
    probabilities. Rows are sorted by (true_name, pred_name) ascending and
    confidence descending — confident-wrong samples cluster at the top of
    each confusion bucket, which is the kind that teaches you the most
    on manual review.

    For pipelines that emit a patient-level rollup inside the inference
    payload (``patient_level.samples``), we use that — it's the rollup
    matching the headline "patient-level macro F1" metric. Otherwise we
    fall back to the volume-level ``samples`` list.

    Returns the CSV path written, or '' if no misclassifications existed
    (the file is still written, with a header row only).
    """
    if not isinstance(payload, dict):
        return ""

    # Prefer patient-level samples if the pipeline emitted them (2D path).
    # MIL and 3D emit one sample per patient already.
    pat_block = payload.get("patient_level") if isinstance(payload, dict) else None
    samples: List[Dict[str, Any]] = []
    if isinstance(pat_block, dict) and isinstance(pat_block.get("samples"), list):
        samples = list(pat_block["samples"])
        sample_level = "patient"
    else:
        samples = list(payload.get("samples") or [])
        sample_level = "volume_or_bag"

    class_names: List[str] = list(payload.get("class_names") or [])

    # Filter to misclassified rows.
    wrong = [s for s in samples if int(s.get("pred_label", -1)) != int(s.get("true_label", -2))]

    # Sort: group by (true_name, pred_name); within each group, highest
    # confidence first (confident wrongs first — those are the ones that
    # most likely indicate a real failure mode rather than a borderline call).
    wrong.sort(key=lambda s: (
        str(s.get("true_name", "")),
        str(s.get("pred_name", "")),
        -float(s.get("confidence", 0.0)),
    ))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"_{suffix}" if suffix else ""
    out_path = os.path.join(
        output_dir,
        f"{model_type}_{timestamp}{tag}_misclassifications.csv",
    )

    base_columns = [
        "phase", "model_type", "sample_level",
        "patient_id", "volume_id",
        "true_label", "true_name",
        "pred_label", "pred_name",
        "confidence",
    ]
    prob_columns = [f"prob_{c}" for c in class_names]
    columns = base_columns + prob_columns

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for s in wrong:
            probs = s.get("probabilities") or {}
            # 2D patient-level samples carry ``volume_ids`` (a list) instead
            # of a single ``volume_id``. Join with ';' so the CSV stays one
            # row per patient and the user can still see all underlying paths.
            vid = s.get("volume_id")
            if vid is None and isinstance(s.get("volume_ids"), list):
                vid = ";".join(str(v) for v in s["volume_ids"])
            row = {
                "phase": phase,
                "model_type": model_type,
                "sample_level": sample_level,
                "patient_id": s.get("patient_id"),
                "volume_id": vid,
                "true_label": s.get("true_label"),
                "true_name": s.get("true_name"),
                "pred_label": s.get("pred_label"),
                "pred_name": s.get("pred_name"),
                "confidence": s.get("confidence"),
            }
            for c in class_names:
                row[f"prob_{c}"] = probs.get(c)
            w.writerow(row)

    n_total = len(samples)
    n_wrong = len(wrong)
    logger.info(
        f"[{phase}] Misclassifications: {n_wrong}/{n_total} "
        f"({(n_wrong / n_total * 100.0) if n_total else 0:.1f}%) -> {out_path}"
    )
    return out_path


def _run_test_inference(
    model, test_loader, device, logger, validate_fn,
    metrics_path: str, output_dir: str, model_type: str,
    phase: str, prob_file_suffix: str = "",
) -> Dict[str, Any]:
    """Run validate_fn on test_loader, log a summary, append a metrics row,
    and save inference probabilities. ``phase`` labels the metrics row
    (e.g. "dapt_test", "test"); ``prob_file_suffix`` disambiguates the
    probability-file name when multiple test runs share an output dir.
    """
    logger.info(f"Running evaluation on the {phase} set...")
    test_metrics = validate_fn(model, test_loader, device, logger, return_probabilities=True)
    logger.info(
        f"[{phase}] loss: {test_metrics['loss']:.4f} | accuracy: {test_metrics['accuracy']:.4f} | "
        f"macro_f1: {test_metrics['macro_f1']:.4f}"
    )

    test_row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "epoch": None,
        "model_type": model_type,
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
        _save_inference_probabilities(output_dir, model_type, prob_payload, logger, suffix=prob_file_suffix)
        _save_misclassifications_csv(
            output_dir, model_type, prob_payload, logger,
            phase=phase, suffix=prob_file_suffix,
        )
    return test_metrics

def _save_cv_aggregate_metrics(
    fold_results: List[Dict[str, Any]],
    metrics_path: str,
    model_type: str,
    n_folds: int,
    logger,
) -> None:
    """Average per-fold test metrics and write a single 'test' row to metrics.jsonl.

    The row mirrors the schema of a regular test row so downstream scripts
    (ablation_plots.py, build_thesis_results.py) read it without modification.
    """
    import numpy as np

    def _avg(key_path):
        vals = []
        for r in fold_results:
            v = r
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    def _avg_list(key_path):
        lists = []
        for r in fold_results:
            v = r
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, list) and v:
                lists.append([float(x) for x in v])
        if not lists:
            return None
        length = len(lists[0])
        return [float(np.mean([lst[i] for lst in lists if i < len(lst)])) for i in range(length)]

    pl, sl = "patient_level", "slice_level"

    # per_class_f1_ci95 is [[lo,hi],[lo,hi],[lo,hi]] — average per class per bound.
    def _avg_pc_ci():
        all_ci = [r.get(pl, {}).get("per_class_f1_ci95") for r in fold_results]
        all_ci = [c for c in all_ci if isinstance(c, list) and c]
        if not all_ci:
            return None
        n_classes = len(all_ci[0])
        return [
            [float(np.mean([all_ci[f][c][b] for f in range(len(all_ci))])) for b in range(2)]
            for c in range(n_classes)
        ]

    patient_agg: Dict[str, Any] = {
        "accuracy": _avg([pl, "accuracy"]),
        "balanced_accuracy": _avg([pl, "balanced_accuracy"]),
        "macro_precision": _avg([pl, "macro_precision"]),
        "macro_recall": _avg([pl, "macro_recall"]),
        "macro_f1": _avg([pl, "macro_f1"]),
        "per_class_precision": _avg_list([pl, "per_class_precision"]),
        "per_class_recall": _avg_list([pl, "per_class_recall"]),
        "per_class_f1": _avg_list([pl, "per_class_f1"]),
        "macro_f1_ci95": _avg_list([pl, "macro_f1_ci95"]),
        "balanced_accuracy_ci95": _avg_list([pl, "balanced_accuracy_ci95"]),
        "per_class_f1_ci95": _avg_pc_ci(),
        "num_patients": sum(r.get(pl, {}).get("num_patients", 0) for r in fold_results),
    }

    agg_row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": "test",
        "epoch": None,
        "model_type": model_type,
        "cv_folds": n_folds,
        "test_loss": _avg(["loss"]),
        "test_accuracy": _avg(["accuracy"]),
        "test_balanced_accuracy": _avg(["balanced_accuracy"]),
        "test_macro_precision": _avg(["macro_precision"]),
        "test_macro_recall": _avg(["macro_recall"]),
        "test_macro_f1": _avg(["macro_f1"]),
    }
    if any(sl in r for r in fold_results):
        agg_row["test_slice"] = {
            "accuracy": _avg([sl, "accuracy"]),
            "macro_f1": _avg([sl, "macro_f1"]),
            "num_slices": sum(r.get(sl, {}).get("num_slices", 0) for r in fold_results),
        }
    if any(pl in r for r in fold_results):
        agg_row["test_patient"] = patient_agg

    _append_metrics_row(metrics_path, agg_row)
    mf1 = patient_agg.get("macro_f1") or agg_row.get("test_macro_f1")
    logger.info(
        f"[CV] {n_folds}-fold average → MacroF1={mf1:.4f} "
        f"(per-fold rows tagged test_fold_0 … test_fold_{n_folds-1})"
    )


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
                        choices=["swin_unetr",
                                 "efficientnet_b0_2d", "densenet121_2d", "resnet50_2d",
                                 "swin_tiny_2d", "swinv2_base_2d", "swinv2_tiny_2d", "resnet50_2d_rin", "densenet121_2d_rin",
                                 "mil_resnet50", "mil_swin_tiny", "mil_swinv2_base", "mil_swinv2_tiny"],
                        help="Model architecture to use. '_2d' uses the per-slice 2D pipeline; "
                             "'_2d_rin' uses the RadImageNet-pretrained 2D variants "
                             "(ResNet50 / DenseNet121); 'mil_resnet50' / 'mil_swin_tiny' use the "
                             "MIL pipeline (whole-slice per-slice DAPT + attention-MIL bag "
                             "finetune); 'swin_unetr' uses the full 3D pipeline.")
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
    parser.add_argument("--linear-probe", action="store_true",
                        help="DAPT with the backbone frozen for the entire run (head-only linear probe). "
                             "Uses --linear-probe-lr in place of --dapt-lr. Checkpoints are tagged dapt_lp "
                             "so they do not overwrite full-DAPT runs.")
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3,
                        help="Head-only LR for --linear-probe. A fresh linear head converges faster on a "
                             "frozen backbone than the full-model DAPT LR.")
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Number of stratified CV folds for the BigLunge finetune phase. "
                             "1 (default) uses the original fixed 70/15/15 split. "
                             "5 runs 5-fold CV and writes per-fold metrics (test_fold_k) "
                             "plus an averaged 'test' row.")
    parser.add_argument("--finetune-freeze-backbone-epochs", type=int, default=0,
                        help="LP-FT recipe: freeze the backbone for the first N fine-tune epochs (head trains "
                             "alone), then unfreeze and apply differential LR. 0 disables (full diff-LR from "
                             "epoch 1, the previous default). 5-10 is a reasonable range for small target sets "
                             "like BigLunge.")
    parser.add_argument("--finetune-backbone-lr-scale", type=float, default=0.1,
                        help="Backbone LR multiplier for fine-tune differential LR (backbone_lr = finetune_lr * scale).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps to increase effective batch size")
    parser.add_argument("--seg-loss-weight", type=float, default=0.1,
                        help="Weight for the auxiliary segmentation loss (only active for SwinUNETR; ignored otherwise).")
    parser.add_argument("--use-advanced-fpn", action="store_true",
                        help="Enable the advanced FPN neck (SPM/CAM/MBFFM + optional TFPN).")
    parser.add_argument("--use-det-seg", action="store_true",
                        help="Enable detection + segmentation heads and losses (requires tumor masks).")
    parser.add_argument("--bbox-loss-weight", type=float, default=0.1,
                        help="Weight for the bounding-box regression loss when --use-det-seg is active.")
    parser.add_argument("--bbox-source", type=str, default="mask", choices=["mask", "xml"],
                        help="Bounding box source for det/seg targets. 'mask' derives boxes from tumor masks.")
    parser.add_argument("--fpn-channels", type=int, default=256,
                        help="Channel width for the advanced FPN neck.")
    parser.add_argument("--tfpn-heads", type=int, default=4,
                        help="Number of attention heads in each TFPN block.")
    parser.add_argument("--tfpn-layers", type=int, default=1,
                        help="Number of TFPN layers per pyramid level.")
    parser.add_argument("--tfpn-levels", type=int, default=1,
                        help="Number of highest pyramid levels to run TFPN on.")
    parser.add_argument("--disable-tfpn", action="store_true",
                        help="Disable TFPN blocks inside the advanced FPN neck.")
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
    parser.add_argument("--output-dir", type=str, default="results/output")
    parser.add_argument("--checkpoint-dir", type=str, default="/home/data/trained_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--testing", default=False, action="store_true", help="Run with a tiny subset for testing")
    parser.add_argument("--clear-cache", default=False, action="store_true",
                        help="Delete the MONAI PersistentDataset cache before building datasets")
    parser.add_argument("--depth-size", type=int, default=128, help="Depth size for the 3D images (must be divisible by 32 for SwinUNETR)")

    parser.add_argument("--tumor-mask-suffix", type=str, default="_label_tc.nii.gz",
                        help="Per-patient tumor mask suffix expected under the BigLunge patient folder "
                             "(e.g. patient_081613/patient_081613_label_tc.nii.gz). Ignored for Lung-PET-CT-Dx, "
                             "which uses per-series '{series_uid}_mask.nii.gz' sidecars.")

    # 2D pipeline knobs
    parser.add_argument("--img-size-2d", type=int, default=224,
                        help="In-plane size for the 2D tumor-centered slice crop.")
    parser.add_argument("--img-crop-2d", type=int, default=96,
                        help="In-plane size of CropAroundTumord BEFORE Resized. Default 96 matches the "
                             "original DAPT setup on Lung-PET-CT-Dx. BigLunge tumors are larger and "
                             "centroid-edge overflow is visible — use 160 or 192 there. Cache is keyed "
                             "on this value so switching sizes doesn't clobber prior runs.")
    parser.add_argument("--min-tumor-pixels", type=int, default=100,
                        help="Minimum non-zero mask voxels per axial slice for the slice to be included "
                             "as a training sample (after Orient+Spacing to the pipeline grid). Default "
                             "bumped from 1 to 100 to filter algorithmic-segmentation false positives "
                             "(tiny subpleural findings that produce chest-wall crops instead of tumor "
                             "crops). Audit evidence in data_exploration/BigLunge_expl.ipynb.")
    parser.add_argument("--max-slices-per-volume", type=int, default=8,
                        help="Cap tumor slices sampled per volume in the 2D pipeline (0 = no cap). "
                             "Each uncapped slice triggers a full volume reload during cache build, so keep this small.")
    parser.add_argument("--cache-workers", type=int, default=4,
                        help="Parallel workers used when building the 2D/2.5D/3D PersistentDataset cache for the first time.")
    parser.add_argument("--strong-augs", action="store_true",
                        help="Enable the 'strong' 2D augmentation block (heavier RandAffine / intensity + "
                             "RandGaussianSmooth + RandCoarseDropout). Target: closing the train-val gap on "
                             "2D DAPT runs. See docs/2d_augmentations.md. PersistentDataset cache is NOT "
                             "invalidated — MONAI caches only up to the last deterministic transform.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="MixUp Beta(alpha, alpha) parameter for batch-level mixing in the 2D training "
                             "loop. 0 disables (default). Recommended starting point: 0.2. Paired with "
                             "--strong-augs for the overfitting-mitigation experiment.")
    parser.add_argument("--bag-dropout", type=float, default=0.0,
                        help="MIL only: per-instance dropout probability inside each bag at train time. "
                             "Acts as bag-level cutout to mitigate attention collapse onto a single slice. "
                             "0 disables (default). Try 0.10-0.20.")

    # MIL pipeline knobs
    parser.add_argument("--img-size-mil", type=int, default=224,
                        help="In-plane size for the whole-slice MIL pipeline (both DAPT per-slice and "
                             "BigLunge MIL-bag phases). Larger than the 2D default (224) because the "
                             "whole axial slice is fed in — a 10 mm tumor occupies ~6 px at 224, so we "
                             "default higher to preserve tumor-scale features.")
    parser.add_argument("--bag-size", type=int, default=16,
                        help="Number of axial slices per MIL bag, evenly spaced across the lung mask's "
                             "z-extent. Memory scales linearly with this (B * N forward passes through "
                             "the backbone per step).")
    parser.add_argument("--mil-mode", type=str, default="att",
                        choices=["mean", "max", "att", "att_trans", "att_trans_pyramid"],
                        help="MILModel pooling mode (see monai.networks.nets.MILModel). 'att' is the "
                             "Ilse-2018 attention-MIL baseline; 'att_trans' adds transformer blocks "
                             "(more capacity, more overfit risk on small data).")
    parser.add_argument("--mil-trans-blocks", type=int, default=4,
                        help="Number of transformer blocks for mil_mode in {'att_trans', 'att_trans_pyramid'}.")
    parser.add_argument("--mil-trans-dropout", type=float, default=0.0,
                        help="Dropout rate in the MIL transformer encoder.")
    parser.add_argument("--lung-mask-suffix", type=str, default="_label_lungs.nii.gz",
                        help="Per-patient lung-chamber mask suffix used by the BigLunge MIL bag builder "
                             "(e.g. patient_081613/patient_081613_label_lungs.nii.gz).")

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


def create_dataloaders(args, dataset_type, data_path, csv_path="", depth_size=64, phase: str = "dapt",
                       cv_fold: int = -1):
    """Build train/val/test DataLoaders for the current phase.

    ``phase`` matters only for the MIL pipeline, which uses different datasets
    for DAPT (per-slice whole-slice) vs fine-tune / inference (bag-level).
    All other pipelines ignore ``phase``.

    ``cv_fold`` selects a specific stratified k-fold split when >= 0 (the
    number of folds comes from ``args.cv_folds``). -1 uses the original fixed split.
    """
    pipeline = get_pipeline(args.model_type)
    include_mask = bool(getattr(args, "use_det_seg", False))
    include_bbox = bool(getattr(args, "use_det_seg", False))
    n_folds = int(getattr(args, "cv_folds", 1))
    if pipeline == "mil":
        if phase in ("finetune", "inference"):
            train_ds, val_ds, test_ds = create_dataset_mil_bag(
                data_path=data_path,
                csv_path=csv_path,
                dataset_type=dataset_type,
                img_size=args.img_size_mil,
                bag_size=args.bag_size,
                lung_mask_suffix=args.lung_mask_suffix,
                tumor_mask_suffix=args.tumor_mask_suffix,
                testing=args.testing,
                cache_workers=args.cache_workers,
                strong_augs=bool(getattr(args, "strong_augs", False)),
                clear_cache=bool(getattr(args, "clear_cache", False)),
                include_mask=include_mask,
                include_bbox=include_bbox,
                cv_fold=cv_fold,
                cv_folds=n_folds,
            )
            collate_fn = simple_collate_fn_mil
        else:  # dapt — bag-level dataset on Lung-PET-CT-Dx
            train_ds, val_ds, test_ds = create_dataset_mil_bag_dapt(
                data_path=data_path,
                img_size=args.img_size_mil,
                bag_size=args.bag_size,
                testing=args.testing,
                cache_workers=args.cache_workers,
                strong_augs=bool(getattr(args, "strong_augs", False)),
                clear_cache=bool(getattr(args, "clear_cache", False)),
                cv_fold=cv_fold,
                cv_folds=n_folds,
            )
            collate_fn = simple_collate_fn_mil
    elif pipeline == "2d":
        max_slices = args.max_slices_per_volume if args.max_slices_per_volume and args.max_slices_per_volume > 0 else None
        train_ds, val_ds, test_ds = create_dataset_2d(
            data_path=data_path,
            csv_path=csv_path,
            dataset_type=dataset_type,
            img_size=args.img_size_2d,
            tumor_mask_suffix=args.tumor_mask_suffix,
            max_slices_per_volume=max_slices,
            min_tumor_pixels=int(getattr(args, "min_tumor_pixels", 100)),
            crop_size=int(getattr(args, "img_crop_2d", 96)),
            testing=args.testing,
            cache_workers=args.cache_workers,
            strong_augs=bool(getattr(args, "strong_augs", False)),
            clear_cache=bool(getattr(args, "clear_cache", False)),
            include_mask=include_mask,
            include_bbox=include_bbox,
            cv_fold=cv_fold,
            cv_folds=n_folds,
        )
        collate_fn = simple_collate_fn_2d
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
            strong_augs=bool(getattr(args, "strong_augs", False)),
            clear_cache=bool(getattr(args, "clear_cache", False)),
            include_bbox=include_bbox,
            cv_fold=cv_fold,
            cv_folds=n_folds,
        )
        collate_fn = simple_collate_fn

    if hasattr(train_ds, "data") and len(train_ds.data) > 0:
        train_labels = [item["scan_label"] for item in train_ds.data]
        class_counts = Counter(train_labels)
        num_samples = len(train_labels)

        # Invert class frequencies to create weights
        class_weights_dict = {cls: num_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights_dict[label] for label in train_labels]

        # Make the oversampling effect visible. The factor is the ratio of
        # the most-oversampled to the least-oversampled class — i.e. how
        # many times more often a minority sample is drawn relative to the
        # majority sample. 1.0 means WRS is a no-op (perfectly balanced).
        if class_weights_dict:
            w_max = max(class_weights_dict.values())
            w_min = min(class_weights_dict.values())
            oversample_factor = w_max / w_min if w_min > 0 else float("inf")
            counts_str = ", ".join(f"{cls}={cnt}" for cls, cnt in sorted(class_counts.items()))
            print(
                f"[WRS {dataset_type}] counts {{{counts_str}}} | "
                f"oversample_factor={oversample_factor:.2f} (max_weight/min_weight)"
            )

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


_HEAD_PREFIXES = (
    "classification_head", "dense_1", "dense_2",
    # MIL wrapper: attention and myfc are the bag-level classifier; both are
    # freshly initialized when we load a DAPT backbone, so they belong in the
    # "head" group for differential LR.
    "mil.attention", "mil.myfc",
)
_HEAD_TENSOR_SUFFIXES = (
    "._fc.weight", "._fc.bias",  # MONAI EfficientNetBN
    ".fc.weight", ".fc.bias",    # torchvision ResNet (via TorchVisionFCModel)
    ".myfc.weight", ".myfc.bias",  # MONAI MILModel classifier
)


def _is_head_param(name: str) -> bool:
    """Heuristic match for 'this tensor belongs to the classification head'.

    Single source of truth — used by both _set_backbone_frozen and the
    differential-LR split in run_training_phase. Add new patterns here when
    adding a backbone whose head has a different attribute name, and both
    code paths stay consistent.
    """
    if name.startswith(_HEAD_PREFIXES):
        return True
    if "class_layers" in name:  # MONAI DenseNet121
        return True
    if name.endswith(_HEAD_TENSOR_SUFFIXES):
        return True
    return False


def _set_backbone_frozen(model, frozen: bool, logger=None) -> int:
    """Freeze/unfreeze every parameter that is NOT part of a classification head.

    Returns the number of backbone parameters affected. See _is_head_param
    for the head-detection heuristic.
    """
    n_backbone = 0
    n_head = 0
    for name, param in model.named_parameters():
        is_head = _is_head_param(name)
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


def _ensure_writable_dir(path: str) -> Tuple[bool, str]:
    """Create dir if needed and verify write access with a tiny probe file."""
    try:
        os.makedirs(path, exist_ok=True)
        probe_name = f".write_test_{os.getpid()}"
        probe_path = os.path.join(path, probe_name)
        with open(probe_path, "w") as f:
            f.write("ok")
        os.remove(probe_path)
        return True, ""
    except OSError as exc:
        return False, str(exc)


def _resolve_checkpoint_dir(checkpoint_dir: str, output_dir: str) -> Tuple[str, str]:
    """Return a writable checkpoint dir and an optional warning message."""
    ok, err = _ensure_writable_dir(checkpoint_dir)
    if ok:
        return checkpoint_dir, ""
    fallback_dir = os.path.join(output_dir, "checkpoints")
    ok_fallback, err_fallback = _ensure_writable_dir(fallback_dir)
    if not ok_fallback:
        raise RuntimeError(
            f"Checkpoint dir not writable: {checkpoint_dir} ({err}). "
            f"Fallback failed: {fallback_dir} ({err_fallback})."
        )
    msg = (
        f"[checkpoint] '{checkpoint_dir}' not writable ({err}). "
        f"Falling back to '{fallback_dir}'."
    )
    return fallback_dir, msg


def run_training_phase(
    model, train_loader, val_loader, device, epochs, lr, weight_decay,
    checkpoint_dir, logger, phase_name, patience=10, scaler=None, use_segmentation=False, accumulation_steps=4,
    model_type="swin_unetr",
    warmup_epochs: int = 0, warmup_start_lr: float = 5e-6, freeze_backbone_epochs: int = 0,
    monitor_window: int = 3,
    differential_lr: bool = False,
    backbone_lr_scale: float = 0.1,
    seg_loss_weight: float = 0.1,
    use_det_seg: bool = False,
    bbox_loss_weight: float = 0.1,
    mixup_alpha: float = 0.0,
    bag_dropout: float = 0.0,
    train_fn=train_epoch,
    validate_fn=validate_epoch,
    metrics_path: str = "",
):
    seg_loss_weight = max(0.0, float(seg_loss_weight))
    bbox_loss_weight = max(0.0, float(bbox_loss_weight))

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
            is_head = _is_head_param(name)
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
        # Cosine over the full run for the head group. During the freeze
        # window, the backbone group has requires_grad=False so its scheduled
        # LR does not matter; after unfreeze, both groups follow the same
        # cosine schedule from their base LRs onward (with partial decay
        # already applied — accepted trade-off for keeping the scheduler
        # logic simple). See flaws.md 1.6.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if warmup_epochs > 0:
            logger.info(
                f"[{phase_name}] Ignoring warmup_epochs={warmup_epochs} because differential_lr=True."
            )
        if freeze_backbone_epochs > 0:
            logger.info(
                f"[{phase_name}] LP-FT mode: freezing backbone for {freeze_backbone_epochs} epochs, "
                f"then differential LR (backbone={lr * backbone_lr_scale:.2e} / head={lr:.2e})."
            )
    else:
        scheduler = _build_scheduler(optimizer, epochs, warmup_epochs, warmup_start_lr, lr)

    if freeze_backbone_epochs > 0:
        n_backbone = _set_backbone_frozen(model, frozen=True, logger=logger)
        if n_backbone == 0:
            logger.warning(
                f"[{phase_name}] freeze_backbone_epochs={freeze_backbone_epochs} requested but "
                f"no backbone parameters were identified — check _HEAD_PREFIXES for this model type."
            )

    monitor_window = max(1, int(monitor_window))
    phase_prefix = phase_name.lower().replace(' ', '_').replace('_phase', '')
    stamp_day_month = datetime.now().strftime("%h_%d_%m")
    # Checkpoints live under {checkpoint_dir}/{pipeline}/{model_type}/, mirroring
    # the output tree. The filename drops the model_type prefix since the
    # directory already encodes it; phase ("dapt" / "dapt_lp" / "finetune") stays
    # in the filename so DAPT and finetune checkpoints don't overwrite each other.
    pipeline_dir = get_pipeline(model_type)
    ckpt_save_dir = os.path.join(checkpoint_dir, pipeline_dir, model_type)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    rolling_history = {
        "loss": deque(maxlen=monitor_window),
        "accuracy": deque(maxlen=monitor_window),
        "balanced_accuracy": deque(maxlen=monitor_window),
        "macro_precision": deque(maxlen=monitor_window),
        "macro_recall": deque(maxlen=monitor_window),
        "macro_f1": deque(maxlen=monitor_window),
    }

    # Dual-best tracking. `_pbest_raw.pth` is the single-epoch peak (the
    # conventional "best" reported in the thesis); `_pbest_roll.pth` is the
    # rolling-window peak that drives early stopping. The function returns
    # the raw checkpoint so downstream phases load the cleaner peak. See
    # flaws.md 1.1 for the original ambiguity this resolves.
    best_raw_macro_f1 = -1.0
    best_roll_macro_f1 = -1.0
    best_raw_ckpt = None
    best_roll_ckpt = None
    epochs_no_improve = 0

    if use_segmentation:
        logger.info(f"[{phase_name}] Segmentation auxiliary loss enabled with seg_loss_weight={seg_loss_weight:.3f}")
    if use_det_seg:
        logger.info(f"[{phase_name}] Detection/segmentation enabled with bbox_loss_weight={bbox_loss_weight:.3f}")

    for epoch in range(1, epochs + 1):
        # Unfreeze exactly once, at the boundary. Same boundary regardless of
        # whether diff_lr is on (LP-FT case) or off (single-LR freeze case).
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
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
            use_det_seg=use_det_seg,
            accumulation_steps=accumulation_steps, # Pass down here
            seg_loss_weight=seg_loss_weight,
            bbox_loss_weight=bbox_loss_weight,
            mixup_alpha=mixup_alpha,
            bag_dropout=bag_dropout,
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
            # When mixup_active is True, train_macro_f1 is dominant-label
            # agreement on mixed inputs (lam>=0.5), not a real classifier F1.
            # See flaws.md 1.2.
            "mixup_alpha": float(mixup_alpha),
            "mixup_active": bool(mixup_alpha > 0.0),
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

        # Save raw-best (single-epoch peak) and roll-best (rolling-window peak)
        # as separate files. Early stopping still tracks rolling — it's the
        # smoother monitor — but the canonical checkpoint we report is raw.
        if raw_macro_f1 > best_raw_macro_f1:
            best_raw_macro_f1 = raw_macro_f1
            best_raw_ckpt = os.path.join(ckpt_save_dir, f"{stamp_day_month}_{phase_prefix}_pbest_raw.pth")
            torch.save(model.state_dict(), best_raw_ckpt)
            logger.info(
                f"[*] New raw-best @ ep{epoch}: {monitor_level}_macro_f1={raw_macro_f1:.4f} -> {best_raw_ckpt}"
            )

        if rolling_macro_f1 > best_roll_macro_f1:
            best_roll_macro_f1 = rolling_macro_f1
            best_roll_ckpt = os.path.join(ckpt_save_dir, f"{stamp_day_month}_{phase_prefix}_pbest_roll.pth")
            torch.save(model.state_dict(), best_roll_ckpt)
            logger.info(
                f"[*] New roll-best @ ep{epoch}: rolling_macro_f1({monitor_window})={best_roll_macro_f1:.4f} "
                f"(current_raw={raw_macro_f1:.4f}) -> {best_roll_ckpt}"
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_ckpt = os.path.join(
                ckpt_save_dir,
                f"{stamp_day_month}_{phase_prefix}_epoch_{epoch}.pth",
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

    if best_raw_ckpt is None and best_roll_ckpt is None:
        logger.warning(f"[{phase_name}] No checkpoint saved — val macro-F1 never improved.")
    if best_raw_ckpt is not None:
        logger.info(
            f"[{phase_name}] Final best: raw={best_raw_macro_f1:.4f} -> {best_raw_ckpt} | "
            f"roll{monitor_window}={best_roll_macro_f1:.4f} -> {best_roll_ckpt}"
        )
    return best_raw_ckpt or best_roll_ckpt


def main():
    args = parse_args()
    
    if not args.initial_checkpoint and args.model_type == "swin_unetr":
        args.initial_checkpoint = "/home/data/pre_trained_models/model_swin_unetr_btcv_segmentation_v1.pt"
    if not args.initial_checkpoint and args.model_type == "swin_tiny_2d":
        # Default backbone init: RadImageNet-pretrained Swin-Tiny. The
        # SwinTiny2DClassifier wrapper performs the MS->timm key remap and
        # the 3-ch->1-ch stem averaging on load. Pass a different path via
        # --initial-checkpoint to swap in the img2rin variant or an
        # ImageNet-pretrained Swin-Tiny.
        args.initial_checkpoint = "/home/hansstem/RadImageNet_swin/rin_swintf.pth"
    if not args.initial_checkpoint and args.model_type == "mil_swin_tiny":
        # mil_swin_tiny: same RadImageNet-pretrained Swin-Tiny init as swin_tiny_2d.
        # MILSwinTinyClassifier loads this directly in its constructor.
        args.initial_checkpoint = "/home/hansstem/RadImageNet_swin/rin_swintf.pth"
    if not args.initial_checkpoint and args.model_type == "resnet50_2d_rin":
        args.initial_checkpoint = "/home/data/RadImageNet/ResNet50/ResNet50.pt"
    if not args.initial_checkpoint and args.model_type == "densenet121_2d_rin":
        args.initial_checkpoint = "/home/data/RadImageNet/DenseNet/DenseNet121.pt"
        
    # Organize outputs as: {output_dir}/{pipeline}/{model_type}/
    pipeline_dir = get_pipeline(args.model_type)
    args.output_dir = os.path.join(args.output_dir, pipeline_dir, args.model_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.checkpoint_dir, checkpoint_msg = _resolve_checkpoint_dir(args.checkpoint_dir, args.output_dir)

    # --clear-cache is now scoped: each create_dataset_* builder rmtree's
    # only its own run-specific cache parent (the parameterized subdir
    # for THIS run's img_size / depth_size / bag_size / etc.). Sibling
    # caches keep their entries. Plumbed via the clear_cache kwarg
    # passed into each create_* call by create_dataloaders below.


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=not args.disable_amp and device.type == "cuda")
    
    logger = create_logger(output_dir=args.output_dir, dist_rank=-1, name=f"{args.model_type}")
    logger.info(f"Running {args.model_type} 3D Classification Pipeline")
    logger.info(f"Mode: {args.mode} | Testing: {args.testing} | Device: {device} | AMP: {not args.disable_amp}")
    if checkpoint_msg:
        logger.warning(checkpoint_msg)
        print(checkpoint_msg)
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
    # seg output. The 2.5D/2D/MIL EfficientNet paths have no seg output at all.
    #
    # Dataset gating: the loss is only useful when tumor masks are available
    # in the data list. Lung-PET-CT-Dx (DAPT source) has per-series masks;
    # BigLunge's 3D path has only lung masks (no tumor masks), so during
    # BigLunge fine-tune the loss is permanently inert — but if we still pass
    # use_segmentation=True we'd pay the full decoder forward+backward each
    # step for nothing. Phase-specific flags below collapse that waste.
    use_segmentation_loss = (args.model_type == "swin_unetr")
    dapt_use_seg = use_segmentation_loss          # Lung-PET-CT-Dx has tumor masks
    finetune_use_seg = False                       # BigLunge 3D path has no tumor masks
    use_det_seg = bool(getattr(args, "use_det_seg", False))
    tfpn_enabled = not bool(getattr(args, "disable_tfpn", False))
    if args.bbox_source != "mask":
        logger.warning(
            f"bbox_source='{args.bbox_source}' requested but only 'mask' is wired up; "
            "falling back to mask-derived boxes."
        )

    # Pipeline dispatch. For MIL the train/validate fns are phase-specific:
    # DAPT runs as per-slice whole-slice classification (shares the 2D loop);
    # fine-tune / inference runs as bag-level MIL.
    pipeline = get_pipeline(args.model_type)
    if pipeline == "mil":
        # DAPT now uses bag-level dataset → bag-level train/validate loops.
        dapt_train_fn, dapt_validate_fn = train_epoch_mil, validate_epoch_mil
        ft_train_fn, ft_validate_fn = train_epoch_mil, validate_epoch_mil
    elif pipeline == "2d":
        dapt_train_fn, dapt_validate_fn = train_epoch_2d, validate_epoch_2d
        ft_train_fn, ft_validate_fn = train_epoch_2d, validate_epoch_2d
    else:
        dapt_train_fn, dapt_validate_fn = train_epoch, validate_epoch
        ft_train_fn, ft_validate_fn = train_epoch, validate_epoch
    logger.info(f"Pipeline: {pipeline} (model_type={args.model_type})")

    # Model construction. get_sclc_model handles all pipelines including MIL.
    # For MIL, the model is built as a full MIL model from the start; DAPT
    # uses bag-level training via create_dataset_mil_bag_dapt so no 2D
    # classifier intermediary is needed.
    # Save construction kwargs so CV folds can rebuild a fresh model without
    # trying to clone UninitializedParameter tensors (LazyConv FPN laterals).
    _model_kwargs = dict(
        checkpoint_path=args.initial_checkpoint,
        model_type=args.model_type,
        in_channels=1,
        depth_size=args.depth_size,
        mil_mode=args.mil_mode,
        mil_trans_blocks=args.mil_trans_blocks,
        mil_trans_dropout=args.mil_trans_dropout,
        use_advanced_fpn=bool(getattr(args, "use_advanced_fpn", False)),
        use_det_seg=use_det_seg,
        fpn_channels=args.fpn_channels,
        tfpn_enabled=tfpn_enabled,
        tfpn_heads=args.tfpn_heads,
        tfpn_layers=args.tfpn_layers,
        tfpn_levels=args.tfpn_levels,
    )
    model = get_sclc_model(**_model_kwargs).to(device)
    # LazyConv lateral projections in AdvancedFPNNeck are UninitializedParameter
    # until the first forward pass; skip the count rather than crash.
    if any(isinstance(p, torch.nn.parameter.UninitializedParameter)
           for p in model.parameters()):
        logger.info(f"Initialized {args.model_type} Classifier (FPN LazyConv params "
                    "will materialize on first forward pass).")
        print(f"Initialized {args.model_type} Classifier (LazyConv params pending).")
    else:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Initialized {args.model_type} Classifier. Total Params: {num_params:,}")
        print(f"Initialized {args.model_type} Classifier. Total Params: {num_params:,}")

    current_checkpoint = args.initial_checkpoint

    n_folds = int(getattr(args, "cv_folds", 1))
    fold_test_results: List[Dict[str, Any]] = []
    best_dapt_ckpt = None  # set inside fold loop; used by --mode dapt reporting

    for fold_idx in range(n_folds):
        cv_fold = fold_idx if n_folds > 1 else -1

        if fold_idx > 0:
            # Rebuild from scratch rather than cloning state_dict: models with
            # AdvancedFPN have UninitializedParameter (LazyConv laterals) that
            # cannot be cloned before a forward pass materialises them.
            model = get_sclc_model(**_model_kwargs).to(device)
            logger.info(f"[CV fold {fold_idx+1}/{n_folds}] Rebuilt model from initial checkpoint.")

        if n_folds > 1:
            logger.info(f"\n{'='*60}\nCV Fold {fold_idx+1}/{n_folds}\n{'='*60}")

        fold_ckpt_dir = (
            os.path.join(args.checkpoint_dir, f"fold_{fold_idx}")
            if n_folds > 1 else args.checkpoint_dir
        )

        # --- PHASE 1: DAPT ---
        if args.mode in ["full", "dapt"]:
            logger.info(f"Setting up DAPT Datasets from: {args.dapt_dataset}")
            train_loader, val_loader, dapt_test_loader = create_dataloaders(
                args, "lung_pet_ct_dx", args.dapt_dataset,
                depth_size=args.depth_size, phase="dapt",
                cv_fold=cv_fold,
            )

            # Linear-probe DAPT: freeze backbone for the entire run, use a higher
            # head LR, tag checkpoint as dapt_lp.
            dapt_phase_name = "dapt_lp" if args.linear_probe else "dapt"
            dapt_lr = args.linear_probe_lr if args.linear_probe else args.dapt_lr
            dapt_freeze_epochs = args.dapt_epochs + 1 if args.linear_probe else 0
            if args.linear_probe:
                logger.info(
                    f"[linear-probe] Backbone frozen for all {args.dapt_epochs} epochs. "
                    f"Head LR = {dapt_lr:.1e} (overrides --dapt-lr={args.dapt_lr:.1e})."
                )

            best_dapt_ckpt = run_training_phase(
                model, train_loader, val_loader, device,
                args.dapt_epochs, dapt_lr, args.dapt_weight_decay, fold_ckpt_dir, logger,
                dapt_phase_name, scaler=scaler,
                use_segmentation=dapt_use_seg and not args.linear_probe,
                use_det_seg=use_det_seg and not args.linear_probe,
                accumulation_steps=args.accumulation_steps,
                model_type=args.model_type,
                monitor_window=args.monitor_rolling_window,
                seg_loss_weight=args.seg_loss_weight,
                bbox_loss_weight=args.bbox_loss_weight,
                warmup_epochs=args.dapt_warmup_epochs,
                warmup_start_lr=args.warmup_start_lr,
                freeze_backbone_epochs=dapt_freeze_epochs,
                mixup_alpha=args.mixup_alpha,
                bag_dropout=float(getattr(args, "bag_dropout", 0.0)),
                train_fn=dapt_train_fn, validate_fn=dapt_validate_fn,
                metrics_path=metrics_path,
            )
            current_checkpoint = best_dapt_ckpt

            # DAPT test-set inference: clean read of DAPT generalization before
            # BigLunge fine-tune perturbations.
            logger.info(f"\n{'='*60}\nRunning DAPT Test Set Inference\n{'='*60}")
            if best_dapt_ckpt and os.path.isfile(best_dapt_ckpt):
                model.load_state_dict(torch.load(best_dapt_ckpt, map_location=device))
                logger.info(f"Loaded best DAPT checkpoint for DAPT test inference: {best_dapt_ckpt}")
            else:
                logger.warning(
                    "No best DAPT checkpoint on disk — running DAPT test inference on last-epoch weights."
                )
            dapt_test_phase = f"dapt_test_fold_{fold_idx}" if n_folds > 1 else "dapt_test"
            _run_test_inference(
                model=model,
                test_loader=dapt_test_loader,
                device=device,
                logger=logger,
                validate_fn=dapt_validate_fn,
                metrics_path=metrics_path,
                output_dir=args.output_dir,
                model_type=args.model_type,
                phase=dapt_test_phase,
                prob_file_suffix=f"dapt_fold{fold_idx}" if n_folds > 1 else "dapt",
            )

        # --- PHASE 2: FINETUNE ---
        if args.mode in ["full", "finetune"]:
            is_already_mil = isinstance(
                model, (MILResNet50Classifier, MILSwinTinyClassifier, MILSwinV2BaseClassifier, MILSwinV2TinyClassifier)
            )
            if pipeline == "mil" and not is_already_mil:
                # Should not be reached with the current factory (get_sclc_model
                # always builds a MIL model for mil_* types), but kept as a
                # safety net in case a checkpoint from an older run is loaded.
                logger.warning(
                    "[MIL] Model is not a MIL type before finetune — this is unexpected. "
                    "Proceeding with current model weights."
                )
            else:
                if args.mode == "finetune" and args.model_checkpoint:
                    sd = torch.load(args.model_checkpoint, map_location=device)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    # Route DAPT-side checkpoints through backbone transfer.
                    if isinstance(model, MILResNet50Classifier) and any(
                        k.startswith("backbone.features.") for k in sd.keys()
                    ):
                        model.load_backbone_from_dapt(sd, logger=logger)
                    elif isinstance(model, (MILSwinTinyClassifier, MILSwinV2BaseClassifier, MILSwinV2TinyClassifier)) and any(
                        k.startswith("swin.") for k in sd.keys()
                    ):
                        model.load_backbone_from_dapt(sd, logger=logger)
                    else:
                        model.load_state_dict(sd)
                    logger.info(f"Loaded model checkpoint for fine-tuning: {args.model_checkpoint}")
                elif current_checkpoint and current_checkpoint != args.initial_checkpoint and os.path.isfile(current_checkpoint):
                    model.load_state_dict(torch.load(current_checkpoint, map_location=device))
                    logger.info(f"Loaded DAPT checkpoint for fine-tuning: {current_checkpoint}")
                else:
                    logger.info("Fine-tuning from initial in-memory weights (no DAPT checkpoint).")

            logger.info(f"Setting up FineTuning Datasets from: {args.finetune_dataset}")
            train_loader, val_loader, test_loader = create_dataloaders(
                args, "big_lunge", args.finetune_dataset, args.finetune_csv,
                depth_size=args.depth_size, phase="finetune",
                cv_fold=cv_fold,
            )

            best_finetune_ckpt = run_training_phase(
                model, train_loader, val_loader, device,
                args.finetune_epochs, args.finetune_lr, args.weight_decay, fold_ckpt_dir, logger,
                "finetune", scaler=scaler,
                use_segmentation=finetune_use_seg,
                use_det_seg=use_det_seg,
                accumulation_steps=args.accumulation_steps,
                model_type=args.model_type,
                monitor_window=args.monitor_rolling_window,
                differential_lr=True,
                backbone_lr_scale=args.finetune_backbone_lr_scale,
                seg_loss_weight=args.seg_loss_weight,
                bbox_loss_weight=args.bbox_loss_weight,
                warmup_epochs=args.finetune_warmup_epochs,
                warmup_start_lr=args.warmup_start_lr,
                freeze_backbone_epochs=int(getattr(args, "finetune_freeze_backbone_epochs", 0)),
                mixup_alpha=args.mixup_alpha,
                bag_dropout=float(getattr(args, "bag_dropout", 0.0)),
                train_fn=ft_train_fn, validate_fn=ft_validate_fn,
                metrics_path=metrics_path,
            )

            if args.mode == "full":
                if best_finetune_ckpt and os.path.isfile(best_finetune_ckpt):
                    model.load_state_dict(torch.load(best_finetune_ckpt, map_location=device))
                    logger.info("Loaded best FineTune checkpoint for inference.")
                elif best_dapt_ckpt and os.path.isfile(best_dapt_ckpt):
                    if not isinstance(model, (MILResNet50Classifier, MILSwinTinyClassifier, MILSwinV2BaseClassifier, MILSwinV2TinyClassifier)):
                        model.load_state_dict(torch.load(best_dapt_ckpt, map_location=device))
                        logger.info("Loaded best DAPT checkpoint for final inference (no finetune ckpt).")
                    else:
                        logger.warning(
                            "[MIL] No fine-tune checkpoint; MIL model retains DAPT weights. "
                            "Test metrics reflect DAPT-only adaptation."
                        )

                test_phase = f"test_fold_{fold_idx}" if n_folds > 1 else "test"
                fold_metrics = _run_test_inference(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    logger=logger,
                    validate_fn=ft_validate_fn,
                    metrics_path=metrics_path,
                    output_dir=args.output_dir,
                    model_type=args.model_type,
                    phase=test_phase,
                    prob_file_suffix=f"_fold{fold_idx}" if n_folds > 1 else "",
                )
                if n_folds > 1:
                    fold_test_results.append(fold_metrics)

    # Aggregate CV metrics into a single 'test' row after all folds.
    if n_folds > 1 and fold_test_results and args.mode == "full":
        _save_cv_aggregate_metrics(fold_test_results, metrics_path, args.model_type, n_folds, logger)

    # --- PHASE 3: INFERENCE (standalone --mode inference only) ---
    if args.mode == "inference":
        logger.info(f"\n{'='*60}\nStarting Inference Phase\n{'='*60}")

        if args.model_checkpoint:
            sd = torch.load(args.model_checkpoint, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # Route DAPT-side checkpoints through backbone transfer; load
            # full MIL checkpoints with strict=False.
            if isinstance(model, MILResNet50Classifier) and any(
                k.startswith("backbone.features.") for k in sd.keys()
            ):
                model.load_backbone_from_dapt(sd, logger=logger)
            elif isinstance(model, (MILSwinTinyClassifier, MILSwinV2BaseClassifier)) and any(
                k.startswith("swin.") for k in sd.keys()
            ):
                model.load_backbone_from_dapt(sd, logger=logger)
            else:
                model.load_state_dict(sd, strict=False)
            logger.info(f"Loaded model checkpoint from: {args.model_checkpoint}")
        else:
            logger.info("No --model-checkpoint provided for inference mode. Using initial weights.")

        logger.info("Setting up Test Datasets...")
        _, _, test_loader = create_dataloaders(
            args, "big_lunge", args.finetune_dataset, args.finetune_csv,
            depth_size=args.depth_size, phase="inference",
        )

        _run_test_inference(
            model=model,
            test_loader=test_loader,
            device=device,
            logger=logger,
            validate_fn=ft_validate_fn,
            metrics_path=metrics_path,
            output_dir=args.output_dir,
            model_type=args.model_type,
            phase="test",
            prob_file_suffix="",
        )
            
    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
