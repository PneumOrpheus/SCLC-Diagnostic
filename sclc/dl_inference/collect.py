"""Collect per-patient penultimate-layer features + softmax probabilities for
every saved DL model on every relevant split.

Output structure: ``results/dl_features/<model_type>/<dataset>_<split>.npz``
containing:

  * ``features`` : (N, D) penultimate-layer activations, one row per patient
    (slice-level outputs are mean-aggregated across each patient's slices to
    match the validator's patient-level aggregation).
  * ``probs``    : (N, 3) softmax outputs.
  * ``patient_ids`` : (N,) string array.
  * ``true_labels`` : (N,) int array.
  * ``model_type``, ``dataset``, ``split`` : str scalars (metadata).

Datasets/splits collected per pipeline:

  * 2D / MIL (LPCT-Dx + BigLunge):
      - ``lpcd_val``, ``lpcd_test`` from the model's ``dapt_pbest_raw`` checkpoint
      - ``biglunge_val``, ``biglunge_test`` from ``finetune_pbest_raw``
  * 3D (swin_unetr): same.

CLI:
    python -m sclc.dl_inference.collect --models efficientnet_b0_2d swin_tiny_2d
    python -m sclc.dl_inference.collect              # all models
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

logging.getLogger("monai").setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROV_DIR = REPO_ROOT / "results" / "thesis"
OUT_DIR = REPO_ROOT / "results" / "dl_features"
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"
CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]


# ---------- model-type → pipeline / config / classifier head ---------------

PIPELINE_BY_MODEL: Dict[str, str] = {
    "efficientnet_b0_2d":  "2d",
    "densenet121_2d":      "2d",
    "resnet50_2d":         "2d",
    "swin_tiny_2d":        "2d",
    "resnet50_2d_rin":     "2d",
    "densenet121_2d_rin":  "2d",
    "mil_resnet50":        "mil",
    "mil_swin_tiny":       "mil",
    "swin_unetr":          "3d",
}

CONFIG_BY_MODEL: Dict[str, str] = {
    "efficientnet_b0_2d":  "2d_efficientnet_b0.yaml",
    "densenet121_2d":      "2d_densenet121.yaml",
    "resnet50_2d":         "2d_resnet50.yaml",
    "swin_tiny_2d":        "2d_swin_tiny.yaml",
    "resnet50_2d_rin":     "2d_resnet50_rin.yaml",
    "densenet121_2d_rin":  "2d_densenet121_rin.yaml",
    "mil_resnet50":        "mil_resnet50.yaml",
    "mil_swin_tiny":       "mil_swin_tiny.yaml",
    "swin_unetr":          "3d_swin_unetr.yaml",
}

# Path to each model's final classification linear layer. We register a
# forward-pre-hook on this layer to capture the penultimate-layer feature
# vector that gets fed into it. ``getattr`` walks dot paths.
HEAD_PATH: Dict[str, str] = {
    "efficientnet_b0_2d":  "efficientnet._fc",
    "densenet121_2d":      "densenet.class_layers.out",
    "resnet50_2d":         "backbone.fc",
    "swin_tiny_2d":        "swin.head.fc",
    "resnet50_2d_rin":     "classification_head",
    "densenet121_2d_rin":  "classification_head",
    "mil_resnet50":        "mil.myfc",
    "mil_swin_tiny":       "mil.myfc",
    "swin_unetr":          "classification_head",
}


def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    mod = root
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


# ---------- args + loaders ---------------------------------------------------

def _build_args(model_type: str) -> argparse.Namespace:
    """Run main.parse_args with --config <model_type yaml> + --testing False to
    populate every argparse default for this model."""
    from sclc.main import parse_args
    config_path = CONFIG_DIR / CONFIG_BY_MODEL[model_type]
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    import sys
    saved = sys.argv
    sys.argv = ["dl_inference", "--config", str(config_path), "--mode", "inference"]
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    args.model_type = model_type
    return args


def _build_loaders(args, dataset_type: str, phase: str):
    """Wrap main.create_dataloaders so we don't accidentally drop class-weights
    setup. Returns (train_loader, val_loader, test_loader)."""
    from sclc.main import create_dataloaders
    if dataset_type == "lung_pet_ct_dx":
        data_path = args.dapt_dataset
        csv_path = ""
    else:
        data_path = args.finetune_dataset
        csv_path = args.finetune_csv
    depth = int(getattr(args, "depth_size", 64))
    return create_dataloaders(args, dataset_type, data_path, csv_path=csv_path,
                              depth_size=depth, phase=phase)


# ---------- forward + hook + per-patient aggregation -----------------------

def _capture_features(
    model: nn.Module, loader, model_type: str, device: torch.device,
    pipeline: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """Forward every batch through ``model`` while capturing the penultimate
    feature vector. Returns three dicts keyed by patient_id:

      * features[pid] : mean penultimate vector across slices/bag/volume.
      * probs[pid]    : mean softmax across slices/bag/volume.
      * label[pid]    : ground-truth class.
    """
    head = _resolve_module(model, HEAD_PATH[model_type])
    captured: List[torch.Tensor] = []
    def pre_hook(_mod, inp):
        # inp is a tuple; the first element is the (B, D) input to the head.
        x = inp[0]
        captured.append(x.detach().cpu())
    handle = head.register_forward_pre_hook(pre_hook)

    feats_per_pid: Dict[str, List[np.ndarray]] = defaultdict(list)
    probs_per_pid: Dict[str, List[np.ndarray]] = defaultdict(list)
    label_per_pid: Dict[str, int] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            captured.clear()
            x = batch["image"].to(device)
            y = batch["label"].cpu().numpy().astype(int)
            pids = batch.get("patient_id")
            if pids is None:
                # 3D dataset stores patient on each item via image path; fall
                # back to image basename.
                pids = [Path(p).stem for p in batch.get("image_path", [""] * len(y))]
            elif isinstance(pids, torch.Tensor):
                pids = [str(int(p)) for p in pids.tolist()]

            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()

            if not captured:
                raise RuntimeError(f"Hook captured nothing for {model_type}")
            feats = captured[0].numpy()  # (B, D)
            # MIL: B=1 per bag, but feats may already be (B, D); shape OK.
            for i, pid in enumerate(pids):
                feats_per_pid[pid].append(feats[i])
                probs_per_pid[pid].append(probs[i])
                label_per_pid[pid] = int(y[i])

    handle.remove()

    feat_out = {pid: np.mean(np.stack(v), axis=0) for pid, v in feats_per_pid.items()}
    prob_out = {pid: np.mean(np.stack(v), axis=0) for pid, v in probs_per_pid.items()}
    return feat_out, prob_out, label_per_pid


def _save_npz(
    path: Path, feats: Dict[str, np.ndarray], probs: Dict[str, np.ndarray],
    labels: Dict[str, int], model_type: str, dataset: str, split: str,
) -> None:
    pids = sorted(feats.keys())
    F = np.stack([feats[p] for p in pids]) if pids else np.zeros((0, 0))
    P = np.stack([probs[p] for p in pids]) if pids else np.zeros((0, 3))
    Y = np.array([labels[p] for p in pids], dtype=int)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=F, probs=P, patient_ids=np.array(pids), true_labels=Y,
        model_type=model_type, dataset=dataset, split=split,
    )
    print(f"  [save] {path.name} — N={len(pids)} D={F.shape[1] if F.size else 0}")


# ---------- per-model driver ------------------------------------------------

def _provenance_checkpoints(model_type: str) -> Dict[str, Optional[str]]:
    """Return {dapt_pbest_raw, finetune_pbest_raw} from the model's per-model
    provenance.json (which build_thesis_results.py rebuilds)."""
    pipeline = PIPELINE_BY_MODEL[model_type]
    p = PROV_DIR / pipeline / "per_model" / model_type / "_provenance.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        prov = json.load(f)
    ck = prov.get("checkpoints", {}) or {}
    return {
        "dapt_pbest_raw":     ck.get("dapt_pbest_raw"),
        "finetune_pbest_raw": ck.get("finetune_pbest_raw"),
    }


def collect_one(model_type: str, device: torch.device) -> None:
    from sclc.models import get_sclc_model
    pipeline = PIPELINE_BY_MODEL[model_type]
    args = _build_args(model_type)
    args.batch_size = 1   # for clean per-patient mapping; 3D + MIL tolerate this
    args.num_workers = 2
    args.testing = False
    args.clear_cache = False

    out_dir = OUT_DIR / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = _provenance_checkpoints(model_type)

    # ---- LPCT-Dx (val + test) via DAPT checkpoint ------------------------
    dapt_ckpt = ckpts.get("dapt_pbest_raw")
    if dapt_ckpt and Path(dapt_ckpt).is_file():
        print(f"[collect] {model_type}: LPCT-Dx using DAPT ckpt {dapt_ckpt}")
        model = get_sclc_model(checkpoint_path=dapt_ckpt, model_type=model_type)
        model = model.to(device).eval()
        _, val_loader, test_loader = _build_loaders(args, "lung_pet_ct_dx", phase="dapt")
        for split, loader in [("val", val_loader), ("test", test_loader)]:
            feats, probs, labels = _capture_features(model, loader, model_type, device, pipeline)
            _save_npz(out_dir / f"lpcd_{split}.npz", feats, probs, labels,
                      model_type, "lpcd", split)
        del model; torch.cuda.empty_cache()
    else:
        print(f"[collect] {model_type}: no DAPT checkpoint → skip LPCT-Dx")

    # ---- BigLunge (val + test) via FT checkpoint -------------------------
    ft_ckpt = ckpts.get("finetune_pbest_raw")
    if ft_ckpt and Path(ft_ckpt).is_file():
        print(f"[collect] {model_type}: BigLunge using FT ckpt {ft_ckpt}")
        model = get_sclc_model(checkpoint_path=ft_ckpt, model_type=model_type)
        model = model.to(device).eval()
        # MIL inference uses the bag dataset; route via phase=inference.
        phase = "inference" if pipeline == "mil" else "finetune"
        _, val_loader, test_loader = _build_loaders(args, "big_lunge", phase=phase)
        for split, loader in [("val", val_loader), ("test", test_loader)]:
            feats, probs, labels = _capture_features(model, loader, model_type, device, pipeline)
            _save_npz(out_dir / f"biglunge_{split}.npz", feats, probs, labels,
                      model_type, "biglunge", split)
        del model; torch.cuda.empty_cache()
    else:
        print(f"[collect] {model_type}: no FT checkpoint → skip BigLunge")


# ---------- top-level -------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None,
                   help="Optional list of model_types (default: all).")
    p.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto).")
    args = p.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = args.models or list(PIPELINE_BY_MODEL)
    for mt in todo:
        if mt not in PIPELINE_BY_MODEL:
            print(f"[collect] unknown model_type {mt}; skip")
            continue
        try:
            collect_one(mt, device)
        except Exception as exc:
            print(f"[collect] FAILED for {mt}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
