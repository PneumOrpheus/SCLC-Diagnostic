# 2D Augmentation Experiment — Reference

This document records the 2D per-slice augmentation changes introduced to
address the train-val gap observed across the first three 2D DAPT runs
(EffNet-B0, ResNet50, DenseNet121). It is intended as a thesis reference:
**what** changed, **why** it was changed, and **how** it maps onto the
command line.

---

## Motivation

Across all three 2D backbones the same overfitting signature appeared on
the Lung-PET-CT-Dx DAPT val split (n=52 patients):

| Model | Train MacroF1 (peak) | Val Pat MacroF1 (best-roll) | Gap |
|---|---|---|---|
| EffNet-B0  | ~0.87 | 0.737 | 0.13 |
| ResNet50   | ~0.98 | 0.737 | 0.24 |
| DenseNet121| ~0.89 | 0.609 | 0.28 |

The loss function, LR, and sampler strategy had already been fixed under
Fixes 0–3 (sampler-only class balancing, patient-level checkpoint monitor,
rolling=3, LR warmup). The remaining dominant signal is overfitting — the
training set memorizes 3730 slices across 15-30 epochs while the validation
score plateaus or drifts downward after epoch 5–10.

Two complementary levers address this directly:

1. **Stronger per-sample augmentation** — increase the effective diversity
   each slice contributes, especially important under the
   `WeightedRandomSampler` where each Small-Cell slice is replayed ~6×
   per epoch.
2. **MixUp** — batch-level regularization that discourages the model from
   memorizing individual (slice → label) mappings by training on convex
   combinations instead.

Both are opt-in via CLI / YAML flags; the default pipeline is unchanged
so prior runs remain reproducible.

---

## Changes at a glance

### Augmentation block (`data/transforms.py:_build_2d_pipeline`)

The deterministic prefix (`Load → Orientation → Spacing → SliceSelect →
CropAroundTumor → ScaleIntensityRange → Resize → SqueezeDim`) is
**unchanged**. Only the random block differs, and MONAI's
`PersistentDataset` cache does not need to be rebuilt (the cache is keyed
on the last deterministic transform).

| Transform | Mild (default, `--strong-augs` off) | Strong (`--strong-augs`) |
|---|---|---|
| `RandFlipd` (axis 0) | p=0.5 | p=0.5 |
| `RandFlipd` (axis 1) | p=0.5 | p=0.5 |
| `RandAffined` — prob | 0.5 | **0.8** |
| `RandAffined` — rotate | ±15° (rad 0.26) | **±20° (rad 0.35)** |
| `RandAffined` — translate | (8, 8) px | **(12, 12) px** |
| `RandAffined` — scale | ±10% | **±15%** |
| `RandScaleIntensityd` | factors=0.10, p=0.5 | **factors=0.15, p=0.7** |
| `RandShiftIntensityd` | offsets=0.10, p=0.5 | **offsets=0.15, p=0.7** |
| `RandGaussianNoised` | std=0.01, p=0.3 | **std=0.02, p=0.3** |
| `RandGaussianSmoothd` | — | **σ_x,σ_y ∈ [0.5, 1.0], p=0.2** |
| `RandCoarseDropoutd` | — | **holes=3, size=(24, 24), p=0.3** |

Normalization (`NormalizeIntensityd`) and tensor conversion run after the
random block and are unchanged.

### MixUp (`training/train_2d.py`)

New helper `_mixup_batch(x, y, alpha)` and a `mixup_alpha` kwarg on
`train_epoch_2d`. When `alpha > 0`:

```
lam ~ Beta(alpha, alpha), then lam ← max(lam, 1 − lam)
x_mix = lam · x + (1 − lam) · x[permuted]
loss  = lam · CE(logits, y_a) + (1 − lam) · CE(logits, y_b)
```

The `lam ← max(lam, 1 − lam)` flip keeps `y_a` as the larger-weighted
label, so the running train-F1 logged during training remains
interpretable as "how well we predict the dominant label of each mix"
instead of flipping identity mid-epoch.

Validation is **unaffected** — MixUp is a training-only signal.

When `mixup_alpha = 0.0` (default) the mix step is a no-op (`lam = 1.0`,
no index permutation) so the training loop is byte-identical to the
pre-MixUp behavior. This is the fallback used by all prior experiments.

---

## CLI / YAML surface

Two flags on `main.py`:

| Flag | Default | Effect |
|---|---|---|
| `--strong-augs` | off | Swap the mild aug block for the strong block described above. Only the 2D pipeline consumes this flag; 2.5D and 3D ignore it. |
| `--mixup-alpha FLOAT` | `0.0` | Beta(α, α) shape for MixUp. `0` disables. Recommended starting point: `0.2`. Applies to both DAPT and fine-tune phases when set. |

YAML equivalents (for `configs/experiments/2d_*.yaml`):

```yaml
training:
  strong_augs: true
  mixup_alpha: 0.2
```

---

## Recommended first experiment (EffNet-B0 only)

From `runs_2d.md` / `fixes_2d.md`, the plan was to validate the
intervention on EffNet-B0 first before applying to the other two
backbones. Commands:

```bash
# DAPT with strong augs + MixUp α=0.2
python main.py \
    --config configs/experiments/2d_efficientnet_b0.yaml \
    --mode dapt \
    --strong-augs \
    --mixup-alpha 0.2
```

The saved checkpoint lands at:

```
/home/data/trained_models/2d/efficientnet_b0_2d/{stamp}_dapt_pbest.pth
```

### What to look for

The intervention is **working** if:

- **Train MacroF1 drops meaningfully** (from ~0.87 to ~0.75-0.80).
  MixUp at α=0.2 specifically pulls the training score down —
  that's the mechanism.
- **Val MacroF1 holds or rises** (≥ 0.737 rolling, ideally with
  tighter 95% bootstrap CI than prior run's [0.61, 0.83]).
- **Train-val gap compresses** from 0.13 toward ≤ 0.08.

The intervention is **not working** if:

- Train MacroF1 barely moves — augs aren't being applied (verify by
  inspecting `effective_config.yaml`), or the model is ignoring the
  perturbations.
- Val MacroF1 drops by > 0.05 — over-regularization; drop
  `--mixup-alpha` to 0.1 or ablate `RandCoarseDropoutd` first.
- Epoch-1 train loss > 2.0 — usually a stem/BN pre-existing issue
  resurfacing, not augmentation. Check the EffNet stem-init memory
  before blaming augs.

### Why not apply to all three backbones at once

All three models currently occupy different points in the overfitting
regime (DenseNet is the most overfit, ResNet50 the most unstable, EffNet
the most stable). The augmentation intervention interacts with the BN /
DropConnect / weight-init behavior of each backbone differently —
running all three in parallel would bundle three separate ablations into
one experiment and hide which backbone benefits most. Confirm the
mechanism works on EffNet first, then apply the same config to ResNet50
and DenseNet as independent replications.

---

## Follow-ups to consider (not implemented)

- **Label smoothing sweep** (0.05 vs 0.10). Currently fixed at 0.10
  across all runs; smoothing interacts with MixUp (both are
  distribution-softening regularizers) and could be tuned down when
  MixUp is active.
- **RandCoarseDropout sizing**. `spatial_size=(24, 24)` on a 224×224
  image masks ~1.1% of area per hole × 3 holes = ~3.5% per dropout
  event. May need to grow to `(32, 32)` for ResNet50 (larger effective
  receptive field).
- **MixUp on val disabled — confirm.** The current implementation only
  calls `_mixup_batch` from `train_epoch_2d`; `validate_epoch_2d` is
  untouched. If MixUp ever sneaks into val, validation numbers become
  meaningless — re-check after any training-loop refactor.
- **CutMix** (not MixUp) is a natural follow-on. CutMix replaces a
  rectangular region of image A with a region of image B, rather than
  a global convex blend. For 2D CT with spatially localized tumors,
  CutMix can better preserve textural signal than MixUp. Would be a
  third lane in the augmentation ablation.

---

## File inventory

| File | Change |
|---|---|
| `data/transforms.py` | `_build_2d_pipeline` accepts `strong_augs`; new `RandGaussianSmoothd` + `RandCoarseDropoutd`; `get_train_transforms_2d` takes `strong_augs`. |
| `data/dataset_2d.py` | `create_dataset_2d` takes and forwards `strong_augs`. |
| `training/train_2d.py` | `_mixup_batch` helper; `train_epoch_2d` takes `mixup_alpha`. |
| `training/train.py` | `train_epoch` gains `**_unused` so 3D pipeline tolerates the new kwarg. |
| `main.py` | `--strong-augs` + `--mixup-alpha` CLI flags; plumbed into `create_dataloaders` and `run_training_phase` for both DAPT and fine-tune. |
