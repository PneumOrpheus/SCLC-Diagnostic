# Pipeline Flaws — SCLC 3D Classification

_Audit target: `main.py`, `model_selection.py`, `data/data_loader.py`, `data/transforms.py`, `training/train.py`, plus the mask generator `data_exploration/create_masks_2.py` and the cached artifacts under `~/.cache/monai_lung_pet_ct_clean/`._

Ordered by severity. The first two items are, on their own, sufficient to explain "no model is learning anything." Everything below them is real but secondary — fix #1 and #2 before you touch the rest.

---

## 1. (CRITICAL) Your training set is silently frozen at 12 samples with ZERO Small Cell Carcinoma

This is the headline bug. It is not a subtle one. It is the one that is breaking everything.

`data/data_loader.py::create_dataset` caches, for each split, a file called `valid_data.json` inside `~/.cache/monai_lung_pet_ct_clean/<mode_key>_img{img_size}_d{depth_size}/<split>/`. Once that JSON exists, the loader takes the fast path (line ~283):

```python
if os.path.exists(valid_data_file) and not warm_cache:
    ...
    ds = PersistentDataset(data=valid_data, transform=transforms, cache_dir=current_cache_dir)
```

It never re-reads `data_list`, never re-scans `/home/data/Lung-PET-CT-Dx-Clean`, and never checks whether the cache was written in `--testing` mode or in a normal run. The cache is also keyed only by `img_size` and `depth_size`, not by `testing`, not by `val_frac`/`test_frac`, not by `seed`, and not by any hash of the dataset list.

I inspected your current caches on disk:

```
~/.cache/monai_lung_pet_ct_clean/3d_img224_d64/{train,val,test}   →  12 / 12 / 12   (pure --testing leftovers)
~/.cache/monai_lung_pet_ct_clean/3d_img224_d128/train             →  12   (STALE --testing leftover)
~/.cache/monai_lung_pet_ct_clean/3d_img224_d128/val               →  78   (full split — rebuilt later)
~/.cache/monai_lung_pet_ct_clean/3d_img224_d128/test              →  81   (full split — rebuilt later)
```

The `3d_img224_d128/train/valid_data.json` contains exactly these 12 entries:

- 6 × Adenocarcinoma (class 0), drawn from 3 patients (A0064, A0246, A0250)
- 6 × Squamous (class 2), drawn from 2 patients (G0032, G0059)
- **0 × Small Cell Carcinoma (class 1)**

At some point you ran with `--testing`, which caps each split at 12 entries (`data_loader.py` lines 119–122). That run wrote `valid_data.json` with 12 entries for train. Later you changed something that caused `val`/`test` to be rebuilt fresh (my guess is those cache directories got deleted manually, or the first run crashed before hitting them), but the train cache file survived. Every run since then has silently trained on those same 12 non-SCLC samples.

### Everything you are seeing in the logs falls out of this

Look at `output/2026-04-13_12_logs.txt` and `output/2026-04-13_11_logs.txt`:

- Every epoch reports `Epoch N [6/6]`. Six batches. Batch size 2. That is literally 12 samples per epoch. The old log `Best-so-far-med-double-dipping2026-03-31_19_logs.txt` reported `[N/434]` — 434 batches per epoch — which is what the pipeline actually looks like when the cache is healthy.
- The class weight print says `Class Weights: [0.6666667 4.        0.6666667]`. That is not a real inverse-frequency weight. It is what `compute_class_weights` falls back to when class 1 has zero members: `class_counts.get(cls_idx, 1)` treats the missing class as "1 sample" and spits out `total / (num_classes * 1) = 12 / 3 = 4.0`. The "4.0" is a ghost — it is the default value for a class that is entirely absent from the dataset.
- Not a single confusion matrix across 29 epochs ever predicts Small Cell. Not one. The model cannot learn a class it has never seen. It literally does not know what an SCLC scan looks like.
- The best val accuracy you hit (≈78%) is exactly `61 / 78` — it is the "always predict Adenocarcinoma" ceiling on your 78-sample val set (61 Adeno + 6 SCLC + 11 Squamous). The model isn't learning; it's just discovering that guessing "adeno" is the best it can do when the only training signal splits evenly between adeno and squamous.

### Why this is fundamental and not just unlucky

There is no invalidation logic anywhere in the pipeline. No checksum of `data_list`, no version counter, no testing-flag suffix in the cache path, no CLI flag for "force rebuild" short of `warm_cache=True` (which is hardcoded `False` in `create_dataloaders`). If a teammate picks this repo up next week and runs `--testing` once to make sure imports work, their next real run will silently train on 12 samples and they will spend a week debugging the loss curve.

### What to actually do

1. **Immediately**: `rm -rf ~/.cache/monai_lung_pet_ct_clean/`. Re-run. Sanity-check the first epoch header says something closer to `[400+/400+]`. Re-check that the class weight print shows realistic values like `[0.47, 3.29, 1.82]` (that is what the 2026-03-31 run had).
2. **Harden the cache**: include `testing` in the cache directory name (e.g., `3d_img224_d128_testfull` vs `3d_img224_d128_test12`). Alternatively, write a small `meta.json` next to `valid_data.json` containing the `data_list` length, the `testing` flag, `val_frac`, `test_frac`, `seed`, and the number of patients; on load, compare it to the current run and force rebuild on mismatch.
3. **Make `--testing` less dangerous**: do not write `valid_data.json` at all in testing mode, or write it under a `_testing/` subdirectory. Testing mode should be non-destructive to real caches.
4. **Add a `--clear-cache` flag** to `main.py` so you never have to remember the raw cache path.
5. **Assert non-zero per-class counts** in `main.py` right after the dataset is built. If class 1 has zero training samples, abort with a clear message rather than silently training a broken model.

Until you do this, **nothing else in this document matters**. Every other fix is downstream of having a real training set.

---

## 2. (CRITICAL) `class_weights` is computed, logged, and then never actually used

`main.py::run_training_phase` calls `compute_class_weights`, prints the result to the log, and passes the tensor into `train_epoch` via the `class_weights=class_weights` keyword argument. `train_epoch` accepts it in the signature. Then, in `training/train.py` line 68:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

That's it. The argument is never read. The criterion is constructed with no `weight=` argument. The same is true in `validate_epoch`. The entire imbalance-weighting story in the README (and in the function signatures) is cosmetic — the loss function has never seen those weights.

On Lung-PET-CT-Dx (≈462 adeno : 102 squamous : 72 SCLC volumes) this would turn an unweighted cross-entropy into a strong bias toward predicting adenocarcinoma. On BigLunge, which you've said is also imbalanced, there is no `WeightedRandomSampler` and no class weighting either, so the fine-tuning phase runs with no imbalance correction whatsoever.

Worse: `label_smoothing=0.1` smooths the minority-class targets from `[0, 1, 0]` to `[0.033, 0.933, 0.033]`, which further damps the gradient the model receives from the already-rare SCLC samples. Label smoothing on a dataset with a 6× imbalance, without class weighting, is actively harmful for the minority class.

### Fix

```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights,          # actually use the tensor
    label_smoothing=0.1,
)
```

And decide whether you want label smoothing and class weighting and a weighted sampler simultaneously. In my opinion, pick two: a `WeightedRandomSampler` during DAPT plus `CrossEntropyLoss(weight=…)` during fine-tune is plenty. Stacking all three on top of label smoothing is overcorrecting and makes the effective loss very hard to reason about.

---

## 3. (HIGH) DAPT rebalances with a sampler; fine-tuning on BigLunge does not rebalance at all

In `main.py::create_dataloaders`, the `WeightedRandomSampler` branch is gated on `dataset_type == "lung_pet_ct_dx"`:

```python
if dataset_type == "lung_pet_ct_dx" and hasattr(train_ds, "data") and len(train_ds.data) > 0:
    ...
    train_loader = DataLoader(train_ds, ..., sampler=sampler, ...)
else:
    train_loader = DataLoader(train_ds, ..., shuffle=True, ...)
```

BigLunge hits the `else` branch and trains on the raw, imbalanced distribution with vanilla `shuffle=True`. No sampler, no class weight (see flaw #2), no focal loss, nothing. For a thesis target that is specifically "does SCLC show up better than chance," fine-tuning without any imbalance handling is a big blind spot. You should either (a) build an equivalent `WeightedRandomSampler` for BigLunge, (b) weight the loss (flaw #2), or (c) both.

Note: once flaw #2 is fixed, the loss is weighted everywhere for free. That is probably the lowest-friction fix.

---

## 4. (HIGH) Orientation normalization is disabled, so "depth" isn't always anatomical Z

In both `get_train_transforms_3d` and `get_val_transforms_3d`, `Orientationd(axcodes="RAS")` is commented out with a note that some files have malformed affines. This is a load-bearing comment in a way you may not appreciate:

- Different scanners and different dcm2niix conversions produce volumes whose in-memory axis order is not standardized. For most Lung-PET-CT-Dx scans, axis -1 does correspond to the axial/superior–inferior direction, but not for all of them. `data_exploration/eda_summary.csv` has entries with z-spacing ≈5 mm and x/y spacing ≈0.8 mm (clearly axial-last), but there is no guarantee every single volume in the cleaned set matches.
- Your `ExtractSubVolumed` transform in `data/transforms.py` unconditionally crops along `volume.shape[-1]`, assuming the last axis is craniocaudal. When a scan happens to be stored with a different axis order, the "depth" slab you extract is actually a sagittal or coronal slab centered on a sagittal or coronal midpoint. For that patient, the spatial context that reaches the model is anatomically different from every other patient's.
- `Spacingd(pixdim=(1.5, 1.5, 2.0))` has the same problem: it assumes axis 2 is the axis with the large z-spacing. Without `Orientationd`, it is resampling the wrong axis for any transposed patient.
- Random flips on all three axes (`RandFlipd`) are heavy-handed and can mask this inconsistency during training, but they do not fix it — and at validation time, where there are no flips, the mismatched patients reach the model in whatever arbitrary orientation they were stored.

The note says "malformed affines" but the right response to "some affines are broken" is to _fix those specific volumes_ (or exclude them) rather than to disable orientation normalization for the entire cohort. It is strictly safer to `Orientationd(axcodes="RAS", allow_missing_keys=True)` with a try/except around the whole transform pipeline in the cache-validation loop that drops volumes whose affine fails, than to leave the transform off.

If you keep `Orientationd` disabled, at minimum log a warning in `create_dataset` whenever a volume's affine doesn't match the expected RAS-ish convention, and have `ExtractSubVolumed` pick the crop axis by finding the axis whose voxel spacing is largest (i.e. the axial axis) instead of hard-coding `-1`.

---

## 5. (HIGH) The segmentation auxiliary loss leaks into samples that have no real mask

In `training/train.py` lines 88–93:

```python
if masks is not None and torch.sum(masks) > 0:
    masks = masks.float()
    seg_loss = nn.functional.binary_cross_entropy_with_logits(seg_outputs, masks)
    loss = cls_loss + (0.5 * seg_loss)
```

And in `simple_collate_fn` lines 46–55, items without a `mask` key get an all-zero tensor substituted at collate time so the batch can stack.

Now imagine a BigLunge-style batch, or any batch that mixes scans that have a mask with scans that don't: `torch.sum(masks) > 0` is true (because _some_ samples have a real mask), so `seg_loss` is computed for the whole batch, including the samples whose "ground truth mask" is a fake tensor of zeros. For those samples, BCE is actively teaching the encoder: "for this CT, there is no tumor anywhere." That is a corrupted gradient pushing the model toward predicting empty segmentations, and it propagates back through the same encoder that your classification head depends on. It is a silent poison term.

For the current Lung-PET-CT-Dx-Clean run this may not bite you because every sample in the clean set has a real mask. But the DAPT-then-finetune design assumes the seg-loss path will coexist with BigLunge, where masks are absent entirely. As soon as you do a mixed run, or as soon as one bad mask slips through as all-zeros, the encoder gets that corrupted signal.

### Fix

Carry a per-sample `has_mask` boolean through the collate and only compute the segmentation loss on the masked subset of the batch:

```python
has_mask = torch.tensor([("mask" in item) for item in batch], dtype=torch.bool)
# ...
if has_mask.any():
    seg_loss = F.binary_cross_entropy_with_logits(
        seg_outputs[has_mask], masks[has_mask]
    )
    loss = cls_loss + 0.5 * seg_loss
else:
    loss = cls_loss
```

Also reject masks that are all-zero after transforms (they contribute nothing and they indicate an upstream failure worth seeing).

---

## 6. (MEDIUM) No class stratification in train/val/test splits

`data/data_loader.py` does a patient-level random shuffle with `np.random.default_rng(seed=42)` and then slices by fraction. With only ~72 SCLC patients total, a 10% val fraction gives something like 7 SCLC val patients in expectation, and basic variance means a realistic run can give you 4 or 10. Combined with flaw #1 (training is already stuck on 12 samples with zero SCLC), your validation signal for SCLC is also noisy enough that even a correctly trained model could look unstable from one run to the next.

Use a stratified split. `sklearn.model_selection.StratifiedShuffleSplit` over the per-patient label is a one-liner and guarantees every split contains every class.

---

## 7. (MEDIUM) `RandScaleIntensityd` and `RandShiftIntensityd` run with prob=1.0

In `data/transforms.py::get_train_transforms_3d`:

```python
RandScaleIntensityd(keys=[...], factors=0.1, prob=1.0),
RandShiftIntensityd(keys=[...], offsets=0.1, prob=1.0),
```

Probability 1.0 means "apply to every training sample, every epoch." Combined with the `NormalizeIntensityd` that runs afterward to re-standardize, this is not catastrophic, but it is unusual and worth thinking about. For classification you usually want some training samples to pass through unperturbed so the model sees the real intensity distribution. 0.3–0.5 is a more typical setting.

---

## 8. (MEDIUM) `best_ckpt` can be referenced before assignment

In `main.py::run_training_phase`:

```python
best_acc = 0.0
...
if acc > best_acc:
    best_acc = acc
    best_ckpt = os.path.join(...)
    torch.save(...)
...
return best_ckpt
```

`best_ckpt` is only defined inside the `if`. If accuracy never strictly exceeds 0.0 across the entire phase — which can happen when the pipeline is in the broken state described in flaw #1 — then `best_ckpt` is undefined and the function raises `UnboundLocalError` at return time, aborting the DAPT → fine-tune transition. Initialize `best_ckpt = None` at the top of the function and have the caller handle the "no improvement ever" case explicitly.

Similarly, `create_dataloaders` returns `test_loader` from the finetune branch but is shadowed by the earlier DAPT assignment in `full` mode; the inference phase's `if 'test_loader' not in locals()` check relies on Python scoping subtleties that are easy to get wrong when you refactor. Prefer explicit passing.

---

## 9. (MEDIUM) No gradient clipping under AMP

`train_epoch` uses `torch.amp.autocast` and a `GradScaler`, but never calls `torch.nn.utils.clip_grad_norm_`. With a 3D model, a large effective batch from accumulation, and label-smoothed cross-entropy, an early-epoch NaN/inf is not rare. When the scaler's `step` detects inf it skips the optimizer step silently — which looks exactly like "the model isn't learning" in the log. Add:

```python
if scaler is not None:
    scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

before the `scaler.step(optimizer)` (or plain `optimizer.step()`) call, inside the accumulation-boundary branch.

---

## 10. (MEDIUM) `Resized` target differs between train and val

Train uses `spatial_size=(img_size, img_size, depth_size)`. Val uses `spatial_size=(img_size, img_size, -1)`. The `-1` means "don't touch this axis." Because `ExtractSubVolumed` is run before `Resized` and always forces depth to `depth_size`, the two code paths happen to produce the same shape today. But this is coincidence, not design. If anyone ever rearranges the transform order or changes `ExtractSubVolumed`, val and train will silently start producing differently-shaped tensors, and val batches will refuse to stack. Make the two configs identical, or drop `ExtractSubVolumed` from the val path and let `Resized` do the resizing by itself — but don't leave them subtly misaligned.

---

## 11. (LOW) The default `--initial-checkpoint` is a SwinUNETR checkpoint, loaded for every model

`main.py` defaults `--initial-checkpoint` to `/home/data/temp/model_swin_unetr_btcv_segmentation_v1.pt`, and `get_sclc_model` is called unconditionally with it regardless of `--model-type`. For `resnet50` and the new `vit` path, the function happily runs `torch.load(...)` on this swin checkpoint and then calls `load_state_dict(state_dict, strict=False)` against a ResNet or a ViT. Zero keys match, every load is a no-op, and the log prints "Pretrained weights loaded successfully" as if something had happened. It hasn't.

Either make `--initial-checkpoint` model-type-aware (one default per architecture), or skip the load when the checkpoint filename clearly belongs to a different architecture, or at minimum log how many keys actually matched:

```python
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"[*] loaded {len(state_dict) - len(unexpected)} / {len(state_dict)} keys")
```

---

## 12. (LOW) `AddPlaceholderTargetsd` silently labels unlabeled samples as class 0

In `data/transforms.py::AddPlaceholderTargetsd`:

```python
if "scan_label" not in d:
    d["scan_label"] = 0
```

Today `data_list` always sets `scan_label` upstream in `get_lung_pet_ct_dx_data_list` / `get_biglunge_data_list`, so this branch isn't hit. But if an upstream change ever drops that field — for example, a new dataset loader that forgets it — every affected sample becomes a silent Adenocarcinoma without warning. A placeholder class label is a footgun. Assign `-1` or raise, not `0`.

---

## 13. (LOW) `ResNetClassifier` uses `resnet50(pretrained=True)` but MedicalNet weights expect a specific preprocessing

Your pipeline ends with `NormalizeIntensityd` producing zero-mean unit-variance volumes on top of a [0, 1] intensity rescale. MedicalNet (the source of the `pretrained=True` weights MONAI downloads for `resnet50`) was pretrained with a different intensity normalization scheme on MrBrainS / other datasets. This mismatch is not a _bug_ — it's a transfer-learning caveat — but it means the "pretrained" prior is weaker than you may be assuming. Worth a sentence in the thesis.

---

## 14. (LOW) Hooks on `swin_unetr.swinViT` keep references alive across forward passes

`SwinUNETRClassifier` registers a forward hook that stashes the encoder's deepest features on `self.deepest_features`. When `return_segmentation=False`, the classifier also calls `self.swin_unetr.swinViT(x.contiguous())` directly, which fires the same hook. The hook is harmless but redundant: two computation paths both populate the same attribute. More importantly, the attribute holds a reference to the feature tensor past the end of `forward`, which can pin it in memory if anything else keeps a reference to the model. Explicitly `self.deepest_features = None` at the top of each `forward`.

---

## Summary, in plain English

You are not training a 3D classifier on Lung-PET-CT-Dx. You are training it on six adenocarcinoma scans and six squamous scans from five patients, with no SCLC examples at all, and telling the optimizer via label smoothing that no class should be too confidently predicted. The loss function was supposed to be class-weighted but silently isn't. The validation set has the right distribution, so the confusion matrices you see are a genuine measurement of what the model has learned — and what it has learned is "predict adeno, never predict SCLC, sometimes predict squamous when it's having a good day." That behavior is not a modeling failure. It is the mathematically correct response to the training set the pipeline is actually feeding it.

Fix #1 and #2, then re-run a single DAPT epoch and check that (a) the batch count is in the hundreds, (b) the class weight line reports realistic non-placeholder values, and (c) the confusion matrix contains non-zero entries in the SCLC column within the first 3–5 epochs. If any of those three signals are still wrong, come back to this document and work down the list.
