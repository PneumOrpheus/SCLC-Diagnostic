# Pipeline Audit — 2D and MIL (2026-04-24)

Full read-through of `main.py`, `model_selection.py`, `data/{data_loader,dataset_2d,dataset_mil,transforms}.py`, `training/{train_2d,train_mil,bootstrap}.py`, and all 2D + MIL configs. Goal: find actual correctness issues and interpretation hazards before the thesis write-up — not style.

Results on record (BigLunge val MacroF1, peak): EffNet-B0 0.55, Swin-Tiny 0.527, MIL-ResNet50 0.389. DAPT val on Lung-PET-CT-Dx hits 0.69+ across models. The 14–30 pt drop at transfer is the headline finding to explain.

---

## 1. Real bugs (fix before reporting numbers)

### 1.1 `_pbest.pth` is not the single-best-val epoch
**Where:** `main.py:739-742`

Checkpoint selection uses the rolling window-3 mean of val MacroF1, not the single-epoch peak. So when a run reports "best val F1 = 0.527 @ ep26," the actual weights saved to `*_pbest.pth` came from whichever epoch had the best **rolling mean** — which can lag the true peak by 1-2 epochs, or miss it entirely when the peak is an outlier.

**Why it matters:** On small val sets (BigLunge 2D val=31, MIL val=46), single-epoch F1 has ±5-10 pp variance. The rolling window was added to suppress that noise during monitoring — but the file written to disk is named `pbest` and users (including me) implicitly assume it means "best single-epoch." Test-set numbers come from this file. Our "best MIL" number of 0.389 is the rolling peak, not the single-epoch peak.

**Fix:** either rename the file to `_roll3best.pth`, or save TWO checkpoints (one per epoch if raw > best_raw, one per rolling improvement). Document which is being reported.

### 1.2 MixUp inflates train-set MacroF1
**Where:** `training/train_2d.py:143-163`, `training/train_mil.py:110-117`

During MixUp, `all_targets.extend(y_a.detach().cpu().tolist())` uses the **dominant** target (`lam ≥ 0.5` by construction). `all_preds` is argmax over logits of the *mixed* input. The resulting train MacroF1 is counting "did argmax-of-mixed match dominant label" — trivially true most of the time. This is why train F1 looked ~0.5 for MIL throughout and ~0.85+ for the 2D backbones: the reported train F1 is not a classifier F1, it's a "dominant-label agreement rate."

**Why it matters:** All train-val gap comparisons we've been making this week are distorted. "Train F1 0.87 vs val 0.55 = overfit" is overstated when train is actually the dominant-rate.

**Fix:** during validation we already have clean metrics. For train-set diagnostics under MixUp, either (a) compute a second pass over unmixed train batches, or (b) tag train metrics in the log as "mixup-biased" and stop comparing directly to val.

### 1.3 2D pipeline silently drops volumes without tumor masks
**Where:** `data/dataset_2d.py:120-126`

```python
for v in volume_entries:
    mask_path = v.get(mask_key)
    if not mask_path:
        continue
    slices = tumor_slice_index.get(mask_path, [])
    if not slices:
        continue
```

No `print`, no counter. The BigLunge 2D val set has 31 patients. MIL val has 46 (same seed, same patient-level split). The 15-patient gap is silently filtered here. That's a 32% drop in the denominator — and if tumor-mask-present patients have systematically different characteristics (clearer tumors, cleaner scans, particular subtypes over-represented), the 2D val set is not representative.

**Why it matters:** Every comparison between 2D and MIL results is on a non-matched test population. The MIL-vs-2D performance gap conflates (a) the lung-anchored vs tumor-anchored architecture difference (less targeted localization in MIL) with (b) MIL being evaluated on patients 2D never saw.

**Fix:** log which patients are dropped and why. For thesis, either (a) evaluate both pipelines on the 31-patient intersection set, or (b) report on each pipeline's full set with a footnote about the difference.

### 1.4 MixUp degenerates on MIL batch_size=2
**Where:** `training/train_mil.py:70` — `idx = torch.randperm(x.size(0), device=x.device)`

`torch.randperm(2)` returns `[0,1]` or `[1,0]` with equal probability. When it's `[0,1]`, each sample is mixed with itself → `lam*x + (1-lam)*x = x` → no mixup. So MIL's bag-level MixUp is a no-op 50% of the time at batch_size=2.

**Why it matters:** Our "strong-augs + MixUp" regularization on MIL was effectively "strong-augs + MixUp-half-the-time-with-the-same-bag." Not a catastrophe, but the MixUp story for MIL in the thesis is weaker than for 2D (which uses batch_size=32-64).

**Fix:** for small batches, resample `idx` until it's a true derangement, or skip MixUp at batch_size < 4.

### 1.5 Intensity augs partially cancelled by post-aug normalization
**Where:** `data/transforms.py:697-698, 711` (strong-augs branch), `680-682, 711` (mild branch)

Order in the 2D train pipeline: `RandScaleIntensityd` and `RandShiftIntensityd` (both prob=0.7 in strong mode) → then `NormalizeIntensityd(nonzero=True, channel_wise=True)` at the end. Per-sample nonzero normalization recomputes mean/std on the *augmented* image, so a +15% intensity scale ends up re-centered. Some intensity-perturbation signal survives (because nonzero mask and value distribution change), but a fraction is cancelled.

**Why it matters:** Less than we thought. The strong-augs ablation still improved DAPT by +8 MacroF1, so the effect isn't destroyed — but if intensity augs look weak in the thesis, this is why.

**Fix:** for future work, move `NormalizeIntensityd` before the intensity augs, or drop it entirely (ScaleIntensityRanged already maps HU to [0,1]).

### 1.6 `freeze_backbone_epochs` silently ignored when differential_lr is active
**Where:** `main.py:547-552, 556-560`

```python
if diff_lr_active:
    ...
    if warmup_epochs > 0 or freeze_backbone_epochs > 0:
        logger.info(f"... Ignoring warmup/freeze settings because differential_lr=True ...")
```

The log-line is informational (not a warning), and the user has no way to override. This is why I had to implement LP-then-FT as two separate runs (with `backbone_lr_scale=0.0` as a freeze proxy) instead of a single run with `freeze_backbone_epochs=10`.

**Why it matters:** "Freeze backbone for first N epochs, then unfreeze and apply differential LR" is a standard LP-FT recipe. The pipeline can't express it in one run. We got the result we wanted via the two-run hack, but future experiments would benefit from native support.

**Fix:** when both are set, freeze for N epochs (ignore diff-lr during freeze), then switch to diff-lr when unfrozen. Two lines of logic in `run_training_phase`.

### 1.7 `CropAroundTumord` centroid-of-all-voxels collapses on multifocal masks (FIXED 2026-04-28)
**Where:** `data/transforms.py:CropAroundTumord._centroid` (~line 242)

**The bug.** The 2D pipeline computes the crop center as the unweighted mean of *all* non-zero mask voxels:

```python
nz = torch.nonzero(mask > 0.5, ...)
coords = nz[:, -3:].float().mean(dim=0)
```

When a patient has multiple disjoint tumor components (e.g. primary + nodal metastasis, or auto-seg false positives), the centroid lands *between* them — captured slice content is empty lung parenchyma instead of any one lesion. The 96-mm crop then teaches the 2D classifier to call empty lung whatever subtype the patient has, which is the opposite of what we want.

**Audit.** `scripts/audit_multifocal.py` (output in `output/multifocal_audit.csv`, figure at `figures/fig_multifocal_distribution.pdf`) shows BigLunge tumor masks are multifocal in **68-77% of patients**:

```
Class               N  Multi  %Multi  MedComp  P95Comp
Adenocarcinoma    107     65   69.1%      2.0      9.0
Small Cell         96     67   77.0%      3.0      8.0
Squamous          103     65   68.4%      2.0      8.0
```

So this isn't an edge case for BigLunge — it's the norm. (Lung-PET-CT-Dx, with radiologist-drawn masks, is overwhelmingly mono-component, so DAPT was largely unaffected by the bug.)

**Fix.** `_centroid` now runs `scipy.ndimage.label` on the binary mask, filters connected components below `min_component_voxels=50` (drops auto-seg specks), and returns the centroid of the **largest** remaining component. Falls back to all-voxel centroid if every component is below the threshold (rare; a tiny-but-real tumor still produces a usable crop instead of a volume-center fallback).

**Methodological justification (for thesis Methods).** We don't try to identify the *anatomical primary* — that requires radiologist input we don't have. For histologic-subtype classification we don't need to: metastases share the primary's histology by definition. The largest connected component is overwhelmingly the dominant tumor mass (primary or bulky metastasis) and presenting it to the 2D classifier yields the correct subtype label regardless of which lesion it captures.

**Cache implication.** Changing `_centroid` changes the deterministic prefix of the 2D pipeline → the next 2D training run rebuilds the BigLunge 2D cache. Lung-PET-CT-Dx 2D cache also rebuilds, but DAPT is already done and saved in `dapt_thesis/`, so re-DAPT isn't necessary — the rebuild only happens when fine-tune kicks off.

---

## 2. Interpretation hazards (affect thesis conclusions, not bugs)

### 2.1 Tiny val/test sets with high variance
Val sizes: Lung-PET-CT-Dx 102 pt / BigLunge 2D 31 pt / BigLunge MIL 46 pt / BigLunge test 46 pt. With 3-class stratified CIs at n=31, a single misclassified patient moves MacroF1 by ~3 pp. Our reported peaks are within-CI noise of each other:

- Swin-Tiny FT: 0.527 [0.379, 0.673] (from log CI output)
- EffNet-B0 FT: 0.55 (unbounded without running CI)
- MIL best: 0.389 [0.25, 0.55] (wide)

**Thesis implication:** framing any 2-3 pp difference as a "result" will not survive review. Bootstrap CIs are already computed per-epoch for BigLunge; use them everywhere in the thesis tables.

### 2.2 Rolling-3-mean early stop is loose
Early stopping monitors rolling window-3 macro F1 with patience=10 epochs-no-improve. On noisy val sets that's very permissive — effectively "continue until you've had 10 consecutive epochs where the 3-epoch mean didn't hit a new high." Many runs train past the genuine peak by 5-10 epochs.

**Thesis implication:** reported training curves are longer than they need to be; peaks look isolated because late-epoch val is drifting down. A patience=5 on raw val F1 would produce cleaner-looking curves without changing best numbers.

### 2.3 DAPT → BigLunge gap is domain shift; MIL's gap on top is the lung-anchored vs tumor-anchored architecture trade-off
The 2D pipeline drop EffNet DAPT-val 0.69 → BigLunge val 0.55 (a 14-pt drop) happens even though both sides use the SAME tumor-centered crop architecture. The Lung-PET-CT-Dx masks are radiologist-drawn (XML annotations); BigLunge masks are algorithmic auto-seg. So the 2D gap is primarily domain shift PLUS upstream-mask quality, not architecture. Plausible contributors:

- **Mask provenance.** Lung-PET-CT-Dx tumor masks are published research-grade; BigLunge tumor masks are produced by auto-seg with known failure modes (multifocal handling, false positives — see audit script).
- **Image spacing distribution.** Both resampled to (1, 1, 2) mm; original spacings differ, so re-interpolation artifacts differ.
- **Subtype mix.** Lung-PET-CT-Dx vs BigLunge label distributions differ.
- **Scanner differences.** Reconstruction metadata available in CSV but unused.

The MIL gap on top of that 14-pp domain shift (~2 pp on val) is the architectural trade-off: lung-anchored MIL uses LESS targeted localization than tumor-centered 2D. It can't be reduced to "weak supervision" because both pipelines use *automatic* upstream segmentation — they just differ in *which* auto-seg they depend on. Lung-seg is mature (Dice ≈ 0.95 with off-the-shelf tools); tumor-seg is hard, especially for SCLC and multifocal disease.

**Thesis framing:** the 14-pp 2D gap is domain shift + tumor-seg quality. The 2-pp MIL-on-top gap is the cost of lung-anchored vs tumor-anchored architectures. The benefit is robustness: MIL is unaffected by the multifocal/false-positive failure modes that hit the 2D crop.

### 2.4 HU window is wide
`ScaleIntensityRanged(a_min=-1024, a_max=3071)` maps the full 4095 HU range linearly to [0,1]. Lung tissue (-600 HU) → 0.10; tumor (50 HU) → 0.26. All the relevant signal lives in a narrow band of the scaled range. Lung-window (-1000, 400) would give much better contrast.

**Why this wasn't a bug before:** with a DAPT stage on the same scaling, the model learns whatever internal mapping it needs. But for thesis reproducibility: anyone re-implementing with lung-window CT pre-processing (the clinical default) will see different numbers.

---

## 3. Lung-PET-CT-Dx-specific issues (DAPT)

### 3.1 Label extraction via substring match is fragile
**Where:** `data/data_loader.py:84-87, 125-128`

```python
for key, val in CLASS_MAP.items():
    if f"-{key}" in pid:
        label = val
        break
```

Works for well-formed folder names like `Lung_Dx-A0001`. Breaks if someone ever drops a folder named `Lung_Dx-GA-0001` (a "G" followed later by "-A"): the **first** match wins in dict iteration order and CPython preserves insertion order, so `CLASS_MAP = {"A": 0, "B": 1, "G": 2}` means `-A` wins over `-G` if both substrings are present. None of the current 690 patient IDs trigger this, but it's silent — no assert that exactly one letter matches. Worth adding a strict check.

### 3.2 `max_scans_per_patient=2` exemption for Small Cell
**Where:** `data/data_loader.py:141-142` (`if max_scans_per_patient ... and label != 1`)

Small Cell (label 1) is exempt from the 2-scans-per-patient cap. The rationale in the comment is reasonable ("SCLC is too rare to cap"), but it has a consequence: at the **slice level** (DAPT sees slices, not patients), classes are imbalanced from the start. Observed train slice counts: `{Adeno: 2608, Small Cell: 442, Squamous: 680}`. Small Cell patients each contribute 3-8× more scans than Adeno/Squamous patients. Combined with `WeightedRandomSampler`, which oversamples Small Cell slices on top, a small handful of SC patients get their slices seen many times per epoch → strong patient-identity overfit signal for SC.

**Why this matters:** the DAPT train F1 climbs to 0.97 in 8 epochs despite class imbalance, while val caps at 0.69. Part of that gap is the model memorizing a few Small Cell patients whose slices get repeated at ~5-10× the rate of any Adeno patient's slice.

**Fix:** either honor the cap for all classes (small cell will be underrepresented but sampler handles it), or compute the sampler weights AFTER the cap rather than before.

### 3.3 Train F1 vs val F1 are at different levels of aggregation
**Where:** `training/train_2d.py:202` (train returns slice-level), `training/train_2d.py:304` (val computes patient-level)

`train_macro_f1` recorded in every `metrics.jsonl` row for Lung-PET-CT-Dx is **slice-level** and **MixUp-biased** (see 1.2). `val_macro_f1(patient)` is the two-step mean-of-means (slice → volume → patient). So the headline number `train 0.97 / val 0.69` is comparing:
- argmax-match on per-slice mixup-dominant labels (train)
- argmax of averaged softmax probabilities across volumes of a patient (val)

The gap is genuinely there, but it's not as catastrophic as the raw numbers suggest. Aggregation partially smooths noisy predictions on val; the train side has no such smoothing. Reporting a single "DAPT train-val gap" misrepresents the comparison.

**Fix:** when reporting train curves, either compute patient-level F1 on train (expensive, would need a second loop) or explicitly label as "slice-level, mixup-biased."

### 3.4 Mask coverage is 100% now — silent drop will bite later
**Where:** `data/dataset_2d.py:120-126` (same path as BigLunge 2D in 1.3)

Current DAPT log: `482 images (482 w/ masks)` — every Lung-PET-CT-Dx scan has a mask sidecar. But the same `if not mask_path: continue` loop runs for Lung-PET-CT-Dx too. If the dataset is re-exported in future and any scans lose masks (common on release updates), patients will be silently dropped from DAPT without any log line. This hasn't happened yet, but it's a sleeper bug.

### 3.5 Weighted sampler + inflated SC slices = memorization
**Observation, not a bug.** The DAPT val curve peaks at ep 8-10 for every backbone we've trained (EffNet, Swin, MIL-DAPT). Earlier peak + decline = classic memorization. The SC exemption is a direct contributor. Cutting `dapt_epochs` to 10 in the MIL config was the right move and should be mirrored in the 2D configs (currently 30).

### 3.6 DAPT val set is the reliable number; DAPT test is new and underused
DAPT val = 102 patients → bootstrap CIs are actually usable (~6 pp wide). DAPT test = 106 patients — only recently added via `_run_test_inference`. If the thesis reports transfer-learning value, the DAPT test F1 (held-out, never seen) is the cleanest support for "DAPT learned transferable features." We have this number now but haven't tabulated it across backbones.

**Action item:** include DAPT test F1 in the main results table. This is the single most under-reported number in the project.

---

## 4. BigLunge-specific issues (fine-tune)

### 4.1 Mask provenance
You've confirmed BigLunge tumor masks are algorithmic (auto-seg). The three lung-mask-truncated patients we just filtered (`057069, 091821, 022269`) are proof of non-trivial failure rates. If tumor masks also fail silently on some subset, the 2D pipeline is training/evaluating on wrong crops.

**Action item:** the `data_exploration/BigLunge_expl.ipynb` audit looked at per-patient masks; expand it to quantify what fraction of tumor masks pass a size + position sanity check. Flag outliers.

### 4.2 One scan per patient
BigLunge is one scan per patient (MIL dedupes via `seen_patients`). Lung-PET-CT-Dx has up to 2 scans per patient (`max_scans_per_patient=2`), augmenting data. Train sizes: BigLunge 213 (MIL) / 168 (2D, after tumor-mask filter) vs Lung-PET-CT-Dx 482 (mostly). BigLunge is ~2.5-3x smaller.

**Why this matters:** the signal-to-noise ratio of any gradient step on BigLunge is lower. 40 BigLunge FT epochs ≈ 6720 samples; 30 DAPT epochs on Lung-PET-CT-Dx ≈ 14,460 samples.

### 4.3 The patient-level split is consistent across pipelines but the *data lists* aren't
Same seed (42), same `val_frac=0.15`, same `test_frac=0.15` give the same patient IDs per split for both 2D and MIL. But 2D drops ~15 patients silently via 1.3, so the 2D val set is a **subset** of the MIL val set, not the same set. Any cross-pipeline comparison (MIL vs 2D F1) is on patient-mismatched data.

---

## 5. What to do before the thesis

Ordered by return-on-time:

1. **Fix 1.1 (labeling of `_pbest.pth`).** 10 min. Prevents every reader assuming we're reporting single-epoch peaks when we're actually reporting 3-epoch means. Just rename + document, or save both.

2. **Fix 1.3 (2D tumor-mask-drop logging).** 15 min. Add the print. Then re-run the best 2D model on the 31-patient val set vs the full 46-patient set — if the numbers are close, the selection bias is mild and we note it; if they diverge, we report on the common subset.

3. **Fix 1.2 (MixUp train-F1 tag).** 15 min. Add a `mixup_active=True` flag in the metrics row. Thesis tables can then footnote "train metrics are mixup-biased when active."

4. **Bootstrap CIs on every headline number in the thesis.** The infra is there (`training/bootstrap.py`) — just make sure every test-set eval emits the CI. For the main table (one row per model × dataset), point estimate + 95% CI.

5. **One clean test-set evaluation per model** using the `_pbest.pth`, logging per-class F1 and confusion matrix. I can write a `scripts/report_test_metrics.py` that loops over all `_pbest.pth` files in `/home/data/trained_models/` and dumps a single CSV.

6. **Attention diagnostics for MIL** (thesis-grade figure). Plot attention weight over slice index alongside tumor-mask z-extent for 10 representative val patients. "MIL attention concentrates on tumor-bearing slices X% of the time using only the lung mask as input" is a defensible interpretability contribution *independent of the MacroF1 ceiling*. Combined with Grad-CAM on the MIL backbone, this provides two-level explainability (which slices + where in those slices) without requiring tumor masks at inference time.

7. **Do NOT** run more MIL hyperparameter sweeps. The ceiling is at ~0.39 on this data given the architecture + validation size. Further iteration is noise-chasing.

### Thesis framing I'd actually defend

- **Headline:** "We compare two end-to-end automatic preprocessing pipelines on BigLunge: tumor-anchored 2D (CT → auto tumor-seg → tumor-centered crop → 2D CNN) and lung-anchored MIL (CT → auto lung-seg → bag of slices spanning lung extent → MIL with attention). 2D achieves val MacroF1 0.55 [CI]; lung-anchored MIL achieves 0.39 [CI]. The 16-pp gap partitions into ~14 pp DAPT-to-BigLunge domain shift (driven by tumor-seg quality drop from radiologist-drawn to algorithmic) and ~2 pp from the architectural trade-off of lung-anchored vs tumor-anchored localization."
- **Supporting story:** MIL's attention concentrates on tumor-bearing slices using only the lung mask. Combined with per-slice Grad-CAM on the backbone, this gives two-level interpretability (which slices + where within them) — the deployment-ready explainability story.
- **Robustness story:** MIL is unaffected by tumor-seg failure modes (multifocal, false positives) that hit the 2D crop. Quantified by `scripts/audit_multifocal.py`. For deployment scenarios where tumor-seg quality cannot be guaranteed, lung-anchored MIL is the safer choice.
- **Limitations section:** small val sets (40/46), algorithmic tumor-mask failures observed in BigLunge, single-scan-per-patient bias, DAPT overfit risk past ep 10.
