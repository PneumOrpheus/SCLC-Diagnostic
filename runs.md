# Training Runs — SCLC 3D Classification

Chronological log of SwinUNETR runs with configuration deltas and outcomes. See `flaws.md` for pipeline-level analysis and the reasoning behind each change.

Checkpoints live in `/home/data/trained_models/`. Naming: `{DD_MM}_{model}_{phase}_best.pth` for the best-by-rolling-val-macro-F1 checkpoint per phase. Phase-specific tag was introduced mid-Round 2, so early runs use the `best_swin_unetr_{phase}_new.pth` scheme.

---

## Round 1 — 2026-04-14

### ResNet18 (`resnet_2026-04-14_08_logs.txt`)
- DAPT best: `/home/data/trained_models/best_resnet18_dapt_new.pth`
- Fine-tune best: `/home/data/trained_models/best_resnet18_finetune_new.pth`
- Test macro-F1: **0.1429** (random)
- Notes: Both critical bugs present — DAPT seg loss dominated cls, fine-tune collapsed at unfreeze epoch.

### SwinUNETR (`swin_unetr_2026-04-14_12_logs.txt`)
- DAPT best: `/home/data/trained_models/best_swin_unetr_dapt_new.pth` (14_04 07:54)
- Fine-tune best: `/home/data/trained_models/best_swin_unetr_finetune_new.pth` (14_04 17:58)
- Test macro-F1: **0.2600**
- Notes: Same two failure modes. Flatlined last 8 epochs at `MacroF1=0.3411`.

---

## Round 2 — 2026-04-15 (fixes #4, #5, E.2)

**Landed:** differential LR for fine-tune (backbone_lr = head_lr × 0.1), `--seg-loss-weight=0.1` (was 0.5 effective), train macro-F1 metric, rolling-3 val selection.

### SwinUNETR (`swin_unetr_2026-04-15_06_logs.txt`)
- DAPT best: `/home/data/trained_models/15_04_swin_unetr_dapt_best.pth` (replaced in Round 3 — same filename)
- Fine-tune best: `/home/data/trained_models/best_swin_unetr_finetune_new.pth` (15_04 10:48)
- DAPT val rolling-3 best: 0.2813 (ep8)
- Fine-tune val rolling-3 best: ~0.34 band
- Notes: Both bugs confirmed fixed. **DAPT classifier is genuinely learning** (train F1 0.35→0.68 over 28 epochs). New failure mode: **DAPT overfits after ~ep8** — train climbs, val plateaus/declines. Scan-diversity ceiling hypothesis.
- User manually interrupted fine-tune at ep27 (flat loss).

---

## Round 3 — 2026-04-15 (Overfit-1 mitigations)

**Landed:** `--dapt-weight-decay=1e-2` (was 1e-3), stronger train augmentation in `data/transforms.py` (`RandAffined` p=0.5, `RandGaussianNoised` p=0.3, `RandCoarseDropoutd` p=0.3, intensity augs raised to p=0.5), `max_scans_per_patient=2` in Lung-PET-CT-Dx loader, ViT disabled, `--anno` warn for non-SwinUNETR.

Data split shrank: train 538→478, val 125→102, test 132→106 images.

### SwinUNETR (`swin_unetr_2026-04-15_11_logs.txt`)
- DAPT best: `/home/data/trained_models/15_04_swin_unetr_dapt_best.pth` (15_04 12:35, overwrote Round 2)
- Fine-tune best: `/home/data/trained_models/15_04_swin_unetr_finetune_best.pth` (15_04 13:22)
- DAPT val rolling-3 best: 0.2368 (ep12)
- Fine-tune val rolling-3 best: 0.3097 (ep19)
- Test accuracy: **0.3714**
- Notes: Overfitting fixed — **but overshot into under-fitting.** Train F1 now plateaus 0.30–0.44 in both phases (Round 2 DAPT reached 0.68). Train/val gap closed but both low. Early stopped fine-tune at ep30.
- Regularization dial went too far in one shot (four levers at once).

---

## Round 4 — 2026-04-15 (dial-back)

**Landed:** `--dapt-weight-decay` 1e-2 → **3e-3**, removed `RandCoarseDropoutd` (20×20×10 cutouts on sparse tumors were erasing lesions). Kept scan cap, `RandAffined`, `RandGaussianNoised`, raised intensity augs.

### SwinUNETR (`swin_unetr_2026-04-15_14_logs.txt`)
- DAPT best: `/home/data/trained_models/15_04_swin_unetr_dapt_best.pth` (overwrote Round 3)
- Fine-tune best: `/home/data/trained_models/15_04_swin_unetr_finetune_best.pth` (overwrote Round 3)
- DAPT val rolling-3 best: 0.2404 (ep6)
- DAPT train F1 peak: 0.40 (ep10) — target was 0.5–0.6, **still under-fitting**
- Fine-tune val rolling-3 best: 0.2589 (ep21)
- Fine-tune early stopped at ep31 (10 epochs no improvement)
- Test accuracy: **0.3429** | Test macro-F1: **0.2706**
- Per-class test F1: Adeno=0.41, **SCLC=0.00**, Squamous=0.40
- Notes: Dial-back did not improve over Round 3. **Small Cell completely missed** (0/10 correct). Val F1 extremely noisy (12 SCLC val samples). Train/val gap reopening in fine-tune (0.44 vs 0.26). Regularization tuning alone is insufficient — the core issue is likely sample size and SCLC class poverty.

---

## Round 5 — 2026-04-16 (structural changes)

**Landed:** Classification head simplified from `Linear(768→256)→GELU→Dropout(0.3)→Linear(256→3)` (197K params) to single `Linear(768→3)` (2.3K params). DAPT epochs 15→**30**. SCLC (class 1) exempted from `max_scans_per_patient=2` cap — keeps all scans for the rare class. `--mode finetune` now works standalone from BTCV weights.

### Round 5 Hyperparameter Snapshot (baseline before Round 6)

- Run mode: `full`
- Model: `swin_unetr`
- Seed: `42`
- AMP: enabled (`--disable-amp` not set)
- Batch size: `2`
- Accumulation steps: `4` (effective batch size `8`)
- Depth size: `128`
- Num workers: `4`
- PET channel: disabled (`--use-pet` not set)
- Segmentation aux: enabled (`--anno`) with `--seg-loss-weight=0.1`
- Monitoring metric: rolling val macro-F1 with `--monitor-rolling-window=3`
- Early stopping: patience `10` epochs without rolling macro-F1 improvement

- DAPT epochs: `30`
- DAPT optimizer: AdamW
- DAPT LR: `1e-4`
- DAPT weight decay: `3e-3`
- DAPT scheduler: cosine annealing (`T_max=30`)
- DAPT sampling: `WeightedRandomSampler` (inverse class frequency on train split)
- DAPT scan cap: `max_scans_per_patient=2` for classes 0 and 2; class 1 (SCLC) uncapped

- Fine-tune epoch budget: `40` (early stopped at epoch `26`)
- Fine-tune optimizer: AdamW with differential LR
- Fine-tune head LR: `3e-5`
- Fine-tune backbone LR: `3e-6` (`--finetune-backbone-lr-scale=0.1`)
- Fine-tune weight decay: `1e-3`
- Fine-tune scheduler: cosine annealing (`T_max=40`)

- Classification loss: CrossEntropy with label smoothing `0.1`
- Aux segmentation loss (when mask exists): `0.5 * BCEWithLogits + 0.5 * Dice`
- Combined train loss (masked samples): `cls_loss + 0.1 * seg_loss`

- Train augmentation (3D): flip x/y/z each `p=0.5`; `RandAffined p=0.5` (`rotate_range=(0.1,0.1,0.1)`, `translate_range=(8,8,4)`, `scale_range=(0.1,0.1,0.1)`)
- Intensity augmentation: `RandScaleIntensity p=0.5 (factors=0.1)`, `RandShiftIntensity p=0.5 (offsets=0.1)`, `RandGaussianNoised p=0.3 (std=0.01)`
- Round 5 kept `RandCoarseDropoutd` removed (dropped in Round 4)

### SwinUNETR (`swin_unetr_2026-04-16_06_logs.txt`)
- DAPT best: `/home/data/trained_models/best_so_far_16_04/16_04_swin_unetr_dapt_best.pth`
- Fine-tune best: `/home/data/trained_models/best_so_far_16_04/16_04_swin_unetr_finetune_best.pth`
- DAPT val rolling-3 best: **0.3950** (ep30) — still climbing at end, up from Round 4's 0.2404
- DAPT train F1 peak: 0.48 (ep28–30) — healthier than Round 4's 0.40
- DAPT val balanced-acc: reached **0.5135** (ep29) — first time above chance for all 3 classes
- Fine-tune val rolling-3 best: 0.2980 (ep16)
- Fine-tune early stopped at ep26 (10 epochs no improvement)
- Test accuracy: **0.5143** | Test macro-F1: **0.4740** | Balanced accuracy: **0.4831**
- Per-class test F1: Adeno=0.40, **SCLC=0.375** (3/10 correct), Squamous=0.65
- Confusion matrix: SCLC → 3 correct, 3 predicted Adeno, 4 predicted Squamous
- Notes: **Best run by far.** Test macro-F1 nearly doubled (0.27→0.47). SCLC is now being detected (was 0/10, now 3/10). The longer DAPT was the biggest factor — val F1 was still climbing at ep30, suggesting even more epochs could help. Fine-tune plateau is still tight (0.28–0.31 val F1 band). DAPT encoder features are clearly better.

---

## Round 6 — 2026-04-16 (long DAPT-only trial, no other changes)

**Goal tested:** Extend DAPT length while keeping Round 5 setup intact.

**Command used:**
`python main.py --mode dapt --model-type swin_unetr --dapt-epochs 60 --dapt-lr 1e-4 --dapt-weight-decay 3e-3 --seg-loss-weight 0.1 --monitor-rolling-window 5 --seed 42`

### SwinUNETR (`swin_unetr_2026-04-16_10_logs.txt`)
- Mode: DAPT only (no fine-tune, no test inference)
- DAPT best checkpoint: `/home/data/trained_models/16_04_swin_unetr_dapt_best.pth`
- DAPT val rolling-5 best: **0.2574** (epoch 16)
- DAPT val current macro-F1 best: **0.3111** (epoch 16)
- DAPT train macro-F1 peak: **0.4810** (epoch 24)
- Periodic checkpoints written: epoch 10 and epoch 20
- Early stopping: triggered at **epoch 26** (not epoch 28)

Notes:
- This run underperformed Round 5 DAPT (`rolling-3 best 0.3950` at epoch 30).
- Early stopping at epoch 26 is expected from the configured logic: the best rolling score happened at epoch 16, then there were 10 consecutive non-improving epochs (17–26), so patience=10 terminated training.
- The larger monitor window (`roll5` vs `roll3`) smooths peaks more aggressively and can make best-score updates harder on a noisy small validation set.

### Analysis

**Headline:** Longer DAPT did **not** help. The Round 5 hypothesis ("val F1 still climbing at ep30, more epochs should help") does not hold up.

**Training trajectory (nearly identical to Round 5):**
- Train MacroF1: 0.39 (ep1) → 0.42 (ep5) → 0.44 (ep18) → **0.48 peak at ep24**. Round 5 peaked at 0.48 around ep28–30. The encoder is learning at the same rate and plateauing at the same place. There is no "more headroom if we train longer" signal on the training side either.
- Train loss drifts from 1.19 → 1.07. Modest but continuous. This is not a model that is silent on the data — it is slowly fitting.

**Validation is the real story:**
- Single-epoch val MacroF1 swings wildly between 0.066 and 0.311 across consecutive epochs (e.g. ep22=0.0655 right after ep21=0.1523 and before ep24=0.2273). This is the expected noise profile for a 102-sample val set with 12 SCLC — one or two flipped predictions move the score by 0.05–0.10.
- Rolling-5 peaks at **0.2574 (ep16)** and monotonically drifts down afterwards: 0.239, 0.245, 0.245, 0.235, 0.203, 0.180, 0.159, 0.151, 0.156, 0.177. Train MacroF1 is still climbing over this same window (0.43 → 0.48). That is the classic overfitting shape.
- Best **single-epoch** val in this run is 0.3111 (ep16). Round 5's rolling-3 best of 0.3950 came from three consecutive good epochs at the end of the run. Replaying Round 6 on a rolling-3 window would score roughly (0.2635+0.2730+0.3111)/3 ≈ **0.282** at ep14–16 — better than rolling-5's 0.2574 but still meaningfully below Round 5's 0.3950. So the monitor-window change explains **part** of the delta, not most of it.

**What Round 5's 0.3950 probably was:** a lucky 3-epoch window at the tail, not a genuine trajectory. The underlying encoder generalization on Lung-PET-CT-Dx val looks capped in the ~0.25–0.30 rolling band given current data/augmentations. Extending the budget to 60 epochs exposed that ceiling — after ep16 the model gets worse on val while still improving on train.

**Implications:**
1. **DAPT val macro-F1 is a poor checkpoint-selection signal** on this val set. It is noisy enough that "best rolling epoch" is largely a function of where the 3/5-epoch lucky window lands, not where the encoder is actually most transferable.
2. **30 epochs is probably already too many**, not too few. The useful portion of DAPT on this setup looks like roughly the first 15–20 epochs; beyond that train keeps rising and val drifts down.
3. **Don't tune DAPT hyperparameters against DAPT val.** The only metric that matters is downstream BigLunge test performance after fine-tune. Two candidate DAPT checkpoints (e.g. ep16 and ep26) can differ a lot on Lung-PET-CT-Dx val and very little on BigLunge test — or vice versa.
4. The monitor-window change (`roll3` → `roll5`) should be reverted for comparability with Rounds 2–5, or at minimum logged alongside both values.

**Suggested next steps:**
- Run the already-implemented `--mode finetune` (BTCV → BigLunge, no DAPT) to measure DAPT's actual contribution. If finetune-only matches or beats Round 5, DAPT on Adeno-heavy Lung-PET-CT-Dx is at best neutral for BigLunge transfer.
- If continuing with DAPT, cap `--dapt-epochs` at ~20 and stop treating DAPT val F1 as a quality signal — use it only for picking *among* candidates, not as the goal.
- Consider evaluating two or three DAPT checkpoints (e.g. the periodic ep10/ep20 saves plus the "best" one) through the same fine-tune and comparing downstream test scores. That is the only way to disentangle DAPT encoder quality from DAPT val noise.

---

## Round 7 — 2026-04-17 (finetune-only: BTCV → BigLunge, no DAPT)

**Goal tested:** Does DAPT on Lung-PET-CT-Dx actually help BigLunge fine-tune, or is BTCV a strong enough starting point on its own? Uses the newly-wired `--mode finetune` standalone path.

**Command used:**
`python main.py --mode finetune`

### SwinUNETR (`swin_unetr_2026-04-17_07_logs_finetune_only.txt`)
- Starting weights: BTCV segmentation (`model_swin_unetr_btcv_segmentation_v1.pt`), loaded in-memory by `get_sclc_model()`
- Fine-tune best checkpoint: `/home/data/trained_models/17_04_swin_unetr_finetune_best.pth`
- Fine-tune val rolling-3 best: **0.4145 (epoch 1)** — peaks immediately, drifts down thereafter
- Fine-tune val rolling-3 trajectory: ep1=0.4145, ep2=0.3941, ep3=0.3850, ep4=0.3784, ep5=0.3714, ep6=0.3514, ep7=0.3382, ep8=0.3463, ep9=0.3782, ep10=0.3970, ep11=0.3929
- Train MacroF1: 0.26–0.38 band across all 11 epochs, no clear upward trend (peak 0.3676 at ep2)
- Train/val loss both stuck at ~1.09–1.11 (3-class CE with label smoothing 0.1 has a floor near 1.03)
- Early stopped at ep11 (10 epochs no rolling improvement since ep1)
- Periodic ep10 checkpoint saved: `17_04_swin_unetr_finetune_epoch_10.pth`
- **No test inference was run** — the current `--mode finetune` path ends after fine-tune without triggering the test loader

### Analysis

**Headline:** Inconclusive on the DAPT-vs-BTCV question, but revealing about the training dynamics.

**Side-by-side with Round 5's fine-tune phase:**

| | val rolling-3 best | best epoch | train F1 peak | n train batches | test F1 |
|---|---|---|---|---|---|
| Round 5 (DAPT → finetune) | 0.2980 | ep16 | ~0.44 | same loader | 0.4740 |
| Round 7 (BTCV → finetune) | **0.4145** | **ep1** | ~0.38 | same loader | not measured |

On the surface Round 7's val looks better. Read carefully and it's the opposite of a good training curve:
- The **peak is at epoch 1**. The model's best state is essentially "BTCV backbone + a single-epoch-trained linear head." Nine more epochs of training make it slightly worse.
- Train F1 never rises. There is no fitting signal — the optimizer is not meaningfully moving either the backbone (LR 3e-6) or the head.
- Round 5's fine-tune, in contrast, actually learned: train F1 climbed to ~0.44 and val slowly approached its ep16 peak.

Combined with the 32-sample val set (12 Adeno, 9 SCLC, 11 Squamous), the 0.4145 number is almost certainly **noise on a favourable random init of the head**, not a real advantage over DAPT. A single SCLC flip moves val F1 by ~3%.

**What this run does establish:**
1. The `--mode finetune` standalone path works end-to-end from BTCV weights — no key-mismatch errors, checkpoint saves, early stopping triggers correctly.
2. The **differential LR is too conservative for a cold-start finetune.** With `backbone=3e-6, head=3e-5` the backbone is effectively frozen and the head saturates within one epoch. This LR schedule was tuned for the DAPT → finetune handoff (where the backbone has already seen the domain) and is not appropriate when starting from BTCV.
3. The **test inference step is missing from the finetune-only path** — a gap in the current `main.py` control flow. Round 5's test F1 only exists because `--mode full` chains through inference; `--mode finetune` doesn't.

**What this run does *not* establish:**
- Whether DAPT helps or hurts downstream test performance. Without the test pass, the only comparable number is val F1 on a 32-sample set, which is too noisy to trust for small differences.

**Suggested next steps (but hold until the ~300-patient BigLunge arrives):**
- Add a test-inference pass to `--mode finetune`, or add a `--mode test --pretrained-checkpoint <path>` entry point, so this checkpoint can be scored on the same test split as Round 5.
- If we do rerun the BTCV-vs-DAPT A/B on current data, raise the fine-tune backbone LR for the BTCV starting point (e.g. backbone=1e-5, head=1e-4) — otherwise we're not measuring "BTCV features" against "DAPT features," we're measuring "frozen BTCV + 1-epoch head" against "fully fine-tuned DAPT model."
- With 32 val samples, tuning decisions on this split should have roughly the confidence interval of a coin flip. Defer the real A/B until the expanded BigLunge provides a val set where a ~0.03 F1 gap actually means something.
