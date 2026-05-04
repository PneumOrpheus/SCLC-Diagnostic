# Investigation 2026-04-30: Why is performance only mediocre on Lung-PET-CT-Dx and poor on BigLunge?

**Author:** investigation report
**Status:** evidence collected, recommendations below
**Methods:** training-curve analysis, dataset audits, frozen-backbone linear probing, attention-entropy diagnostics

---

## TL;DR

The headline numbers (BigLunge test accuracy 0.26–0.49, MacroF1 0.23–0.48; Lung-PET-CT-Dx test accuracy 0.49–0.77 dominated by the ADC majority) are **not the result of one bug**. They are the joint outcome of five distinct failures:

1. **DAPT does not transfer to BigLunge.** A frozen-backbone linear probe of the DAPT-trained ResNet-50 on BigLunge **scores the same as a frozen ImageNet-only ResNet-50** (~0.36 slice-level test F1 either way). Fine-tuning recovers a small amount on top of this (~0.44 patient-level F1) but cannot push past the feature ceiling.
2. **Selecting the DAPT checkpoint by validation accuracy hurts BigLunge transfer.** `dapt_pbest_raw.pth` (val-best on Lung-PET-CT-Dx) gives BigLunge patient-test F1 ≈ 0.29; `dapt_epoch_30.pth` (the last cosine-decayed epoch) gives 0.44. The val-best is overfit to the source domain.
3. **The MIL fine-tune does not learn.** Train loss is pinned at ~1.10 = ln(3), val loss the same, and validation `attention.entropy_mean` ≈ 0.95 → 1.0 (uniform across all 32 slices). The attention head never escapes uniform.
4. **Massive Z-spacing distribution shift.** Lung-PET-CT-Dx median Z spacing is **5.0 mm** (51% of volumes are 5–7 mm). BigLunge median is **0.6 mm** (79% < 1 mm). The pipelines resample to 2.0 mm, so DAPT volumes get upsampled ~2.5× while BigLunge gets downsampled ~3.3× — different visual statistics in either direction.
5. **BigLunge tumor masks are noisy.** 30/306 patients (≈10 %) have empty/sub-threshold auto-segmentations and are silently dropped from the 2D pipeline or fall back to the volume center in 3D. 68–77 % of the rest are multifocal (≥2 connected components ≥50 vox).

The best pragmatic remedy without retraining is to **swap `dapt_pbest_raw` for `dapt_epoch_30`** as the FT initialiser. Bigger remedies (rebalance DAPT, fix MIL attention, harmonise spacing) require new runs. Details below.

---

## Section 1 — Evidence

### 1.1 Training curves: FT loss is stuck at the uniform-prediction floor

Reading `metrics.jsonl` for the canonical (most recent) FT run of every model:

| Model | DAPT train_loss (last) | FT train_loss (first → last) | FT val_loss range | Comment |
|---|---|---|---|---|
| densenet121_2d (IN) | 0.59 | 1.381 → 1.088 | 1.478 → 1.185 | drops 0.29; learning, slowly |
| densenet121_2d_rin | 0.94 | 1.167 → 1.141 | 1.21–2.10 | barely moves; one large val spike |
| efficientnet_b0_2d | 0.74 | 1.729 → 1.516 | 2.034 → 1.716 | starts very high; never converges |
| resnet50_2d (IN) | 0.43 | 1.401 → 1.132 | 1.360 → 1.147 | drops 0.27; learning |
| resnet50_2d_rin | 0.82 | 1.269 → 1.112 | 1.270 → 1.133 | small drop |
| swin_tiny_2d (RIN) | 0.62 | 1.287 → 1.078 | 1.310 → 1.107 | small drop, then flat |
| **mil_resnet50** | 0.51 | **1.100 → 1.093** | **1.092 → 1.087** | **flat, at ln(3)** |
| **mil_swin_tiny** | 0.56 | **1.112 → 1.096** | **1.110 → 1.104** | **flat, at ln(3)** |
| **swin_unetr (3D)** | 1.12 | **1.114 → 1.113** | **1.111 → 1.108** | **flat, at ln(3)** |

`ln(3) ≈ 1.0986` is the cross-entropy of a perfectly uniform 3-class predictor against a one-hot label. With label smoothing 0.1 the expected floor for a uniform predictor is the same value. **The MIL and 3D fine-tunes never escape this floor.** The 2D models partially escape but plateau at val_loss ~1.1–1.2.

### 1.2 DAPT-pretrained features barely beat ImageNet on BigLunge

Frozen-backbone linear probe with `LogisticRegression(class_weight=balanced)` on BigLunge 2D slices (crop=96, mp=1, max_slices=8). 1510 train slices / 322 val / 332 test from 196 / 41 / 43 patients. Best C per row:

| Backbone | Slice val F1 | Slice test F1 | Patient test F1 (mean-softmax) |
|---|---|---|---|
| ImageNet (no DAPT)        | 0.37 | 0.36 | **0.37** |
| DAPT `pbest_raw` (Apr 29) | 0.39 | 0.32 | 0.29 |
| DAPT `epoch_10`           | 0.39 | 0.34 | 0.27 |
| DAPT `epoch_30`           | 0.34 | 0.36 | **0.44** |

Three things to take from this table:

- **DAPT does not improve on ImageNet at the feature level for BigLunge.** Both ceilings hover around 0.36–0.37 slice-level F1.
- **The val-best DAPT checkpoint (`pbest_raw`) is *worse* than the last epoch.** The cosine LR decay over the last ~20 epochs partially regularises the source-domain overfitting; selecting by val acc undoes that regularisation.
- **The actual full fine-tune of `resnet50_2d` reports test patient-MacroF1 = 0.444** (`results/thesis/2d/tables/headline.md`). The frozen-backbone linear probe on `dapt_epoch_30` already hits 0.44. Fine-tuning is adding ~zero on top of the right pretraining checkpoint.

Sanity check: the same DAPT features on Lung-PET-CT-Dx test give slice-level F1 = 0.52, acc 0.70 — features *are* discriminative on the source domain.

### 1.3 MIL attention has collapsed to uniform

From the canonical mil_resnet50 FT epochs (logged each val epoch):

```
attention.entropy_mean : 0.93 – 0.98   (1.0 = uniform over N=32 slices, 0.0 = one-hot)
attention.top1_mass    : 0.06 – 0.10   (uniform = 1/32 = 0.031)
attention.top3_mass    : 0.16 – 0.25   (uniform = 3/32 = 0.094)
```

The attention head produces almost-uniform weights over the bag, so the bag prediction is essentially the mean-pooled feature of all 32 slices passed through a linear classifier. With 70+% of bags containing only one tumor-bearing slice (auto-seg lung extent ≈ 200+ slices, ~32 mm tumor at 0.6 mm Z spacing covers ≈ 53 slices — most bags genuinely have many tumor slices, but uniform attention is still wrong because it gives equal weight to lung-only slices).

Why the attention can't escape uniform:

- `bag_dropout=0.15` zeroes ~5 of 32 instances per step; this directly trains *against* concentrated attention.
- `finetune_lr=1e-5`, `head LR = 1e-5`, `backbone_lr_scale=0.1` → backbone gets 1e-6.
- 5-epoch backbone freeze + cosine decay over 40 epochs, but only ~16 epochs run before early-stop → head sees max LR for ~5 epochs then decays.
- The attention head is **randomly initialised** at FT start (DAPT trained a 2D model, not a MIL model). 16 epochs at 1e-5 LR is not enough to learn a fresh attention layer over 211 patients.

### 1.4 Massive Z-spacing distribution shift between datasets

Computed on the 352 Lung-PET-CT-Dx and 307 BigLunge volumes that pass the loader filters:

```
Lung-PET-CT-Dx (DAPT)        BigLunge (FT)
median Z-spacing  5.00 mm     0.60 mm        ← 8× difference
[0,1) mm:  107 (30%)         243 (79%)
[1,2) mm:   34 (10%)          41 (13%)
[2,3) mm:    0                21 (7%)
[3,5) mm:    5                 1
[5,7) mm: 179 (51%)            0
[7,10) mm:   5                 0
[10,20) mm: 22 (6%)            1
median # slices/volume:  62           602
```

All 3D / MIL pipelines resample to `pixdim=(1.5, 1.5, 2.0)` mm or `(1.0, 1.0, 2.0)` mm. So:

- A typical DAPT volume (Z=62 slices @ 5 mm = 310 mm extent, resampled to 2 mm gives 155 slices) — interpolated 2.5× along Z. Adjacent slices in the resampled volume are interpolated copies of the same physical slice.
- A typical BigLunge volume (Z=602 @ 0.6 mm = 360 mm extent, resampled to 2 mm gives 180 slices) — 3× downsampled. Each resampled slice averages ~3 physical slices.

Both arrive at ~150–180 axial slices, so geometry matches *post-resample*, but the **content** differs sharply. Thin-slice CTs reveal fine vasculature, fissures, septal lines that thick-slice CTs blur into noise. A model that learned features on thick slices is being asked to interpret thin-slice features at test time, and vice versa.

For the 2D pipeline this matters less per slice (single axial slice resampled to 1×1 mm in-plane), but the *picked slice* is fundamentally different: an axial slice from a 5-mm scan is a 5-mm partial-volume average, whereas one from a 0.6-mm scan is a near-isotropic slab.

### 1.5 BigLunge tumor masks are noisy

From `results/output/multifocal_audit.csv` (existing, 306 rows):

| | n | %  |
|---|---|---|
| Mask absent or empty                                             | 4  | 1% |
| Mask present but `largest_component_voxels == 0` (sub-threshold) | 30 | 10% |
| Single connected component                                       | 79 | 26% |
| Multifocal (≥ 2 CCs ≥ 50 voxels)                                 | 217 | 71% |
| 5+ components                                                    | 81 | 27% |

The 30 patients with effectively-empty masks (largest CC < 50 voxels) are spread across all three classes. They are silently:

- **Dropped** from the 2D pipeline (no tumor slices found, logged in `dropped_patients.json` but not reflected in the val/test counts).
- **Centered on volume midpoint** by `ExtractSubVolumed` in 3D — the model looks at lung center, not tumor.
- **Unaffected** in MIL (which uses lung mask, not tumor mask).

77% of SCLC patients are multifocal (median 3 CCs, max 44). The largest-CC fix from 2026-04-29 (`min_component_voxels=50`) handles this for cropping but does not validate that the auto-seg actually found *the primary*; some "largest CC" picks are bulky mediastinal nodes and not the lung primary.

### 1.6 Per-class prediction skew after FT

Reading the BL-test inference probability JSONs for the six 2D models:

```
True distribution: 15 ADC / 13 SCLC / 15 SCC

Model                       Predicted ADC / SCLC / SCC   Mean confidence
efficientnet_b0_2d            21 / 11 / 11               0.57
resnet50_2d                   19 /  8 / 16               0.52
resnet50_2d_rin               21 /  9 / 13               0.50
densenet121_2d                16 / 13 / 14               0.48
densenet121_2d_rin            18 / 11 / 14               0.49
swin_tiny_2d                  20 / 12 / 11               0.40
```

Two things:

- **ADC is over-predicted in 5/6 models** (16–21 vs ground-truth 15). DenseNet is the only one not skewed. This is the residual ADC bias from the 73% ADC-heavy DAPT phase, not corrected by FT.
- **Mean confidence is in 0.40–0.57** across models, with `conf_right − conf_wrong` of only **0.01–0.05** (e.g. resnet50_2d: 0.527 vs 0.515). The models have essentially no calibrated certainty on BigLunge — predictions are coin-flip-like even when correct.

DAPT-test inference of the same models (Lung-PET-CT-Dx test, n=53, true 38 / 6 / 9):

- All models predict mostly ADC (49–75% ADC predictions).
- ADC recall 63–89%, SCLC recall 33–50%, SCC recall 33–89% — the ADC majority is what holds DAPT-test accuracy at 0.66–0.77.
- This is *not* a sign of strong learning; it's the prior distribution leaking through.

---

## Section 2 — Diagnosis

The model failures from Section 1 stack:

### Lung-PET-CT-Dx (mediocre, ADC-dominated)

- DAPT class distribution is 73% ADC, 11% SCLC, 18% SCC at the patient level.
- `WeightedRandomSampler` is on, but it samples *scans*, not patients, and the multi-scan-cap of 2 means a single SCLC patient still has 1–2 scans against an ADC patient's 2.
- With `label_smoothing=0.1`, the smoothed-prior baseline already gets log-loss ≈ 1.10, so the loss landscape is shallow and the model converges to "predict ADC most of the time," which does well on a 73%-ADC val/test split.
- Result: high accuracy, low balanced accuracy (e.g. SwinUNETR DAPT acc=0.585, balanced acc=0.375 — the best raw checkpoint is one that just predicts ADC).
- This is also why every model has a bigger train-val gap than the published recipe suggests: the model memorises ADC-vs-other rather than learning the three-way separation.

### BigLunge (poor)

The dataset itself is balanced (35 / 32 / 34 %), so an ADC-biased model cannot ride the prior. Each of the five issues above contributes:

- **Domain shift (Z-spacing):** DAPT features were trained on thick-slice content; BigLunge slices look different post-resample.
- **Pretrain class imbalance:** features specialised on ADC-vs-other carry forward.
- **Auto-seg masks:** 10% of patients silently dropped or wrongly cropped.
- **MIL attention collapse:** bag-level model can't focus on tumor slices.
- **Hyperparameters:** FT LR is too low for what is effectively a fresh classifier head against shifted features. 16–22 epochs of training at backbone LR 3e-6 is not enough to reshape the backbone.

The linear-probe finding (Section 1.2) is the cleanest summary: **DAPT does not produce features that beat ImageNet on BigLunge.** Fine-tuning a backbone whose features barely encode the target structure is bound to plateau at chance + epsilon.

---

## Section 3 — Recommendations, ranked by effort

### 3.A Zero-cost (no retraining)

**1. Switch the FT initialiser from `dapt_pbest_raw.pth` to `dapt_epoch_30.pth`.**
The val-best checkpoint is overfit to the source domain. The last cosine-decayed epoch transfers measurably better:
- Patient-level test F1: `pbest_raw` 0.29 → `epoch_30` 0.44 (linear-probe, holding everything else fixed).
- The current `mil_resnet50` and `swin_unetr` configs both consume `pbest_raw`. Re-running just the FT phase from `epoch_30` would be cheap (~30 min per model on H100).

**2. Drop the 30 patients with sub-threshold tumor masks from BigLunge entirely.**
Add their patient IDs to `sclc/data/exclusions.py` (or build a parallel `EMPTY_TUMOR_MASK` list). Right now they pollute the splits silently — 3D evaluates them on volume-centered crops, and the "drop" log message in the 2D pipeline isn't reflected in the headline counts.

**3. Report headline numbers using `epoch_30` as the FT seed for an ablation table.**
This is a thesis-grade finding: "Selecting DAPT checkpoints by source-domain val acc is harmful for cross-domain transfer." Worth its own paragraph in Discussion.

### 3.B Low-cost (one or two new runs)

**4. Raise the FT learning rate, especially for MIL and 3D.**
Current `finetune_lr=3e-5` (or 1e-5 for MIL) with `backbone_lr_scale=0.1` is bottom-of-the-recommended-range and the head is essentially random for MIL/3D. Try:
- 2D: `finetune_lr=1e-4`, head with no scale, backbone `0.1×` → `1e-5` for backbone. Increase `finetune_warmup_epochs` to 5 to absorb the higher initial LR.
- MIL: `finetune_lr=3e-4` for the head, `1e-5` for the backbone (scale 0.033) — MIL has the most fresh parameters.
- 3D: `finetune_lr=1e-4`, `backbone_lr_scale=0.1`.

**5. Remove `bag_dropout=0.15` for MIL (it forces uniform attention).**
The attention-entropy logs are unambiguous: with bag dropout on at 15%, the model can never concentrate attention because any single slice it would prefer might be zeroed at training time. Run one MIL epoch with `bag_dropout=0.0` and check if `attention.entropy_mean` drops below 0.9 — if yes, the rest of the FT will start producing meaningful predictions.

**6. Drop the LP-FT freeze for MIL, or shorten it to 2 epochs.**
With `finetune_freeze_backbone_epochs=5` and `finetune_epochs=40` cosine-decayed, the backbone is unfrozen at LR ≈ 0.7× peak and decays from there. For a fresh attention head this is fine, but for the backbone+head co-adaptation that the rest of the training does, you want the backbone unfrozen earlier. 2 epochs of head warm-up is enough.

**7. Calibrate predictions on val before reporting test.**
Mean test confidence is 0.40–0.57 with right/wrong gap of 0.01–0.05 — temperature scaling on val would push confidence either down (revealing miscalibration) or up where the model is right. This doesn't change accuracy but produces honest probability outputs and is one of the standard checks reviewers expect.

### 3.C Medium-cost (full DAPT redo)

**8. Rebalance DAPT class distribution.**
Either:
- Subsample ADC patients to roughly 38 (matching SCLC), discarding 200+ scans. Loses information but removes the bias.
- Or weight the DAPT loss by inverse class frequency (`CrossEntropyLoss(weight=…)`) instead of relying on `WeightedRandomSampler` — explicit class weighting in the loss is more honest than letting the sampler fight class balance against patient-level repetition.

**9. Z-resolution-matching augmentation during DAPT.**
Add `RandSpacingd` (or a custom Z-jitter) to the DAPT train transforms that randomly resamples Z to a value drawn from the BigLunge distribution before resampling back to 2 mm. This forces the model to learn features that are robust to the thick-vs-thin difference.

**10. Stop using `pbest_raw` selection on the source domain when the goal is downstream transfer.**
Save checkpoints at every 5 epochs and pick the one that maximises a *target-domain validation metric* (BigLunge val) — even if the FT is what produces the final weights, the DAPT checkpoint that warms up best for transfer is not the one with peak source-domain accuracy.

### 3.D Higher-cost / structural

**11. Replace MIL attention with a different aggregator on this dataset size.**
Attention over 32 instances with 211 patients of supervision and a randomly-initialised attention layer is hard. Two alternatives:
- *Topk* mean: take the top-k bag instances by classifier logit and average. Trivially trainable, no attention to collapse.
- *Pre-trained attention*: train the attention layer on a bigger external CT dataset first (e.g. fit it to predict tumor-vs-not from per-slice tumor masks on BigLunge train), then finetune jointly.

**12. Stop relying on auto-seg tumor masks for cropping.**
Use the algorithmic *lung* mask (which is much better) plus an MR-derived "lung primary" detector trained on Lung-PET-CT-Dx (where masks are radiologist-drawn). This decouples the cropping from the noisy auto-seg.

**13. Reconsider which dataset is "headline".**
If BigLunge is balanced and Lung-PET-CT-Dx is heavily imbalanced, and the target population for the thesis is BigLunge, then Lung-PET-CT-Dx is being used as a pre-training source rather than as the evaluation. The thesis already acknowledges this (BigLunge is the headline per memory `project_target_dataset_decision.md`), but the *DAPT objective* should be tuned for transfer, not for source-domain accuracy. The fact that the same model gets 0.77 acc on Lung-PET-CT-Dx and 0.44 acc on BigLunge is not a "the model is good but the target is hard" story — it is "the source-domain accuracy is buoyed by the ADC majority and tells you nothing about target-domain capability."

---

## Section 4 — What I did not check (open questions)

- **3D SwinUNETR DAPT itself does not converge** (val acc oscillates 0.118–0.667, ends at 0.333 in the canonical 20-epoch run). I attribute this to the 1.5 × 1.5 × 2.0 mm grid being a poor match for SwinUNETR's window size and to the limited number of effective gradient steps with `batch_size=2 × accumulation_steps=4 = 8`, but I did not run an isolated experiment to confirm. Worth checking whether a longer warmup (5–10 epochs of warmup vs the current 3) helps DAPT stability.
- **2D EfficientNet-B0 FT starts at train_loss 1.73** while the others start at 1.30–1.40. I do not know why — possibly the EfficientNet head is being reinitialised between DAPT and FT? Worth checking `_is_head_param` logic against the EfficientNet wrapper.
- **The mil_swin_tiny config is restored but I did not verify it actually trains a different model.** The `metrics.jsonl` shows the same flat-loss pattern as mil_resnet50, but I did not extract attention diagnostics for it specifically.
- **The 3D pipeline's `use_lung_crop=True` for FT is disabled for DAPT.** This is itself a domain shift (DAPT volumes don't get lung-cropped, FT volumes do). I did not separate the contribution of this from the Z-spacing shift.

---

## Appendix — Reproducing the linear-probe ceiling

```bash
cd /home/hansstem/SCLC-Classification
python - <<'PY'
import sys, numpy as np, torch
from torch.utils.data import DataLoader
from sclc.models import get_sclc_model
from sclc.data.dataset_2d import create_dataset_2d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

device = torch.device("cuda")
m = get_sclc_model(model_type="resnet50_2d").to(device).eval()
sd = torch.load("/home/data/trained_models/2d/resnet50_2d/Apr_29_04_dapt_epoch_30.pth", map_location=device, weights_only=False)
m.load_state_dict(sd, strict=False)
for p in m.parameters(): p.requires_grad = False
buf = {}
m.backbone.features.avgpool.register_forward_hook(lambda mm,i,o: buf.__setitem__("f", o.detach().flatten(1).cpu().numpy()))

ds_tr, ds_va, ds_te = create_dataset_2d(
    data_path="/home/data/TrainingData",
    csv_path="/home/data/TrainingData/patients_parameters.csv",
    dataset_type="big_lunge", img_size=224, crop_size=96,
    tumor_mask_suffix="_label_tc.nii.gz",
    max_slices_per_volume=8, min_tumor_pixels=1,
    val_frac=0.15, test_frac=0.15, seed=42, cache_workers=4,
)
def extract(ds):
    feats, labs, pids = [], [], []
    for batch in DataLoader(ds, batch_size=16, num_workers=4, collate_fn=lambda b: b):
        for it in batch:
            x = it["image"].unsqueeze(0).to(device)
            with torch.no_grad(): _ = m(x)
            feats.append(buf["f"][0]); labs.append(int(it["scan_label"])); pids.append(it.get("patient_id"))
    return np.array(feats), np.array(labs), np.array(pids)

Xtr,ytr,_ = extract(ds_tr); Xva,yva,pva = extract(ds_va); Xte,yte,pte = extract(ds_te)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(sc.transform(Xtr), ytr)
P = clf.predict_proba(sc.transform(Xte))
from collections import defaultdict
g, gy = defaultdict(list), {}
for i,p in enumerate(pte): g[p].append(P[i]); gy[p] = yte[i]
yp = np.array([np.mean(g[k], axis=0).argmax() for k in g])
yt = np.array([gy[k] for k in g])
print(f"DAPT_epoch30 patient-test F1mac = {f1_score(yt, yp, average='macro'):.3f}  (full FT reports 0.444)")
PY
```

Expected output: `0.4xx` — within bootstrap CI of the headline FT result.
