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
