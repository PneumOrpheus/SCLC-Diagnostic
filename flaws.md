# Pipeline Flaws — SCLC 3D Classification

_Audit target: `main.py`, `model_selection.py`, `data/data_loader.py`, `data/transforms.py`, `training/train.py`._
_Latest evidence: `output/swin_unetr_2026-04-15_06_logs.txt` (in progress at time of analysis — DAPT 28/40, fine-tune 7/40)._
_Earlier baselines: `output/swin_unetr_2026-04-14_12_logs.txt`, `output/resnet_2026-04-14_08_logs.txt`._

## Progression

### Round 1 — `2026-04-14_08` (ResNet18) and `2026-04-14_12` (SwinUNETR)

Both backbones reproduced the same two failure modes:

1. **DAPT classification head never left uniform.** Train cls loss pinned at `1.08–1.10` for all 40 DAPT epochs. On SwinUNETR the seg loss dropped `0.91 → 0.30` (real decoder, mask signal flowed) but classification gained nothing. ResNet18's seg "loss" was a no-op entirely — its `forward(return_segmentation=True)` returns `torch.zeros(...)` with no graph connection.
2. **Fine-tune collapsed at the unfreeze epoch.** Frozen-head warmup (epochs 1–5) climbed to macro-F1 ≈ 0.34. Epoch 6, backbone unfroze at peak warmup LR (`3e-5`) → macro-F1 dropped to 0.18 in one epoch and never recovered. SwinUNETR's last 8 epochs flatlined at exactly `MacroF1=0.3411` to four decimals — optimizer dead, constant prediction.

Final test macro-F1: ResNet18 `0.1429`, SwinUNETR `0.2600`. Both random.

### Round 2 — `2026-04-15_06` (SwinUNETR, with fixes #4 + #5 + E.2 landed)

Both critical pipeline bugs from Round 1 are now **confirmed fixed** in live evidence:

**DAPT — seg_loss_weight=0.1 fix worked, dramatically:**
| Metric | Round 1 (seg=0.5) | Round 2 (seg=0.1) |
|---|---|---|
| Train cls loss (final) | pinned at ~1.10 | **0.75 at ep28, still dropping** |
| Train macro-F1 | not logged | **0.3487 → 0.6770** ep1→28, climbing with noise (dips at ep3/11/13/15/21/24) |
| DAPT val rolling-3 best | 0.3868 | 0.2813 (ep8) |

The classification head is genuinely learning for the first time. Previously invisible because (a) we had no train-F1 metric and (b) seg was eating the gradient. Both diagnostics landed in Round 2 and immediately paid off.

**Fine-tune — differential LR fix worked, no epoch-6 cliff:**
```
ep5  TrainMacroF1 0.36   ValMacroF1 0.34
ep6  TrainMacroF1 0.29   ValMacroF1 0.30   ← Round 1 collapsed to 0.18 here
ep7  TrainMacroF1 0.30   ValMacroF1 0.32
```
Backbone at `3.00e-06` vs head at `3.00e-05` = 10× differential. Pretrained features evolve gently instead of getting nuked on the first unfrozen step. **Cliff is gone** across both models on the same fix.

**The new dominant failure mode: DAPT overfitting.** Now that cls is learning, a different problem is visible:

```
          Train MacroF1   Val MacroF1 (cur / roll3)
ep 8      0.3790          0.2884 / 0.2813   ← best rolling val
ep 14     0.4778          0.1920 / 0.2281
ep 20     0.5815          0.0958 / 0.1994
ep 26     0.6338          0.1740 / 0.2153
ep 28     0.6770          0.2758 / 0.2147
```

Train and val diverge after epoch 8. Rolling-3 val F1 has been flat-to-declining for 20 epochs while train keeps climbing. This is **textbook overfitting on a scan-diversity-limited class** — 26 SCLC training patients with ~0.7 scans each is not enough signal-diversity for the model to generalize. It's memorizing specific anatomies.

**Action taken in response:** `patience` reduced from 20 → 10, `--dapt-epochs` default reduced from 40 → 15 in `main.py`. The best DAPT checkpoint historically lands around epoch 8; training past epoch 15 is negative-value and just burns compute / risks worse overfitting.

## Is the data the real problem?

**Round 2 changed the answer to "partially yes, but *differently* than we thought."**

- Round 1 hypothesis: "model can't fit the training data, probably data is too small."
- Round 2 reality: model fits training data fine (train F1 reaches 0.6770 by ep28 and still climbing). **It overfits.** Val generalization stalls after ~8 epochs.
- Patient count (~26 SCLC train / 6 val / 6 test) is enough to *fit*, but not enough to *generalize* given the sparse scan-per-patient distribution.
- The "data too small" framing was wrong in its original form (cls can learn) but right in a new form (can't generalize without more diverse examples).

The remaining levers that could improve generalization *without* collecting more data:
1. **Stronger augmentation** — current intensity aug is at `prob=0.3`. More aggressive random affine, intensity noise, crop/cutout would synthesize diversity.
2. **Higher weight decay** — DAPT currently uses `1e-3`. Overfitting regime suggests `1e-2` or `3e-2`.
3. **`max_scans_per_patient` cap (#8)** — flattens image distribution, lets the model see more diverse *patients* per epoch instead of repeat-sampling the same adeno scans.
4. **Binary SCLC-vs-NSCLC relabeling** — collapses the 3-way task into a 2-way task where the label boundary is cleaner and the per-class sample count doubles.

---

## Fixed (no longer active)

- **A/B — Monitor metric was accuracy; P/R/F1 cosmetic duplicates of accuracy.** Replaced with manual confusion-matrix macro-F1 computation in `validate_epoch`. Rolling-3 smoothing for checkpoint selection.
- **C — Fine-tune unfreeze cliff.** Fixed by differential LR (backbone_lr = head_lr × 0.1) via AdamW param groups. Confirmed in Round 2 evidence: no collapse at epoch 6 for SwinUNETR; schedule gymnastics removed.
- **G — DAPT seg loss dominated cls loss.** Fixed by `--seg-loss-weight` flag (default `0.1`, was effectively `0.5`). Confirmed: cls loss drops from 1.1802 → 0.7502 (ep1 → ep28) and train macro-F1 climbs 0.3487 → 0.6770 instead of staying flat at uniform.
- **E.1 — Single-epoch val metric for checkpoint selection.** Replaced with rolling-window mean (default k=3) via `deque`.
- **E.2 — No train macro-F1.** `train_epoch` now returns `(loss, train_macro_f1)` via the shared `_compute_classification_metrics` helper. Logged per epoch next to val metrics.
- **E.3 — Val metrics not reported as rolling mean.** `run_training_phase` now logs current-vs-rolling for every metric key each epoch.

---

## Active issues

### Overfit-1. (HIGH, NEW) DAPT overfits after ~8 epochs

**Evidence (Round 2 above):** train macro-F1 rises monotonically 0.35 → 0.68, val macro-F1 peaks at 0.28 (ep8) and plateaus/declines through ep28. Train loss halves while val loss drifts up.

**Partial mitigation already in place:** `--dapt-epochs` default reduced to 15, `patience` reduced to 10. Early stop will now trigger around epoch 18–20 instead of 40.

**Full fix (next round):**
1. **Raise DAPT weight decay** from `1e-3` → `1e-2`. Single CLI default change. Highest-impact lever for the overfitting regime.
2. **Strengthen train augmentation** in `data/transforms.py`:
   - `RandAffined` probabilities up from default
   - Add `RandCoarseDropoutd` (MONAI cutout equivalent) with small hole count
   - Raise `RandScaleIntensityd` / `RandShiftIntensityd` back toward `prob=0.5`
   - Consider `RandGaussianNoised` at `prob=0.3`
   - Keep val transforms untouched
3. **Cap `max_scans_per_patient` (issue #8)**. Currently still open. For Lung-PET-CT-Dx, setting this to `2` would shrink the adeno dominance from ~8:1 to ~2:1 at the image level and force the sampler to cover more distinct patients per epoch.
4. **Consider label smoothing increase** from `0.1` → `0.2` in the DAPT cls criterion. Light but zero-cost.

Note: none of these help if the real problem is that 26 SCLC training patients is simply not enough to span the SCLC appearance manifold. Augmentation and regularization can only stretch existing signal, not synthesize missing signal. If Round 3 still overfits hard with all four mitigations, the data ceiling is real.

---

### D. (HIGH) ViT trains from random init on ~300 volumes

`get_sclc_model("", model_type="vit")` constructs MONAI's `ViT` and only loads weights if `--initial-checkpoint` is passed and exists. A 27M-param transformer cannot learn a 3-way label from ~300 CT volumes from scratch.

**Fix (pick one):**
1. **Remove `vit` from `get_sclc_model`'s dispatch entirely.** Cheapest, honest.
2. **Gate it:** raise `ValueError("model_type=vit requires --initial-checkpoint pointing to a 3D-ViT pretrained file")` if the checkpoint is missing or empty.

Recommend option 1. User has already confirmed they don't care about ViT.

---

### F / #8. (MEDIUM) Patient stratification leaves image counts wildly skewed

Adeno patients average ~8 scans/patient, SCLC ~0.7. `train_test_split(stratify=patient_labels)` balances *patients*, not *images*. Image-level ratio is ~10:1:2 before the sampler. `WeightedRandomSampler` then upsamples the same handful of SCLC patients dozens of times per epoch — the model sees the same SCLC volumes repeatedly rather than diverse SCLC examples. This is the likely mechanism behind the Round 2 overfitting.

**Fix:** add `max_scans_per_patient` to `data/data_loader.py::get_lung_pet_ct_dx_data_list`, default `2`. After listing scans for a patient, take the first `min(N, max_scans_per_patient)` (or a deterministic random subset with a fixed seed). This:

- flattens the image-level distribution toward 1 : 1 : 1,
- shrinks the dominant adeno class so the WeightedRandomSampler doesn't have to work as hard,
- makes val and test confusion matrices honest,
- forces per-epoch coverage of more distinct patients, which is the actual mechanism that reduces overfitting.

Apply consistently to train, val, and test so support counts in the val log are interpretable.

Has been open since Round 1. **This is now the single most likely fix for Round 2's overfitting**, because it attacks the scan-diversity problem directly.

---

### H / #7. (MEDIUM) Non-SwinUNETR models silently ignore --anno

`ResNetClassifier`, `ResNet18Classifier`, `ViTClassifier`, `DenseNetClassifier`, `ModelsGenesisClassifier` all return `torch.zeros((B, 1, *spatial), device=x.device)` for the seg branch. No graph connection, zero gradient — but the loss *value* still appears in the log, so `--anno` looks like it's doing something when it isn't.

**Fix:** in `main.py`, after model construction, if `args.anno and args.model_type != "swin_unetr"`:

```python
logger.warning(
    f"--anno is set but model_type={args.model_type} has no real segmentation "
    f"decoder; mask supervision will have NO effect on training. Either switch "
    f"to swin_unetr or drop --anno."
)
```

Not blocking anything, but stop letting the log lie about what's being trained.

---

## Sanity-check sequence for the next run (Round 3)

1. Land **Overfit-1** mitigations: weight decay `1e-2` for DAPT, stronger augmentation in `data/transforms.py`, `max_scans_per_patient=2` (issue F / #8). Optionally label smoothing → 0.2.
2. Run SwinUNETR with the new `--dapt-epochs 15 --patience 10` defaults.
3. Success criteria:
   - DAPT train and val macro-F1 should *track* each other (gap < 0.15) rather than diverging.
   - DAPT val rolling-3 macro-F1 should beat Round 2's `0.2813` — anything above `0.35` is meaningful progress.
   - Fine-tune should show an actually climbing val macro-F1, not a flat 0.27–0.34 band.
4. If DAPT train macro-F1 *stops* climbing (too regularized now), back weight decay off to `3e-3`. The goal is train/val close together, not both low.
5. If train/val still diverge despite all four mitigations, the scan-diversity ceiling is confirmed and the realistic options are (a) collect more SCLC patients, (b) switch to binary SCLC-vs-NSCLC, or (c) ensemble / k-fold pooled predictions for evaluation honesty.

Maybe try out 5-fold-validation split?

Test out RetinaNet from monai??