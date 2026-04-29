# 3D pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| SwinUNETR (3D) | Lung-PET-CT-Dx (test) | 53 | 0.585 | 0.375 [0.248, 0.523] | 0.377 [0.239, 0.525] |  |
| SwinUNETR (3D) | BigLunge (test) | 46 | 0.326 [0.196, 0.457] | 0.330 [0.199, 0.467] | 0.326 [0.188, 0.457] | 0.496 [0.367, 0.617] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| SwinUNETR (3D) | Lung-PET-CT-Dx (test) | 0.727 [0.628, 0.815] | 0.182 [0.000, 0.462] | 0.222 [0.000, 0.500] |
| SwinUNETR (3D) | BigLunge (test) | 0.308 [0.080, 0.533] | 0.375 [0.187, 0.581] | 0.294 [0.114, 0.474] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| SwinUNETR (3D) | Lung-PET-CT-Dx (test) |  |  |  |
| SwinUNETR (3D) | BigLunge (test) | 0.454 [0.279, 0.637] | 0.547 [0.375, 0.723] | 0.488 [0.304, 0.652] |
