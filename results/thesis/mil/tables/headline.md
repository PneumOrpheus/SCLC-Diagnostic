# MIL pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| MIL ResNet-50 | Lung-PET-CT-Dx (test) | 53 | 0.566 [0.452, 0.679] | 0.508 [0.371, 0.653] | 0.450 [0.305, 0.607] | 0.704 [0.570, 0.826] |
| MIL ResNet-50 | BigLunge (test) | 46 | 0.261 [0.152, 0.391] | 0.265 [0.152, 0.390] | 0.262 [0.144, 0.382] | 0.426 [0.310, 0.553] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| MIL ResNet-50 | Lung-PET-CT-Dx (test) | 0.710 [0.582, 0.818] | 0.250 [0.000, 0.667] | 0.389 [0.250, 0.514] |
| MIL ResNet-50 | BigLunge (test) | 0.308 [0.083, 0.519] | 0.256 [0.102, 0.410] | 0.222 [0.000, 0.444] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| MIL ResNet-50 | Lung-PET-CT-Dx (test) | 0.781 [0.637, 0.896] | 0.670 [0.411, 0.911] | 0.662 [0.465, 0.833] |
| MIL ResNet-50 | BigLunge (test) | 0.483 [0.308, 0.669] | 0.440 [0.250, 0.634] | 0.356 [0.198, 0.531] |
