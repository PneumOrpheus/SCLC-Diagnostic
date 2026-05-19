# 2D pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| EffNet-B0 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.679 [0.566, 0.792] | 0.579 [0.414, 0.744] | 0.544 [0.407, 0.685] | 0.821 [0.705, 0.922] |
| EffNet-B0 (2D, ImageNet) | BigLunge (test) | 41 | 0.463 [0.317, 0.610] | 0.456 [0.313, 0.601] | 0.429 [0.287, 0.580] | 0.628 [0.501, 0.750] |
| ResNet-50 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.736 [0.623, 0.849] | 0.634 [0.478, 0.789] | 0.607 [0.420, 0.768] | 0.814 [0.708, 0.902] |
| ResNet-50 (2D, ImageNet) | BigLunge (test) | 41 | 0.415 [0.268, 0.561] | 0.414 [0.267, 0.560] | 0.422 [0.268, 0.560] | 0.639 [0.507, 0.774] |
| ResNet-50 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.660 [0.547, 0.774] | 0.542 [0.375, 0.708] | 0.508 [0.369, 0.649] | 0.798 [0.678, 0.897] |
| ResNet-50 (2D, RadImageNet) | BigLunge (test) | 41 | 0.341 [0.195, 0.488] | 0.337 [0.194, 0.480] | 0.321 [0.187, 0.456] | 0.531 [0.386, 0.670] |
| DenseNet121 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.679 [0.566, 0.792] | 0.522 [0.349, 0.681] | 0.507 [0.349, 0.655] | 0.769 [0.635, 0.884] |
| DenseNet121 (2D, ImageNet) | BigLunge (test) | 41 | 0.463 [0.317, 0.585] | 0.454 [0.315, 0.577] | 0.411 [0.291, 0.545] | 0.608 [0.473, 0.738] |
| DenseNet121 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.642 [0.528, 0.774] | 0.515 [0.380, 0.670] | 0.474 [0.369, 0.608] | 0.787 [0.667, 0.890] |
| DenseNet121 (2D, RadImageNet) | BigLunge (test) | 41 | 0.463 [0.365, 0.561] | 0.452 [0.357, 0.548] | 0.361 [0.262, 0.444] | 0.523 [0.405, 0.637] |
| Swin-Tiny (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.642 [0.509, 0.755] | 0.543 [0.396, 0.679] | 0.489 [0.365, 0.618] | 0.762 [0.658, 0.856] |
| Swin-Tiny (2D, RadImageNet) | BigLunge (test) | 41 | 0.463 [0.317, 0.610] | 0.463 [0.315, 0.610] | 0.454 [0.302, 0.596] | 0.603 [0.453, 0.737] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.812 [0.706, 0.899] | 0.250 [0.000, 0.500] | 0.571 [0.333, 0.800] |
| EffNet-B0 (2D, ImageNet) | BigLunge (test) | 0.581 [0.387, 0.757] | 0.235 [0.000, 0.526] | 0.471 [0.278, 0.640] |
| ResNet-50 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.857 [0.754, 0.932] | 0.444 [0.000, 0.800] | 0.519 [0.345, 0.692] |
| ResNet-50 (2D, ImageNet) | BigLunge (test) | 0.467 [0.240, 0.648] | 0.323 [0.133, 0.500] | 0.476 [0.211, 0.720] |
| ResNet-50 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.824 [0.721, 0.914] | 0.267 [0.000, 0.526] | 0.435 [0.200, 0.643] |
| ResNet-50 (2D, RadImageNet) | BigLunge (test) | 0.333 [0.091, 0.560] | 0.174 [0.000, 0.400] | 0.457 [0.267, 0.625] |
| DenseNet121 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.833 [0.735, 0.914] | 0.308 [0.000, 0.600] | 0.381 [0.118, 0.625] |
| DenseNet121 (2D, ImageNet) | BigLunge (test) | 0.552 [0.333, 0.727] | 0.111 [0.000, 0.316] | 0.571 [0.400, 0.722] |
| DenseNet121 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.818 [0.700, 0.914] | 0.143 [0.000, 0.400] | 0.462 [0.261, 0.643] |
| DenseNet121 (2D, RadImageNet) | BigLunge (test) | 0.467 [0.250, 0.645] | 0.000 [0.000, 0.000] | 0.615 [0.488, 0.737] |
| Swin-Tiny (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.776 [0.656, 0.865] | 0.154 [0.000, 0.429] | 0.538 [0.345, 0.696] |
| Swin-Tiny (2D, RadImageNet) | BigLunge (test) | 0.600 [0.400, 0.786] | 0.414 [0.181, 0.606] | 0.348 [0.095, 0.583] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.886 [0.772, 0.970] | 0.773 [0.560, 0.926] | 0.803 [0.616, 0.957] |
| EffNet-B0 (2D, ImageNet) | BigLunge (test) | 0.709 [0.532, 0.855] | 0.500 [0.305, 0.703] | 0.675 [0.471, 0.860] |
| ResNet-50 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.874 [0.760, 0.956] | 0.801 [0.656, 0.922] | 0.768 [0.619, 0.902] |
| ResNet-50 (2D, ImageNet) | BigLunge (test) | 0.704 [0.524, 0.865] | 0.503 [0.288, 0.728] | 0.712 [0.537, 0.857] |
| ResNet-50 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.914 [0.832, 0.974] | 0.699 [0.447, 0.894] | 0.783 [0.641, 0.904] |
| ResNet-50 (2D, RadImageNet) | BigLunge (test) | 0.587 [0.384, 0.757] | 0.423 [0.228, 0.629] | 0.582 [0.394, 0.757] |
| DenseNet121 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.889 [0.775, 0.972] | 0.645 [0.390, 0.862] | 0.773 [0.581, 0.927] |
| DenseNet121 (2D, ImageNet) | BigLunge (test) | 0.706 [0.532, 0.865] | 0.459 [0.266, 0.659] | 0.659 [0.447, 0.833] |
| DenseNet121 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.867 [0.749, 0.956] | 0.691 [0.479, 0.858] | 0.803 [0.639, 0.939] |
| DenseNet121 (2D, RadImageNet) | BigLunge (test) | 0.571 [0.381, 0.746] | 0.368 [0.209, 0.541] | 0.630 [0.442, 0.807] |
| Swin-Tiny (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.849 [0.732, 0.946] | 0.574 [0.369, 0.766] | 0.864 [0.763, 0.949] |
| Swin-Tiny (2D, RadImageNet) | BigLunge (test) | 0.706 [0.521, 0.876] | 0.525 [0.316, 0.709] | 0.577 [0.389, 0.762] |
