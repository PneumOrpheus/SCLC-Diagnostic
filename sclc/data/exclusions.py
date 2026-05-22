"""Hand-curated exclusion lists shared across pipelines."""
from __future__ import annotations

# Patients with truncated ``_label_lungs.nii.gz`` — MIL bags sample outside the
# lungs and the 3D lung-bbox crop bounds the volume to a wrong region.
TRUNCATED_LUNG_MASK: frozenset[str] = frozenset({
    "patient_057069",
    "patient_091821",
    "patient_022269",
})

# Patients whose ``_label_tc.nii.gz`` largest connected component is < 50 voxels.
# Dropped uniformly across pipelines: the 2D path would drop them anyway
# (no slice survives min_tumor_pixels), and a missing tumor mask makes the
# label suspect for MIL/3D too.
EMPTY_TUMOR_MASK: frozenset[str] = frozenset({
    "patient_002106", "patient_002625", "patient_004891", "patient_005781",
    "patient_009376", "patient_015004", "patient_019222", "patient_022898",
    "patient_028173", "patient_029412", "patient_031539", "patient_035263",
    "patient_036580", "patient_036620", "patient_043831", "patient_044681",
    "patient_056956", "patient_063641", "patient_064077", "patient_069341",
    "patient_071451", "patient_084637", "patient_087599", "patient_089117",
    "patient_090366", "patient_091659", "patient_093884", "patient_095293",
    "patient_095499", "patient_098715",
})
