"""Hand-curated exclusion lists shared across pipelines.

Centralized so the same patient is excluded across the MIL bag pipeline,
the 3D SwinUNETR pipeline (which uses the same lung mask for the lung-bbox
crop), and any future pipeline that depends on lung-mask quality.
"""
from __future__ import annotations

# Patients whose ``_label_lungs.nii.gz`` is truncated (only part of the
# thorax is labeled). Effects:
#   * MIL: bag z-extent samples outside the lungs.
#   * 3D SwinUNETR: lung-bbox crop bounds the volume to a wrong region.
# Drop these from any pipeline that consumes the lung mask.
TRUNCATED_LUNG_MASK: frozenset[str] = frozenset({
    "patient_057069",
    "patient_091821",
    "patient_022269",
})

# Patients whose ``_label_tc.nii.gz`` (algorithmic tumor segmentation) has
# zero or sub-threshold positive voxels: the largest connected component is
# below 50 voxels. Effects:
#   * 2D pipeline silently drops them (no slice_idx survives min_tumor_pixels).
#   * 3D pipeline falls back to volume-center crop (no anatomic centering).
#   * MIL pipeline is technically unaffected (uses lung mask, not tumor
#     mask) but we still drop them for consistency — if the auto-seg
#     couldn't find the tumor, the label is suspect anyway.
# Source: ``results/output/multifocal_audit.csv``, rows where
# ``largest_component_voxels == 0`` (30 patients, 4 of them with
# ``mask_present=False`` and 26 where the file exists but is
# all-zeros / sub-threshold).
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
