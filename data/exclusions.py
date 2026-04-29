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
