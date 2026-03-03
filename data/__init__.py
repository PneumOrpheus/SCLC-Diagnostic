"""
SCLC Data Module
----------------
Provides data loading and preprocessing utilities for CT scan data.
"""

from .data_preprocessing import (
    load_nifti_volume,
    load_numpy_volume,
    load_volume,
    clip_hounsfield_units,
    resample_volume_isotropic,
    apply_windowing,
    create_multichannel_ct,
    normalize_intensity,
    extract_2d_slice,
    prepare_tensor_for_model,
    preprocess_nifti_to_numpy,
    batch_preprocess_directory,
)

from .biglunge_loader import (
    create_biglunge_dataset,
    get_biglunge_data_list,
    load_patient_labels,
    get_class_names,
    get_num_classes,
    CLASS_NAMES,
    NORWEGIAN_CLASS_MAP,
)

__all__ = [
    # Data preprocessing
    "load_nifti_volume",
    "load_numpy_volume",
    "load_volume",
    "clip_hounsfield_units",
    "resample_volume_isotropic",
    "apply_windowing",
    "create_multichannel_ct",
    "normalize_intensity",
    "extract_2d_slice",
    "prepare_tensor_for_model",
    "preprocess_nifti_to_numpy",
    "batch_preprocess_directory",
    # BigLunge dataset
    "create_biglunge_dataset",
    "get_biglunge_data_list",
    "load_patient_labels",
    "get_class_names",
    "get_num_classes",
    "CLASS_NAMES",
    "NORWEGIAN_CLASS_MAP",
]
