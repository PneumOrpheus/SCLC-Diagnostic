"""Model factory + per-family classifier wrappers."""

from .classifiers_2d import (
    DenseNet2DClassifier,
    EfficientNet2DClassifier,
    SwinV2Tiny2DClassifier,
    TorchVisionResNet2DClassifier,
)
from .classifiers_mil import MILSwinV2TinyClassifier
from .factory import (
    MIL_MODEL_TYPES,
    TWO_D_MODEL_TYPES,
    get_pipeline,
    get_sclc_model,
    is_2d_model_type,
    is_mil_model_type,
)
from .swin_unetr import SwinUNETRClassifier

__all__ = [
    "get_sclc_model",
    "get_pipeline",
    "is_2d_model_type",
    "is_mil_model_type",
    "TWO_D_MODEL_TYPES",
    "MIL_MODEL_TYPES",
    "SwinUNETRClassifier",
    "EfficientNet2DClassifier",
    "DenseNet2DClassifier",
    "TorchVisionResNet2DClassifier",
    "SwinV2Tiny2DClassifier",
    "MILSwinV2TinyClassifier",
]
