"""Model factory + per-family classifier wrappers.

Public re-exports keep the historical ``from sclc.models import X`` flat surface
intact, even though the wrappers now live in topic-specific submodules.
"""

from .factory import (
    get_sclc_model,
    get_pipeline,
    is_2d_model_type,
    is_mil_model_type,
    TWO_D_MODEL_TYPES,
    MIL_MODEL_TYPES,
)
from .swin_unetr import SwinUNETRClassifier
from .classifiers_2d import (
    EfficientNet2DClassifier,
    DenseNet2DClassifier,
    TorchVisionResNet2DClassifier,
    SwinTiny2DClassifier,
)
from .classifiers_rin import (
    RadImageNetResNet502DClassifier,
    RadImageNetDenseNet1212DClassifier,
)
from .classifiers_mil import MILResNet50Classifier, MILSwinTinyClassifier

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
    "SwinTiny2DClassifier",
    "RadImageNetResNet502DClassifier",
    "RadImageNetDenseNet1212DClassifier",
    "MILResNet50Classifier",
    "MILSwinTinyClassifier",
]
