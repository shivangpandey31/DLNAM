"""
dlnam — core package.

INVARIANT: this package must never import from `dlnam_sim`. The dependency
arrow points one way only (dlnam_sim -> dlnam), which is what keeps the
simulation / Monte-Carlo machinery a true add-on: deleting `dlnam_sim` leaves
the core fully functional, and the core has zero awareness it exists.
"""

__version__ = "0.1.0"

from .config import (
    ActivationSpec, TransformSpec, SoftCapSpec, ExUSpec, LayerSpec,
    TermSpec, InitSpec, SurfaceTermSpec, SmoothTermSpec, TrendTermSpec,
    CategoricalTermSpec, ModelConfig, TrainConfig, categorical_terms_from_configs, broadcast_terms,
)
from .links import Link, LogLink, LogitLink, IdentityLink, make_link
from .terms.base import AdditiveTerm, Centering, EffectCurve
from .terms.surface import SurfaceTerm
from .terms.smooth import SmoothTerm, TrendTerm
from .terms.categorical import CategoricalTerm
from .model import DLNAM
from .train import Trainer, print_gpu_stats
from .data import DataProcessor, PreparedData
from .evaluate import PerformanceEvaluator
from .visualize import ResultVisualizer
from .inference import (
    EffectEstimate, EffectExtractor, UncertaintyMethod, IntervalUQ,
    ConfidenceIntervalUQ, PredictionIntervalUQ, EnsembleIntervalUQ,
    needs_laplace,
)

__all__ = [
    "__version__",
    "ActivationSpec", "TransformSpec", "SoftCapSpec", "ExUSpec", "LayerSpec",
    "TermSpec", "InitSpec", "SurfaceTermSpec", "SmoothTermSpec", "TrendTermSpec",
    "CategoricalTermSpec", "ModelConfig", "TrainConfig", "categorical_terms_from_configs", "broadcast_terms",
    "Link", "LogLink", "LogitLink", "IdentityLink", "make_link",
    "AdditiveTerm", "Centering", "EffectCurve", "DLNAM", "Trainer", "print_gpu_stats", "DataProcessor", "PreparedData", "PerformanceEvaluator", "ResultVisualizer",
    "SurfaceTerm", "SmoothTerm", "TrendTerm", "CategoricalTerm",
    "EffectEstimate", "EffectExtractor", "UncertaintyMethod",
    "IntervalUQ", "ConfidenceIntervalUQ", "PredictionIntervalUQ",
    "EnsembleIntervalUQ", "needs_laplace",
]
