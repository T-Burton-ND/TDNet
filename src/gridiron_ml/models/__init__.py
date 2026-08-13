"""src.gridiron_ml.models.__init__.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Define model wrappers and checkpoint helpers behind the shared TDNet interface.
"""

from .base import TDModel, validate_model_contract
from .td_boosted import TDBoosted
from .td_ensemble import TDEnsemble
from .td_linear import TDLinear
from .td_kernel import TDKernel
from .td_knn import TDKNN
from .td_mlp import TDMLP, TDStructuredMLP
from .td_naive import TDNaive
from .td_spline import TDSpline
from .td_stat import TDStat, TDStatPercentile, TDStatRobust, TDStatWeighted
from .td_tree import TDTree
from .td_temporal import TDTemporal
from .checkpoints import load_model_checkpoint
from .features import ModelFeatureAdapter
from .names import (
    model_family_label,
    model_label,
    normalize_identifier,
    normalize_model_family,
)
from .registry import (
    MODEL_REGISTRY,
    build_model_from_config,
    get_model_class,
    register_model_family,
)

register_model_family("linear", TDLinear, aliases=["tdlinear"])
register_model_family("stat", TDStat, aliases=["tdstat"])
register_model_family("tree", TDTree, aliases=["tdtree", "td_tree"])
register_model_family("naive", TDNaive, aliases=["tdnaive", "td_naive"])
register_model_family("spline", TDSpline, aliases=["tdspline", "td_spline", "gam"])
register_model_family("boosted", TDBoosted, aliases=["tdboosted", "td_boosted", "gbm"])
register_model_family("neural", TDMLP, aliases=["tdmlp", "td_mlp", "mlp"])
register_model_family(
    "structured_neural",
    TDStructuredMLP,
    aliases=["structured_mlp", "td_structured_mlp"],
)
register_model_family("ensemble", TDEnsemble, aliases=["tdensemble", "td_ensemble"])
register_model_family("kernel", TDKernel, aliases=["tdkernel", "td_kernel"])
register_model_family("temporal", TDTemporal, aliases=["tdtemporal", "td_temporal"])
register_model_family("knn", TDKNN, aliases=["tdknn", "td_knn"])

__all__ = [
    "MODEL_REGISTRY",
    "ModelFeatureAdapter",
    "TDModel",
    "TDBoosted",
    "TDEnsemble",
    "TDLinear",
    "TDKernel",
    "TDKNN",
    "TDMLP",
    "TDNaive",
    "TDSpline",
    "TDStat",
    "TDStatPercentile",
    "TDStatRobust",
    "TDStatWeighted",
    "TDTree",
    "TDTemporal",
    "TDStructuredMLP",
    "build_model_from_config",
    "get_model_class",
    "load_model_checkpoint",
    "model_family_label",
    "model_label",
    "normalize_identifier",
    "normalize_model_family",
    "register_model_family",
    "validate_model_contract",
]
