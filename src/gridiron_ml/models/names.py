"""src.gridiron_ml.models.names.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Define model wrappers and checkpoint helpers behind the shared TDNet interface.
"""

import re


MODEL_FAMILY_ALIASES = {
    "linear": "linear",
    "tdlinear": "linear",
    "td_linear": "linear",
    "stat": "stat",
    "tdstat": "stat",
    "td_stat": "stat",
    "tree": "tree",
    "tdtree": "tree",
    "td_tree": "tree",
    "naive": "naive",
    "tdnaive": "naive",
    "td_naive": "naive",
    "spline": "spline",
    "tdspline": "spline",
    "td_spline": "spline",
    "boosted": "boosted",
    "tdboosted": "boosted",
    "td_boosted": "boosted",
    "neural": "neural",
    "mlp": "neural",
    "tdmlp": "neural",
    "td_mlp": "neural",
    "structured_neural": "structured_neural",
    "structured_mlp": "structured_neural",
    "ensemble": "ensemble",
    "tdensemble": "ensemble",
    "td_ensemble": "ensemble",
    "kernel": "kernel",
    "tdkernel": "kernel",
    "td_kernel": "kernel",
    "temporal": "temporal",
    "tdtemporal": "temporal",
    "td_temporal": "temporal",
    "knn": "knn",
    "tdknn": "knn",
    "td_knn": "knn",
}

MODEL_FAMILY_LABELS = {
    "linear": "Linear",
    "stat": "Stat",
    "tree": "Tree",
    "naive": "Naive",
    "spline": "Spline",
    "boosted": "Boosted Tree",
    "neural": "Neural",
    "structured_neural": "Structured Neural",
    "ensemble": "Ensemble",
    "kernel": "Kernel",
    "temporal": "Temporal",
    "knn": "Historical KNN",
}


def normalize_identifier(value, *, default=None):
    """Run the normalize_identifier step and return its normalized result."""
    if value is None:
        if default is None:
            return None
        value = default
    text = str(value).strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text and default is not None:
        return normalize_identifier(default)
    return text


def normalize_model_family(value, *, default="linear"):
    """Run the normalize_model_family step and return its normalized result."""
    key = normalize_identifier(value, default=default)
    return MODEL_FAMILY_ALIASES.get(key, key)


def model_family_label(value):
    """Run the model_family_label step and return its normalized result."""
    family = normalize_model_family(value)
    return MODEL_FAMILY_LABELS.get(family, family.replace("_", " ").title())


def model_label(model, idx=0):
    """Run the model_label step and return its normalized result."""
    name = normalize_identifier(getattr(model, "model_name", None))
    if name:
        return name

    family = normalize_model_family(getattr(model, "model_family", model.__class__.__name__))
    model_type = normalize_identifier(getattr(model, "model_type", None))
    if model_type:
        return f"{family}_{model_type}" if family == "stat" else model_type
    return f"{family}_{int(idx) + 1}"
