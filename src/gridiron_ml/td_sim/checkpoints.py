"""src.gridiron_ml.td_sim.checkpoints.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from pathlib import Path

from gridiron_ml.models import (
    load_model_checkpoint,
    model_label,
    normalize_model_family,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def discover_model_checkpoints(
    models_root=None, include_models="all", exclude_models=None
):
    """Load model checkpoints and return normalized model specs for TD Sim."""
    root = _resolve(models_root or "models")
    exclude = {
        str(model).strip() for model in (exclude_models or []) if str(model).strip()
    }
    include = _normalize_include(include_models)
    specs = []

    for path in sorted(root.glob("**/*.pkl")):
        path_name = _path_model_name(path)
        if include is not None and path_name not in include:
            continue
        if path_name in exclude:
            continue
        model = load_model_checkpoint(path)
        name = _model_name(model, path)
        if include is not None and name not in include and path_name not in include:
            continue
        if name in exclude:
            continue
        specs.append(
            {
                "name": name,
                "path": path,
                "model": model,
                "family": normalize_model_family(
                    getattr(model, "model_family", _infer_family(path))
                ),
            }
        )

    if include is not None:
        order = {name: idx for idx, name in enumerate(include)}
        specs = sorted(
            specs, key=lambda spec: (order.get(spec["name"], len(order)), spec["name"])
        )
    return specs


def _normalize_include(include_models):
    """Internal helper for the normalize_include step."""
    if include_models in (None, "all"):
        return None
    if isinstance(include_models, str):
        return [include_models.strip()] if include_models.strip() else None
    include = [str(model).strip() for model in include_models if str(model).strip()]
    return include or None


def _model_name(model, path):
    """Internal helper for the model_name step."""
    name = model_label(model)
    if name:
        return name
    return _path_model_name(path)


def _path_model_name(path):
    """Infer a checkpoint model name from the file path before opening it."""

    stem = Path(path).stem
    for prefix in ["tdlinear_", "tdstat_", "tdtree_"]:
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _infer_family(path):
    """Internal helper for the infer_family step."""
    parts = [part.lower() for part in Path(path).parts]
    if "linear" in parts:
        return "linear"
    if "stat" in parts:
        return "stat"
    if "tree" in parts:
        return "tree"
    return "model"


def _resolve(path):
    """Internal helper for the resolve step."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
