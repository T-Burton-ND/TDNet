"""src.gridiron_ml.models.registry.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Define model wrappers and checkpoint helpers behind the shared TDNet interface.
"""

from .names import normalize_model_family


MODEL_REGISTRY = {}


def register_model_family(name, model_cls, aliases=None):
    """Run the register_model_family step and return its normalized result."""
    names = [name] + list(aliases or [])
    for item in names:
        key = normalize_model_family(item)
        if not key:
            raise ValueError("Model family names must be non-empty.")
        MODEL_REGISTRY[key] = model_cls
    return model_cls


def get_model_class(name):
    """Run the get_model_class step and return its normalized result."""
    key = normalize_model_family(name)
    if key not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model family='{name}'. Supported: {supported}")
    return MODEL_REGISTRY[key]


def build_model_from_config(model_cfg, *, members=None):
    """Run the build_model_from_config step and return its normalized result."""
    model_cfg = dict(model_cfg or {})
    family = normalize_model_family(model_cfg.get("family", "linear"))
    config_ref = model_cfg.get("config_path")
    model_cls = get_model_class(family)

    if config_ref and hasattr(model_cls, "from_yaml"):
        return model_cls.from_yaml(config_ref)
    if members is not None:
        return model_cls(model_cfg, members=members)
    return model_cls(model_cfg)
