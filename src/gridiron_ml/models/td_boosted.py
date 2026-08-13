"""Dedicated M4 boosted-tree alias over the established TDTree implementation."""

from .names import normalize_identifier, normalize_model_family
from .td_tree import TDTree


class TDBoosted(TDTree):
    """Gradient-boosted TDTree with a distinct publication model family."""

    def __init__(self, config=None):
        loaded = self._load_config(config)
        loaded.setdefault("model_type", "hist_gradient_boosted")
        loaded.setdefault("model_name", loaded["model_type"])
        super().__init__(loaded)
        self.model_family = normalize_model_family("boosted")
        self.model_name = normalize_identifier(loaded.get("model_name"))
