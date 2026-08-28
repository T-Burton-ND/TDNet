"""Central publication palette and small plotting helpers for TDNet figures."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

TDNET_COLORS = {
    "midnight_gridiron": "#11214F",
    "edge_pink": "#FF5FA2",
    "ion_blue": "#1EA7FF",
    "electric_emerald": "#00C853",
    "gridiron_violet": "#6A37C8",
    "soft_mint": "#4ED8BD",
    "signal_orange": "#E69F00",
    "deep_teal": "#007C83",
    "slate": "#687386",
    "medium_gray": "#9AA1AA",
    "polar_mist": "#E6E9ED",
    "white": "#FFFFFF",
}


def _register_publication_fonts() -> str:
    """Register the locally installed Aptos family, with a portable fallback."""
    configured = os.environ.get("TDNET_PUBLICATION_FONT_DIR")
    font_dir = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[3] / "data" / "fonts" / "aptos"
    )
    if not font_dir.is_dir():
        return "DejaVu Sans"
    # Aptos Display advertises the same family name as Aptos Text. Excluding
    # it keeps Matplotlib from choosing display-cut glyphs for body copy.
    font_paths = [
        path for path in sorted(font_dir.glob("*.ttf"))
        if not path.name.startswith("Aptos-Display")
    ]
    for font_path in font_paths:
        font_manager.fontManager.addfont(font_path)
    return "Aptos"


def apply_tdnet_theme() -> None:
    """Apply the shared journal-oriented defaults to the current matplotlib process."""
    font_family = _register_publication_fonts()
    plt.rcParams.update(
        {
            "axes.facecolor": TDNET_COLORS["white"],
            "figure.facecolor": TDNET_COLORS["white"],
            "axes.edgecolor": TDNET_COLORS["slate"],
            "axes.labelcolor": TDNET_COLORS["midnight_gridiron"],
            "axes.titlecolor": TDNET_COLORS["midnight_gridiron"],
            "xtick.color": TDNET_COLORS["slate"],
            "ytick.color": TDNET_COLORS["slate"],
            "font.family": font_family,
            "font.monospace": ["Aptos Mono", "DejaVu Sans Mono"],
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.color": TDNET_COLORS["polar_mist"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
        }
    )


MODEL_COLORS = {
    "strict_oof_knn_f6_v1_4": TDNET_COLORS["ion_blue"],
    "opponent_adjusted_point_differential_v1_4": TDNET_COLORS["electric_emerald"],
    "all_model_consensus": TDNET_COLORS["midnight_gridiron"],
    "compact_consensus": TDNET_COLORS["soft_mint"],
    "vegas_declared_line": TDNET_COLORS["gridiron_violet"],
    "season_to_date_raw_point_differential": TDNET_COLORS["deep_teal"],
    "season_to_date_opponent_adjusted_point_differential": TDNET_COLORS["electric_emerald"],
    "season_to_date_win_rate": TDNET_COLORS["soft_mint"],
    "home_team_prior": TDNET_COLORS["slate"],
    "random_50": TDNET_COLORS["medium_gray"],
}


# Canonical domains consumed by figure builders and chart-contract tests.
CHART_DOMAINS = {
    "probability": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "brier": (0.0, 1.0),
    "calibration": (0.0, 1.0),
    "random_accuracy": (0.0, 1.0),
}
