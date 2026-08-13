"""Central publication palette and small plotting helpers for TDNet figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


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


def apply_tdnet_theme() -> None:
    """Apply the shared journal-oriented defaults to the current matplotlib process."""
    plt.rcParams.update(
        {
            "axes.facecolor": TDNET_COLORS["white"],
            "figure.facecolor": TDNET_COLORS["white"],
            "axes.edgecolor": TDNET_COLORS["slate"],
            "axes.labelcolor": TDNET_COLORS["midnight_gridiron"],
            "axes.titlecolor": TDNET_COLORS["midnight_gridiron"],
            "xtick.color": TDNET_COLORS["slate"],
            "ytick.color": TDNET_COLORS["slate"],
            "font.size": 9,
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
