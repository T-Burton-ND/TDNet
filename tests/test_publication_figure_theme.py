import json
from pathlib import Path

import matplotlib.pyplot as plt

from gridiron_ml.publication.figure_theme import (
    GRIDIRON_PALETTE,
    TDNET_COLORS,
    apply_tdnet_theme,
)


def test_shared_figure_theme_uses_the_committed_palette():
    palette_path = Path(__file__).resolve().parents[1] / "gridiron.palette.json"
    palette = json.loads(palette_path.read_text(encoding="utf-8"))
    assert GRIDIRON_PALETTE == palette
    assert TDNET_COLORS["parchment"] == palette["colors"]["parchment"]
    assert TDNET_COLORS["figure_primary"] == palette["colors"]["projectorBlue"]

    apply_tdnet_theme()
    assert plt.rcParams["figure.facecolor"] == palette["colors"]["parchment"]
    assert plt.rcParams["savefig.facecolor"] == palette["colors"]["parchment"]
    assert plt.rcParams["axes.facecolor"] == palette["colors"]["parchmentPanel"]
