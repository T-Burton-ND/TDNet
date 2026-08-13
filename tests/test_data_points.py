import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

from gridiron_ml.td_run.data_points import (
    data_point_options,
    load_data_point_catalog,
    plot_data_point_logo_scatter,
    resolve_data_point,
)


def test_data_point_catalog_resolves_names_columns_and_labels():
    catalog = load_data_point_catalog()

    assert "offense_ppa" in catalog
    assert catalog["offense_ppa"].column == "offense_ppa"
    assert resolve_data_point("Offense PPA", catalog).name == "offense_ppa"
    assert resolve_data_point("defense_ppa", catalog).label == "Defense PPA"

    options = data_point_options(catalog)
    assert {"name", "column", "label", "group"}.issubset(options.columns)


def test_logo_scatter_accepts_small_frame_without_logos(tmp_path):
    frame = pd.DataFrame(
        {
            "keys_team": ["Alpha", "Beta"],
            "offense_ppa": [0.12, 0.24],
            "defense_ppa": [-0.05, 0.02],
        }
    )

    fig, ax, plot_df = plot_data_point_logo_scatter(
        "offense_ppa",
        "defense_ppa",
        season=2025,
        week=1,
        root=tmp_path,
        frame=frame,
        logo_dir=tmp_path / "missing_logos",
        palette_path=tmp_path / "missing_palette.csv",
        annotate_missing_logos=True,
    )

    assert plot_df.shape[0] == 2
    assert ax.get_xlabel() == "Offense PPA"
    assert ax.get_ylabel() == "Defense PPA"
    fig.clf()
