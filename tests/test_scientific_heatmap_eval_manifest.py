import pandas as pd

from gridiron_ml.cli.publication.build_scientific_heatmap_eval_manifest import choose_source


def test_heatmap_eval_routes_corrected_and_legacy_tiers():
    old = pd.DataFrame({"source": ["old"]})
    f7 = pd.DataFrame({"source": ["f7"]})
    corrected = pd.DataFrame({"source": ["corrected"]})
    assert choose_source("F0", legacy_roster=old, legacy_f7=f7, corrected=corrected).iloc[0, 0] == "old"
    assert choose_source("F7", legacy_roster=old, legacy_f7=f7, corrected=corrected).iloc[0, 0] == "f7"
    assert choose_source("F6", legacy_roster=old, legacy_f7=f7, corrected=corrected).iloc[0, 0] == "corrected"
