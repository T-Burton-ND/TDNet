import pandas as pd

from gridiron_ml.td_run import TDEval


class ArtifactModel:
    model_family = "dummy"
    model_type = "artifact_smoke"
    model_name = "lasso"
    loss_function = "WinnerAccuracy"

    def __init__(self):
        self.training_history_ = pd.DataFrame(
            {
                "epoch": [1, 2, 3, 1, 2, 3],
                "split": ["train", "train", "train", "val", "val", "val"],
                "optimized_loss": [12.0, 8.0, 5.5, 12.5, 8.8, 6.4],
            }
        )


def test_save_outputs_writes_training_curve_png(tmp_path):
    evaluator = TDEval(
        {"fingerprints": {"version": 0, "root": "."}},
        fingerprints=object(),
        matchup_builder=object(),
        model=ArtifactModel(),
    )

    output_root = evaluator.save_outputs(tmp_path)

    history_csv = output_root / "history" / "training_history.csv"
    training_curve = output_root / "history" / "training_curve.png"

    assert history_csv.exists()
    assert training_curve.exists()
    assert training_curve.stat().st_size > 0


def test_save_outputs_ignores_legacy_loss_partition_flag(tmp_path):
    evaluator = TDEval(
        {
            "fingerprints": {"version": 0, "root": "."},
            "eval": {"split_artifacts_by_loss": True},
        },
        fingerprints=object(),
        matchup_builder=object(),
        model=ArtifactModel(),
    )

    output_root = evaluator.save_outputs(tmp_path)

    assert output_root == tmp_path
    assert (output_root / "history" / "training_history.csv").exists()
