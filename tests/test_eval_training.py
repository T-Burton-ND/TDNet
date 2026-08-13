from gridiron_ml.td_run import DEFAULT_MODEL_SPECS, TDRun, filter_model_specs
from gridiron_ml.td_run.training import (
    build_eval_config,
    checkpoint_path,
    clear_model_run_dir,
    model_run_dir,
)


def test_filter_model_specs_applies_family_and_name_filters():
    specs = filter_model_specs(
        DEFAULT_MODEL_SPECS,
        train_stat=False,
        train_linear=True,
        train_tree=False,
        only_names=["ridge", "random_forest"],
    )

    assert [spec.name for spec in specs] == ["ridge"]
    assert specs[0].family == "linear"


def test_training_path_helpers_keep_artifact_layout(tmp_path):
    spec = filter_model_specs(
        DEFAULT_MODEL_SPECS,
        train_stat=True,
        train_linear=False,
        train_tree=False,
        only_names=["stat_z_index"],
    )[0]

    run_dir = model_run_dir(spec, models_root=tmp_path / "models")
    checkpoint = checkpoint_path(spec, models_root=tmp_path / "models")

    assert run_dir == tmp_path / "models" / "stat" / "models" / "stat_z_index"
    assert checkpoint == run_dir / "models" / "tdstat_stat_z_index.pkl"


def test_clear_model_run_dir_removes_only_selected_model_artifacts(tmp_path):
    selected = filter_model_specs(
        DEFAULT_MODEL_SPECS,
        train_stat=False,
        train_linear=True,
        train_tree=False,
        only_names=["ridge"],
    )[0]
    other = filter_model_specs(
        DEFAULT_MODEL_SPECS,
        train_stat=False,
        train_linear=True,
        train_tree=False,
        only_names=["lasso"],
    )[0]
    models_root = tmp_path / "models"
    selected_file = model_run_dir(selected, models_root=models_root) / "stale.csv"
    other_file = model_run_dir(other, models_root=models_root) / "keep.csv"
    selected_file.parent.mkdir(parents=True)
    other_file.parent.mkdir(parents=True)
    selected_file.write_text("old", encoding="utf-8")
    other_file.write_text("keep", encoding="utf-8")

    cleared = clear_model_run_dir(selected, models_root=models_root)

    assert cleared == model_run_dir(selected, models_root=models_root)
    assert not selected_file.exists()
    assert other_file.exists()


def test_build_eval_config_centralizes_model_training_config(tmp_path):
    spec = filter_model_specs(
        DEFAULT_MODEL_SPECS,
        train_stat=False,
        train_linear=True,
        train_tree=False,
        only_names=["ridge"],
    )[0]

    config = build_eval_config(
        spec,
        project_root=tmp_path,
        fingerprint_version=0,
        postseason=False,
        train_years=[2010, 2011],
        test_years=[2024],
        matchup_config={"representation": "unit_matchup"},
        models_root=tmp_path / "models",
    )

    assert config["fingerprints"] == {
        "version": 0,
        "postseason": False,
        "root": str(tmp_path),
    }
    assert config["model"] == {
        "family": "linear",
        "config_path": str(tmp_path / "configs/models/linear/config_ridge.yaml"),
    }
    assert config["eval"]["train_years"] == [2010, 2011]
    assert config["eval"]["test_years"] == [2024]


def test_td_run_resolves_configured_model_families(tmp_path):
    runner = TDRun(
        {
            "root": str(tmp_path),
            "models": {
                "families": ["linear"],
                "names": ["ridge", "random_forest"],
            },
        }
    )

    specs = runner.selected_model_specs()

    assert [spec.name for spec in specs] == ["ridge"]
