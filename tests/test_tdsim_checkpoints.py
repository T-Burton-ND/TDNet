import pickle
from types import SimpleNamespace

from gridiron_ml.td_sim.checkpoints import discover_model_checkpoints


def test_discover_model_checkpoints_prefilters_included_names_before_loading(tmp_path):
    ridge_path = (
        tmp_path
        / "models"
        / "linear"
        / "models"
        / "ridge"
        / "models"
        / "tdlinear_ridge.pkl"
    )
    bad_lasso_path = (
        tmp_path
        / "models"
        / "linear"
        / "models"
        / "lasso"
        / "models"
        / "tdlinear_lasso.pkl"
    )
    ridge_path.parent.mkdir(parents=True)
    bad_lasso_path.parent.mkdir(parents=True)
    with ridge_path.open("wb") as handle:
        pickle.dump(SimpleNamespace(model_family="linear", model_name="ridge"), handle)
    bad_lasso_path.write_bytes(b"not a pickle")

    specs = discover_model_checkpoints(
        models_root=tmp_path / "models",
        include_models=["ridge"],
    )

    assert [spec["name"] for spec in specs] == ["ridge"]


def test_discover_model_checkpoints_prefilters_excluded_names_before_loading(tmp_path):
    bad_ridge_path = (
        tmp_path
        / "models"
        / "linear"
        / "models"
        / "ridge"
        / "models"
        / "tdlinear_ridge.pkl"
    )
    lasso_path = (
        tmp_path
        / "models"
        / "linear"
        / "models"
        / "lasso"
        / "models"
        / "tdlinear_lasso.pkl"
    )
    bad_ridge_path.parent.mkdir(parents=True)
    lasso_path.parent.mkdir(parents=True)
    bad_ridge_path.write_bytes(b"not a pickle")
    with lasso_path.open("wb") as handle:
        pickle.dump(SimpleNamespace(model_family="linear", model_name="lasso"), handle)

    specs = discover_model_checkpoints(
        models_root=tmp_path / "models",
        include_models="all",
        exclude_models=["ridge"],
    )

    assert [spec["name"] for spec in specs] == ["lasso"]
