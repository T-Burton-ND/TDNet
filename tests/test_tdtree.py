import numpy as np
import pandas as pd
import yaml

from gridiron_ml.td_run import TDEval
from gridiron_ml.models import TDTree, build_model_from_config


def test_tdtree_from_yaml_uses_variant_and_defaults(tmp_path):
    config_path = tmp_path / "tree.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_name": "smoke_tree",
                "model_type": "rf",
                "params": {"n_estimators": 7},
            }
        ),
        encoding="utf-8",
    )

    model = TDTree.from_yaml(config_path)

    assert model.model_family == "tree"
    assert model.model_name == "smoke_tree"
    assert model.model_type == "random_forest"
    assert model.params["n_estimators"] == 7
    assert model.loss_function == "WinnerUpsetAccuracy"


def test_tdtree_hybrid_winner_accuracy_alias_uses_winner_upset_objective():
    model = TDTree({"loss_function": "hybrid_winner_accuracy"})

    assert model.loss_function == "WinnerUpsetAccuracy"


def test_registry_builds_tdtree_from_config_path(tmp_path):
    config_path = tmp_path / "tree.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_name": "eval_tree",
                "model_type": "extra_trees",
                "params": {"n_estimators": 5, "random_state": 9},
            }
        ),
        encoding="utf-8",
    )

    model = build_model_from_config({"family": "tdtree", "config_path": str(config_path)})

    assert isinstance(model, TDTree)
    assert model.model_name == "eval_tree"
    assert model.model_type == "extra_trees"


def test_tdtree_training_prediction_and_feature_importance_smoke():
    X = pd.DataFrame(
        {
            "offense_ppa_diff": [-0.6, -0.2, 0.1, 0.3, 0.8, 1.1],
            "defense_ppa_diff": [0.4, 0.1, -0.2, -0.1, -0.5, -0.7],
            "special_teams_diff": [0.0, 0.1, np.nan, -0.1, 0.3, 0.2],
        }
    )
    y = pd.Series([-14.0, -7.0, -1.0, 3.0, 13.0, 21.0])
    model = TDTree(
        {
            "model_type": "random_forest",
            "params": {"n_estimators": 8, "min_samples_leaf": 1, "random_state": 3, "n_jobs": 1},
        }
    )

    model.train(X, y, X_val=X.tail(2), y_val=y.tail(2))
    pred = model.predict(X, meta_df=pd.DataFrame({"keys_team_home": list("ABCDEF")}))
    importance = model.get_feature_importance()

    assert model.is_trained_
    assert len(pred) == len(X)
    assert {"pred_margin", "pred_proba_home_win", "pred_pick_home", "keys_team_home"}.issubset(pred.columns)
    assert np.isfinite(pred["pred_margin"]).all()
    assert not importance.empty
    assert set(importance.columns) == {"feature", "importance"}
    assert {"train", "val"} == set(model.training_history_["split"])


def test_evaluator_saves_tdtree_feature_importance(tmp_path):
    X = pd.DataFrame({"a_diff": [-1.0, 0.0, 1.0, 2.0], "b_diff": [2.0, 1.0, 0.0, -1.0]})
    y = pd.Series([-7.0, -1.0, 3.0, 10.0])
    model = TDTree({"model_type": "extra_trees", "params": {"n_estimators": 5, "random_state": 4, "n_jobs": 1}})
    model.train(X, y)

    evaluator = TDEval({}, fingerprints=object(), matchup_builder=object(), model=model)
    output_root = evaluator.save_outputs(tmp_path)

    assert (output_root / "feature_importance" / "feature_importance.csv").exists()


def test_evaluator_builds_tree_model_from_simplified_config(tmp_path):
    config_path = tmp_path / "tree.yaml"
    config_path.write_text(yaml.safe_dump({"model_name": "eval_tree", "model_type": "gradient_boosting"}), encoding="utf-8")

    evaluator = TDEval(
        {
            "fingerprints": {"version": 0, "root": "."},
            "model": {"family": "tree", "config_path": str(config_path)},
        },
        fingerprints=object(),
        matchup_builder=object(),
    )

    assert isinstance(evaluator.model, TDTree)
    assert evaluator.model.model_name == "eval_tree"
    assert evaluator.model.model_type == "gradient_boosted"
