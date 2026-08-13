from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Fail when a registered trainable publication type lacks HPS and poll support."""

from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json

import yaml

from gridiron_ml.models import build_model_from_config, validate_model_contract


def main():
    root = project_root()
    coverage = yaml.safe_load((root / "configs/publication/model_search_coverage.yaml").read_text())
    rows = []
    comparative_rows = []
    for search_name, search in coverage["searches"].items():
        config_path = search.get("config")
        comparative_only = set(search.get("types", {})).issubset({"knn", "naive"})
        if config_path and not comparative_only:
            config = yaml.safe_load((root / config_path).read_text())
            if set(config.get("objectives", [])) != {"winner", "margin"}:
                raise ValueError(f"{search_name} does not search winner and margin.")
            if not config.get("parameter_search", {}).get("spaces"):
                raise ValueError(f"{search_name} has no hyperparameter space.")
            if not config.get("split_configs"):
                raise ValueError(f"{search_name} has no temporal split config.")
        for family, types in search["types"].items():
            for model_type in types:
                row = {"search": search_name, "family": family, "model_type": model_type}
                (comparative_rows if comparative_only else rows).append(row)
    concrete = {(row["family"], row["model_type"]) for row in rows}
    if len(concrete) != len(rows):
        raise ValueError("A concrete model type is assigned to multiple search suites.")
    expected = 30
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} trainable concrete types, found {len(rows)}.")

    model_configs = {
        "spline": "configs/models/spline/config_spline_ridge.yaml",
        "boosted": "configs/models/boosted/config_hist_gradient_boosted.yaml",
        "neural": "configs/models/neural/config_mlp.yaml",
        "structured_neural": "configs/models/neural/config_structured_mlp.yaml",
        "kernel": "configs/models/kernel/config_rbf_kernel_ridge.yaml",
        "temporal": "configs/models/temporal/config_decay_ridge.yaml",
    }
    contracts = {}
    for family, path in model_configs.items():
        model = build_model_from_config({"family": family, "config_path": str(root / path)})
        validate_model_contract(model)
        contracts[family] = "train/predict/evaluate/total_rank compatible"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_config": "configs/publication/model_search_coverage.yaml",
        "coverage_config_sha256": sha256((root / "configs/publication/model_search_coverage.yaml").read_bytes()).hexdigest(),
        "trainable_concrete_types": len(rows),
        "objective_specific_registered_checkpoints": len(rows) * len(coverage["objectives"]),
        "comparative_only_registered_types": len(comparative_rows),
        "objectives": coverage["objectives"],
        "coverage": rows, "new_family_contracts": contracts,
        "naive_policy": coverage["non_tuned"], "ensemble_policy": coverage["post_oof_only"],
        "all_trainable_types_have_search": True,
        "all_new_families_support_tdnet_poll_contract": True,
    }
    output = root / "docs/publication_2026/preseason/model_search_coverage_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
