#!/usr/bin/env python3
"""Resolve exact selected raw columns and PCA inputs from frozen transformers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np

from constraint_free_search import digest, write_json

FREEZE = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search/final_fit/freeze/freeze_manifest.json")
OUTPUT = FREEZE.parent / "resolved_feature_provenance.json"


def build(freeze_path: Path = FREEZE, output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError("Resolved feature provenance is immutable")
    frozen = json.loads(freeze_path.read_text())
    cells = []
    for cell in frozen["cells"]:
        bundle_path = Path(cell["model_bundle"])
        if digest(bundle_path) != cell["model_bundle_sha256"]:
            raise ValueError(f"Frozen model changed: {bundle_path}")
        with bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
        names = np.asarray(bundle["source_feature_names"])
        transform = bundle["transformer"]
        if hasattr(transform, "columns_") and hasattr(transform, "kept_"):
            upstream_indices = transform.columns_[transform.kept_]
            upstream = names[upstream_indices].tolist()
            selector = getattr(transform, "selector_", None)
            selected_raw = (names[upstream_indices[selector.get_support()]].tolist()
                            if selector is not None else
                            upstream if getattr(transform, "mode_", None) == "raw" else [])
            pca_inputs = (upstream if getattr(transform, "pca_", None) is not None else [])
            family_pca_inputs = {
                family: names[upstream_indices[indices]].tolist()
                for family, indices, _, _ in getattr(transform, "family_blocks_", [])}
        else:
            upstream = list(cell["selected_source_features"])
            selected_raw = upstream if not cell["pca_loading_blocks"] else []
            pca_inputs = upstream if cell["pca_loading_blocks"] else []
            family_pca_inputs = {}
        if cell["tier"] == "F18" and any("market" in name.lower() for name in upstream):
            raise ValueError("Market predictor entered F18 frozen representation")
        cells.append({
            "tier": cell["tier"], "architecture": cell["architecture"],
            "model_bundle_sha256": cell["model_bundle_sha256"],
            "representation_features": cell["representation_features"],
            "pre_reducer_source_features": upstream,
            "selected_raw_features": selected_raw,
            "global_pca_input_features": pca_inputs,
            "family_pca_input_features": family_pca_inputs,
            "market_as_residual_anchor": bool(bundle["tier"] == "F19" and
                                              bundle["genome"] and
                                              bundle["genome"]["target_mode"] == "residual"),
            "note": "The freeze field selected_source_features records upstream columns for advanced selectors; selected_raw_features here applies the fitted selector mask.",
        })
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "freeze_manifest": str(freeze_path),
              "freeze_manifest_sha256": digest(freeze_path),
              "cells": cells, "models_changed": False}
    write_json(output, result)
    return {"path": str(output), "sha256": digest(output), "cells": len(cells)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
