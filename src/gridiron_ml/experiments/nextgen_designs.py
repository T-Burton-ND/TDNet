"""Deterministic interpretable F06 designs, frozen before outcome screening."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nextgen_contract import validate_feature_manifest
from gridiron_ml.td_run.matchups.unit_matchups import default_counterpart
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Formula:
    name: str
    inputs: tuple[str, ...]
    operation: str
    interpretation: str
    units: str

    @property
    def excel(self):
        refs = [f"[{x}]" for x in self.inputs]
        if self.operation == "identity":
            return "=" + refs[0]
        if self.operation == "mean":
            # Require every input rather than silently change the equation.
            return f'=IF(COUNT({",".join(refs)})={len(refs)},AVERAGE({",".join(refs)}),NA())'
        if self.operation == "quarter_slope":
            return f"=(-3*{refs[0]}-{refs[1]}+{refs[2]}+3*{refs[3]})/10"
        if self.operation == "difference":
            return "=" + refs[0] + "-" + refs[1]
        if self.operation == "product":
            return "=" + "*".join(refs)
        if self.operation == "ratio":
            return f"=IF({refs[1]}>0,{refs[0]}/{refs[1]},NA())"
        raise ValueError(f"Unknown formula: {self.operation}")

    def evaluate(self, frame):
        values = frame.loc[:, list(self.inputs)].apply(pd.to_numeric, errors="coerce")
        if self.operation == "identity":
            return values.iloc[:, 0]
        if self.operation == "mean":
            return values.mean(axis=1, skipna=False)
        if self.operation == "quarter_slope":
            return values.mul([-0.3, -0.1, 0.1, 0.3]).sum(axis=1, min_count=4)
        a, b = values.iloc[:, 0], values.iloc[:, 1]
        if self.operation == "difference":
            return a-b
        if self.operation == "product":
            return a*b
        if self.operation == "ratio":
            return a/b.where(b.gt(0))
        raise ValueError(self.operation)


def baseline_formulas(features: list[str], design: str) -> list[Formula]:
    if design not in "abc" or len(design) != 1:
        raise ValueError("Unknown design")
    formulas = {n: Formula(n, (n,), "identity", "Inherited canonical F6 source state; see frozen source formula and implementation.",
                           "inherited_source_units") for n in features}
    if design == "a":
        return list(formulas.values())
    if design == "b":
        for side in ("offense", "defense"):
            candidates = [
                Formula(f"f06_{side}_productive_success", (f"{side}_success_rate", f"{side}_explosiveness"), "product",
                        "Frequency of successful plays weighted by the explosiveness of successful plays; sustained and explosive movement together.", "explosiveness_per_play"),
                Formula(f"f06_{side}_passing_rushing_ppa_gap", (f"{side}_passing_plays_ppa", f"{side}_rushing_plays_ppa"), "difference",
                        "Passing versus rushing production gap; exposes a one-dimensional attack or defensive vulnerability.", "points_per_play"),
                Formula(f"f06_{side}_passing_rushing_success_gap", (f"{side}_passing_plays_success_rate", f"{side}_rushing_plays_success_rate"), "difference",
                        "Passing versus rushing reliability gap.", "fraction"),
                Formula(f"f06_{side}_front_secondary_havoc_gap", (f"{side}_front_seven_havoc_rate", f"{side}_db_havoc_rate"), "difference",
                        "How disruptive events are distributed between the front and defensive backs.", "fraction"),
                Formula(f"f06_{side}_plays_per_drive", (f"{side}_plays", f"{side}_drives"), "ratio",
                        "Typical drive length in plays; complements rate efficiency with sustained possession.", "plays_per_drive"),
            ]
            for f in candidates:
                if set(f.inputs) <= set(features):
                    formulas[f.name] = f
        return list(formulas.values())
    # C collapses redundant temporal views without fitting outcome-dependent weights.
    for name in features:
        if name.startswith("opp_adj_") and name.endswith("_mean_to_date"):
            stem = name.removesuffix("_mean_to_date")
            inputs = (name, stem+"_last3", stem+"_ewm")
            if set(inputs) <= set(features):
                for item in inputs:
                    formulas.pop(item)
                f = Formula(stem+"_consensus", inputs, "mean",
                            "Equal-weight consensus of season, recent-three and exponentially weighted opponent-context state; balances established level and recent form.",
                            "same_as_three_source_states")
                formulas[f.name] = f
        if name.startswith("time_adj_last1__"):
            stem = name.split("__", 1)[1]
            inputs = tuple(prefix+stem for prefix in ("time_adj_last1__", "time_adj_last3__", "time_adj_ewm_hl3__"))
            if set(inputs) <= set(features):
                for item in inputs:
                    formulas.pop(item)
                f = Formula("time_adj_recent_consensus__"+stem, inputs, "mean",
                            "Consensus of the latest game, recent three games and exponentially weighted form; retains trend and volatility separately because they have different meanings.",
                            "same_as_three_source_states")
                formulas[f.name] = f
    return list(formulas.values())


def reciprocal_counterparts(names: list[str]) -> dict[str, str]:
    """Extend reviewed F6 pairings to named symmetric transforms.

    When no reciprocal matching measure exists, retain the inherited source as
    a same-measure home/away comparison; this does not invent a defense value.
    """
    available = set(names)
    result = {n: n for n in names}
    for n in names:
        cp = default_counterpart(n, available)
        if cp and default_counterpart(cp, available) == n:
            result[n] = cp
        for left, right in (("offense", "defense"), ("defense", "offense")):
            if left in n:
                candidate = n.replace(left, right, 1)
                if candidate in available:
                    result[n] = candidate
    if any(result[result[n]] != n for n in names):
        raise ValueError("Non-reciprocal matchup definition")
    return result


def feature_record(formula: Formula, generation: str, design: str, counterpart: str,
                   *, endpoints: list[str], kind="dynamic"):
    return {
        "name": formula.name, "generation": generation, "designs": [design],
        "raw_endpoints": endpoints, "raw_columns": list(formula.inputs),
        "source_inputs": list(formula.inputs), "equation_excel": formula.excel,
        "units": formula.units, "interpretation": formula.interpretation,
        "film_intuition": formula.interpretation,
        "direction": "context_only", "availability_rule": "week0 preseason freeze or completed prior source week plus 48-hour reporting lag; inherited source semantics audited separately",
        "temporal_cutoff": "before_target_game", "aggregation_window": "inherited canonical F6 temporal state",
        "minimum_sample_rule": "inherited canonical F6 support; no new rare-event split",
        "missingness_policy": "preserve missing; composites require all inputs; training-only median for M2 and native missing for M4",
        "static_or_dynamic": kind, "opponent_adjusted": any("opp_adj" in x for x in formula.inputs),
        "market_derived": False, "target_derived": False, "pregame_win_probability_derived": False,
        "garbage_time_handling": "inherited canonical F6; no new filtering",
        "matchup_counterpart": counterpart,
        "matchup_formula": f"home.[{formula.name}]{'-' if counterpart == formula.name else '+'}away.[{counterpart}]",
        "provenance": "standard/source-defined" if formula.operation == "identity" else "TDNet-derived",
        "source_inspiration": "Canonical TDNet F6; transparent temporal consensus or football interaction",
        "code_path": "src/gridiron_ml/experiments/nextgen_designs.py",
        "version": 1, "operation": formula.operation,
    }


def prepare_baseline_designs(root: Path):
    canonical_manifest = json.loads((ROOT / "docs/publication_2026/feature_manifests/F6.json").read_text())
    schema = json.loads((ROOT / "configs/experiments/nextgen_feature_manifest_schema_v1.json").read_text())
    source_path = root / "canonical/f06_aligned.parquet"
    provenance = json.loads(source_path.with_suffix(".provenance.json").read_text())
    if sha256_file(source_path) != provenance["data_sha256"]:
        raise ValueError("Aligned baseline hash mismatch")
    source = pd.read_parquet(source_path)
    from .nextgen_contract import assert_design_operation_frame
    assert_design_operation_frame(source, "equations")
    from .nextgen_source_policy import baseline_features, assert_excluded_absent
    config = json.loads((ROOT / 'configs/experiments/nextgen_fingerprints_v1.json').read_text())
    features = baseline_features(canonical_manifest["feature_names"], config)
    assert_excluded_absent(source)
    if provenance.get('feature_names') != features:
        raise ValueError('Baseline source feature policy mismatch')
    metadata = [c for c in source if c not in features]
    report = {}
    for design in "abc":
        formulas = baseline_formulas(features, design)
        names = [f.name for f in formulas]
        cp = reciprocal_counterparts(names)
        records = [feature_record(f, "F06", design, cp[f.name], endpoints=["canonical_f6_source"]) for f in formulas]
        validate_feature_manifest(records, schema)
        values = pd.DataFrame({f.name: f.evaluate(source) for f in formulas})
        values = values.replace([np.inf, -np.inf], np.nan)
        if design == "a":
            pd.testing.assert_frame_equal(values, source.loc[:, features], check_dtype=False)
        frame = pd.concat([source.loc[:, metadata], values], axis=1)
        dest = root / "fingerprints" / f"F06_F_{design}"
        dest.mkdir(parents=True, exist_ok=True)
        manifest = dest / "feature_manifest.json"
        manifest.write_text(json.dumps(records, indent=2)+"\n")
        data = dest / "values.parquet"
        frame.to_parquet(data, index=False, compression="zstd")
        atomic_json(dest / "provenance.json", {
            "fingerprint_id": f"F06_F_{design}", "source_sha256": provenance["data_sha256"],
            "schedule_sha256": provenance["schedule_sha256"], "data_sha256": sha256_file(data),
            "manifest_sha256": sha256_file(manifest), "max_design_year": int(frame.season.max()),
            "feature_count": len(names), "formula_selection": "outcome_independent_pre_screening",
            "source_semantics_audit_required_before_training": True,
            "baseline_source_policy": config['baseline_source_policy'],
        })
        report[design] = {"features": len(names), "rows": len(frame), "manifest": str(manifest)}
    atomic_json(root / "results/f06_design_materialization.json", report)
    return report


if __name__ == "__main__":
    cfg = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    print(json.dumps(prepare_baseline_designs(Path(cfg["artifact_root"])), indent=2))
