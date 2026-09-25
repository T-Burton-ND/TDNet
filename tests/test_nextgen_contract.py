"""Small, standard-library guards for setup-only next-generation contracts."""

import json
import importlib.util
import copy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/gridiron_ml/experiments/nextgen_contract.py"
SPEC = importlib.util.spec_from_file_location("nextgen_contract", MODULE_PATH)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)
all_fingerprint_ids = contract.all_fingerprint_ids
assert_average_reference_years = contract.assert_average_reference_years
assert_design_years = contract.assert_design_years
assert_pair_closed = contract.assert_pair_closed
assert_safe_inputs = contract.assert_safe_inputs
parent_of = contract.parent_of
parse_fingerprint_id = contract.parse_fingerprint_id
validate_feature_manifest = contract.validate_feature_manifest
validate_setup = contract.validate_setup
validate_contract = contract.validate_contract


class NextgenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT / "configs/experiments"
        cls.schema = json.loads((base / "nextgen_feature_manifest_schema_v1.json").read_text())
        cls.records = json.loads((base / "nextgen_seed_features_v1.json").read_text())

    def test_setup_and_lineage(self):
        validate_setup(ROOT)
        self.assertEqual(len(all_fingerprint_ids()), 42)
        self.assertEqual(parent_of("F09_F_b"), "F06_F_b")
        self.assertEqual(parent_of("F09_PR_b"), "F06_R_b")
        self.assertEqual(parent_of("F10_PR_b"), "F09_PR_b")
        self.assertEqual(parent_of("F10_LR_b"), "F10_F_b")
        self.assertTrue(all("F07" not in name and "F08" not in name for name in all_fingerprint_ids()))
        with self.assertRaises(ValueError):
            parse_fingerprint_id("F06_PR_a")
        with self.assertRaises(ValueError):
            parse_fingerprint_id("F10_R_a")

    def test_2026_and_average_reference_quarantine(self):
        assert_design_years([2010, 2024, 2025])
        with self.assertRaises(ValueError):
            assert_design_years([2025, 2026])
        assert_average_reference_years(2026, [2010, 2024, 2025])
        with self.assertRaises(ValueError):
            assert_average_reference_years(2026, [2025, 2026])

    def test_market_and_probability_exclusion(self):
        assert_safe_inputs(["player_stats_total", "offense_win_rate_from_prior_games"])
        for name in ("market_spread_close", "pregame_wp", "win_probability", "vegas_total"):
            with self.assertRaises(ValueError):
                assert_safe_inputs([name])

    def test_manifest_pair_and_c_cap(self):
        validate_feature_manifest(self.records, self.schema)
        with self.assertRaises(ValueError):
            assert_pair_closed([self.records[0]["name"]], self.records)
        assert_pair_closed([record["name"] for record in self.records], self.records)
        bad = [dict(record) for record in self.records]
        bad[0]["source_inputs"] = [f"source_{i}" for i in range(6)]
        with self.assertRaises(ValueError):
            validate_feature_manifest(bad, self.schema)
        bad = [dict(record) for record in self.records]
        bad[0]["temporal_cutoff"] = "after_target_game"
        with self.assertRaises(ValueError):
            validate_feature_manifest(bad, self.schema)
        bad = [dict(record) for record in self.records]
        bad[1]["name"] = bad[0]["name"]
        with self.assertRaises(ValueError):
            validate_feature_manifest(bad, self.schema)

    def test_storage_and_static_availability(self):
        config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
        bad_config = copy.deepcopy(config)
        bad_config["artifact_root"] = "/users/example/TDNet/large_artifacts"
        with self.assertRaises(ValueError):
            validate_contract(bad_config, repo_root=ROOT)
        static = copy.deepcopy(self.records)
        static[0]["static_or_dynamic"] = "static_preseason"
        static[0]["availability_rule"] = "available_after_game_final"
        with self.assertRaises(ValueError):
            validate_feature_manifest(static, self.schema)


if __name__ == "__main__":
    unittest.main()
