"""Canonical F10 observed-experience designs, without roster-class inference."""
import json
from pathlib import Path
import pandas as pd
from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_designs import Formula, feature_record
from .nextgen_experience import observed_experience_state
from .nextgen_players import USAGE_METRICS
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file, load_authoritative_schedule
ROOT = Path(__file__).resolve().parents[3]


class ObservedExperienceBuilder(NextgenFeatureBuilder):
    generation = 'F10'

    def __init__(self, root, state, design):
        if design not in ('a', 'b', 'c'):
            raise ValueError('Unknown design')
        self.family = 'observed_experience_' + design
        self.state = state
        names = {m: m + '_weighted_observed_games' for m in USAGE_METRICS}
        self.formulas = [Formula(n, (n,), 'identity',
            'Prior workload-weighted acquired player-game record count; not class year or career games played.', 'observed games')
            for n in names.values()]
        if design == 'b':
            self.formulas.append(Formula('pass_rush_observed_experience_gap',
                (names['pass_attempts'], names['rush_attempts']), 'difference',
                'Difference in prior observed-record experience between passing and rushing users.', 'observed games'))
        elif design == 'c':
            groups = {'offensive_skill': ('pass_attempts', 'rush_attempts', 'receptions'),
                      'defensive': ('tackles',), 'special_teams': ('field_goal_attempts', 'punts', 'kick_returns', 'punt_returns')}
            self.formulas = [Formula(group + '_observed_experience', tuple(names[m] for m in metrics), 'mean',
                'Equal-weight observed experience across listed workloads; all component values required.', 'observed games')
                for group, metrics in groups.items()]
        self.feature_columns = tuple(f.name for f in self.formulas)
        records = []
        for f in self.formulas:
            r = feature_record(f, 'F10', design, f.name, endpoints=['/games/players', '/games'])
            r.update(raw_columns=['teams.categories.types.athletes.id', 'teams.categories.types.athletes.stat'],
                availability_rule='All regular-game records through latest team source kickoff plus 48h; strictly before target',
                aggregation_window='workload from last 12 available regular games; experience from all acquired records through latest team source cutoff',
                minimum_sample_rule='positive workload; finite nonnegative counts and stable IDs for every contributor; observed history required',
                garbage_time_handling='Regular-game boxes only; no play-level garbage adjustment available',
                source_inspiration='TDNet-derived observed history, independent of ambiguous roster year', provenance='TDNet-derived',
                code_path='src/gridiron_ml/experiments/nextgen_f10_experience.py',
                missingness_policy='Unknown identity or unsupported workload stays missing; left-truncated history is not career participation',
                input_equations={n:'=SUMPRODUCT(PriorWorkload,ObservedDistinctGameCounts)/SUM(PriorWorkload)' for n in f.inputs})
            if f.operation == 'identity':
                r['equation_excel'] = r['input_equations'][f.name]
            records.append(r)
        self.manifest_path = root / 'feature_families/F10' / self.family / 'feature_manifest.json'
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(records, indent=2) + '\n'
        if self.manifest_path.exists() and self.manifest_path.read_text() != content:
            raise ValueError('Version changed experience manifest')
        self.manifest_path.write_text(content)

    def build_frame(self):
        metadata = [c for c in self.state if not c.endswith('_weighted_observed_games')]
        return pd.concat([self.state[metadata].reset_index(drop=True),
            pd.DataFrame({f.name: f.evaluate(self.state) for f in self.formulas}).reset_index(drop=True)], axis=1)


def main():
    root = Path(json.loads((ROOT / 'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    preparation = json.loads((root / 'results/player_observation_preparation.json').read_text())
    inventory = json.loads((ROOT / 'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule, digest = load_authoritative_schedule(root, inventory)
    parts = []
    for item in preparation['partitions']:
        path = root / 'canonical/player_observations' / (item['request_id'] + '.parquet')
        if sha256_file(path) != item['data_sha256'] or item['schedule_sha256'] != digest:
            raise ValueError('Player observation provenance mismatch')
        if item['parser_sha256'] != sha256_file(ROOT / 'src/gridiron_ml/experiments/nextgen_players.py'):
            raise ValueError('Player parser changed; rerun preparation')
        parts.append(pd.read_parquet(path, columns=['game_id', 'team', 'athlete_id', 'metric', 'value']))
    state, coverage = observed_experience_state(pd.concat(parts, ignore_index=True), schedule)
    report = {'scope': 'observed experience only, not complete F10', 'schedule_sha256': digest,
              'coverage': coverage, 'designs': {}}
    for d in 'abc':
        b = ObservedExperienceBuilder(root, state, d)
        path = b.materialize(root)
        report['designs'][d] = {'path': str(path), 'rows': len(state), 'features': len(b.feature_columns)}
    atomic_json(root / 'results/f10_experience_materialization.json', report)
    print(json.dumps(report['designs'], indent=2))


if __name__ == '__main__':
    main()
