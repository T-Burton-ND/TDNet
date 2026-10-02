"""Annual average-team references: equal weeks, teams, then seasons.

These references contain no targets, model fits, or 2026 observations. Missing
values are omitted independently per feature at each averaging level; support
counts make that effective weighting explicit.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json


def annual_reference(frame, features, season):
    if not 2010 <= season <= 2026:
        raise ValueError('Reference season outside experiment scope')
    features = tuple(features)
    if not features or len(features) != len(set(features)):
        raise ValueError('Unique feature columns required')
    required = {'season', 'week', 'team', 'target_game_id', *features}
    if not required <= set(frame):
        raise ValueError('Reference requires keyed team-week features')
    if frame.season.gt(2025).any():
        raise ValueError('Quarantined 2026 source rows')
    if frame.duplicated(['target_game_id', 'team']).any():
        raise ValueError('Duplicate team-target state')
    prior = frame.loc[frame.season.lt(season)].copy()
    if prior.empty:
        raise ValueError('No completed prior-season states for reference')
    if prior[['season', 'week', 'team']].isna().any().any():
        raise ValueError('Missing reference grouping key')
    values = prior[list(features)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Nonfinite reference feature')
    # Two target games in one provider week must not double that week's weight.
    weeks = prior.groupby(['season', 'team', 'week'])[list(features)].mean()
    teams = weeks.groupby(level=['season', 'team']).mean()
    years = teams.groupby(level='season').mean()
    means = years.mean()
    return {'reference_season': int(season),
        'source_seasons': sorted(int(y) for y in years.index),
        'construction': 'mean target states within team-week; equal observed weeks within team-season; equal observed teams within season; equal observed seasons',
        'missingness': 'omit missing values separately at each averaging level; all-missing remains null',
        'values': {n: float(means[n]) if pd.notna(means[n]) else None for n in features},
        'support': {n: {'team_weeks': int(weeks[n].notna().sum()),
                         'team_seasons': int(teams[n].notna().sum()),
                         'seasons': int(years[n].notna().sum())} for n in features}}


def freeze_reference(path, payload):
    """A frozen annual reference cannot silently change as weekly data arrives."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('Frozen annual reference differs; explicit version required')
        return path
    atomic_json(path, payload)
    return path


def materialize_references(root, fingerprint):
    from .nextgen_contract import parse_fingerprint_id
    from .nextgen_artifacts import NextgenModelBoundary
    from .nextgen_assembly import assert_family_values
    from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file, load_authoritative_schedule
    parse_fingerprint_id(fingerprint)
    root = Path(root)
    repo = Path(__file__).resolve().parents[3]
    folder = root/'fingerprints'/fingerprint
    provenance = json.loads((folder/'provenance.json').read_text())
    for name, key in [('values.parquet','data_sha256'),('feature_manifest.json','manifest_sha256')]:
        if sha256_file(folder/name) != provenance[key]:
            raise ValueError('Fingerprint content hash mismatch')
    inventory = json.loads((repo/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule, digest = load_authoritative_schedule(root, inventory)
    if digest != provenance['schedule_sha256']:
        raise ValueError('Fingerprint schedule mismatch')
    frame = pd.read_parquet(folder/'values.parquet')
    features = [r['name'] for r in json.loads((folder/'feature_manifest.json').read_text())]
    for family in provenance.get('canonical_families', []):
        boundary = NextgenModelBoundary.from_canonical(root, family['generation'], family['family'], Path(family['manifest_path']))
        assert_family_values(frame, boundary.for_fit())
    # Recheck population and week labels against the fresh schedule authority.
    for game in frame[['target_game_id','season','week','team']].itertuples(index=False):
        match = schedule.loc[schedule.id.eq(game.target_game_id)]
        if len(match) != 1:
            raise ValueError('Unscheduled reference state')
        row = match.iloc[0]
        if (str(row.home_classification).lower() != 'fbs' or str(row.away_classification).lower() != 'fbs'
            or game.team not in (row.home_team,row.away_team) or game.season != row.season or game.week != row.week):
            raise ValueError('Reference state disagrees with schedule')
    result = []
    for season in range(int(frame.season.min())+1, 2027):
        payload = annual_reference(frame, features, season)
        payload.update(fingerprint_id=fingerprint, data_sha256=provenance['data_sha256'],
            manifest_sha256=provenance['manifest_sha256'], schedule_sha256=digest,
            source_semantics_audit_required_before_prediction=True)
        path = root/'average_team_references'/fingerprint/f'{season}.json'
        freeze_reference(path, payload)
        result.append(str(path))
    return {'fingerprint': fingerprint, 'reference_count': len(result), 'paths': result}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fingerprints', nargs='+')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    root = Path(json.loads((repo/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    for fingerprint in args.fingerprints:
        report = materialize_references(root, fingerprint)
        print(json.dumps(report), flush=True)
