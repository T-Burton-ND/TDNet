"""Explicit user-authorized baseline exclusions, shared by build and load."""
EXCLUDED_COACH_SP = frozenset({
    'coach_career_mean_sp_offense', 'coach_career_mean_sp_defense',
})


def baseline_features(features, config):
    policy = config.get('baseline_source_policy', {})
    if (set(policy.get('excluded_features', [])) != EXCLUDED_COACH_SP
            or policy.get('exact_preservation_required') is not False
            or not policy.get('authorization')):
        raise ValueError('Explicit baseline exclusion policy required')
    if not EXCLUDED_COACH_SP <= set(features):
        raise ValueError('Canonical source lacks expected excluded fields')
    return [name for name in features if name not in EXCLUDED_COACH_SP]


def assert_excluded_absent(frame, records=()):
    names = set(frame.columns)
    for record in records:
        names.add(record['name'])
        names.update(record.get('source_inputs', []))
        names.update(record.get('raw_columns', []))
    if names & EXCLUDED_COACH_SP:
        raise ValueError('Excluded coach SP fields remain in predictive artifact')
