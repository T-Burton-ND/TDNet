"""Propose design-C floor trials; only subsequent measured screening accepts them."""
from collections import Counter

import numpy as np

from .nextgen_reduction import check_survivors, pair_groups


def joint_floor_proposal(records, consensus, *, shortfall_exceptions=None):
    """Maximize equal-architecture importance at each exact generation floor.

    The caller must obtain consensus from verified terminal full screening.
    A progressive pool may use a subset of that full reference's feature names.
    Scores suggest a trial, not evidence that the reduced model performs well.
    """
    names = [r['name'] for r in records]
    if not records or any('c' not in r.get('designs', []) for r in records):
        raise ValueError('Joint floor selection is only for design C')
    groups = pair_groups(records)
    check_survivors(names, records, shortfall_exceptions=shortfall_exceptions)
    table = consensus.set_index('source_feature')
    if table.index.duplicated().any() or not set(names) <= set(table.index):
        raise ValueError('Missing or duplicate source importance')
    values = table[['M2_importance', 'M4_importance']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid source importance')
    scores = table[['M2_importance', 'M4_importance']].mean(axis=1).to_dict()
    generations = {r['name']: r['generation'] for r in records}
    if any(len({generations[n] for n in group}) != 1 for group in groups):
        raise ValueError('Cross-generation reciprocal group requires a coupled selection policy')
    counts = Counter(generations.values())
    targets = {g: min(60 if g == 'F06' else 10, count) for g, count in counts.items()}
    selected = []
    for generation in sorted(targets):
        target = targets[generation]
        states = {0: (0., ())}
        for group in groups:
            if generations[group[0]] != generation:
                continue
            cost = len(group)
            value = sum(scores[n] for n in group)
            updated = dict(states)
            for size, (score, kept) in states.items():
                if size+cost > target:
                    continue
                candidate = (score+value, tuple(sorted((*kept, *group))))
                previous = updated.get(size+cost)
                if (previous is None or candidate[0] > previous[0]
                        or (candidate[0] == previous[0] and candidate[1] < previous[1])):
                    updated[size+cost] = candidate
            states = updated
        if target not in states:
            raise ValueError(f'Exact floor cannot preserve reciprocal groups for {generation}')
        selected.extend(states[target][1])
    selected = sorted(selected)
    actual = check_survivors(selected, records, shortfall_exceptions=shortfall_exceptions)
    if actual != targets:
        raise ValueError('Selected generation counts differ from floor targets')
    return {'survivors': selected, 'counts_by_generation': actual,
            'retained_joint_importance': sum(scores[n] for n in selected),
            'selection_basis': 'Exact per-generation floor; maximum summed equal-M2/M4 full-reference source importance with reciprocal groups intact.',
            'accepted': False}
