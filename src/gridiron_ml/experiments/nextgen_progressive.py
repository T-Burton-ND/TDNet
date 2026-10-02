"""Prepare progressive candidate pools without resurrecting pruned features.

This pure assembly step does not authorize a parent, train, prune, or accept a
fingerprint. Callers must verify the previous generation's finalization before
persisting or screening its output.
"""
import pandas as pd

from .nextgen_contract import assert_pair_closed, parent_of, parse_fingerprint_id


def progressive_pool(full_id, parent_id, full, full_records, parent, parent_records):
    """Project full data onto prior survivors plus all new-generation features.

    Shared rows must preserve inherited values and metadata exactly. Restrict
    the cohort to complete games available in both sources and report losses.
    """
    generation, variant, design = parse_fingerprint_id(full_id)
    child_id = f'{generation}_PR_{design}'
    if generation == 'F06' or variant != 'F' or parent_of(child_id) != parent_id:
        raise ValueError('Progressive ancestry must use the previous same-design reduction')
    full_map = {r['name']: r for r in full_records}
    parent_map = {r['name']: r for r in parent_records}
    if (len(full_map) != len(full_records) or len(parent_map) != len(parent_records)
            or not parent_map):
        raise ValueError('Unique nonempty feature manifests required')
    if any(full_map.get(n) != r or r['generation'] == generation for n, r in parent_map.items()):
        raise ValueError('Inherited manifest differs from accumulated full representation')
    new = [r for r in full_records if r['generation'] == generation]
    if not new:
        raise ValueError('Full new-generation family required')
    retained = [r for r in full_records if r['name'] in parent_map or r['generation'] == generation]
    names = [r['name'] for r in retained]
    assert_pair_closed(names, full_records)
    keys = ['target_game_id', 'team']
    for frame, records in ((full, full_map), (parent, parent_map)):
        if (not set(keys).union(records) <= set(frame) or frame[keys].isna().any().any()
                or frame.duplicated(keys).any()
                or not frame.groupby('target_game_id').size().eq(2).all()):
            raise ValueError('Unique paired game identities and declared features required')
    metadata = [c for c in full if c not in full_map]
    if not set(metadata) <= set(parent):
        raise ValueError('Parent lacks inherited metadata')
    keyed = full.set_index(keys)
    prior = parent.set_index(keys)
    shared = keyed.loc[keyed.index.isin(prior.index)].reset_index()
    paired = shared.groupby('target_game_id').size()
    shared = shared.loc[shared.target_game_id.isin(paired.index[paired.eq(2)])]
    if shared.empty:
        raise ValueError('No shared paired target games')
    compare = [c for c in metadata if c not in keys] + list(parent_map)
    left = shared.set_index(keys)[compare]
    right = prior.reindex(left.index)[compare]
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Inherited values or target metadata changed') from exc
    return shared[metadata + names].copy(), retained, {
        'fingerprint_id': child_id,
        'ancestry': [parent_id],
        'full_reference': full_id,
        'inherited_features': list(parent_map),
        'new_features': [r['name'] for r in new],
        'excluded_full_game_ids': sorted(set(full.target_game_id) - set(shared.target_game_id)),
        'accepted': False,
    }
