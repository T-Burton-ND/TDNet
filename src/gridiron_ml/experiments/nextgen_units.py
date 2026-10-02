"""Position-resolved unit production for F12, before temporal aggregation.

Roster membership and position are frozen before Week 0. Unmatched or ambiguous
positions remain unassigned; no player availability is inferred from usage.
"""
from __future__ import annotations

import pandas as pd

from .nextgen_players import stable_id

POSITION_UNIT = {
    'C': 'ol', 'G': 'ol', 'OT': 'ol', 'OG': 'ol', 'T': 'ol', 'OL': 'ol',
    'QB': 'qb', 'RB': 'rb', 'FB': 'rb', 'HB': 'rb',
    'WR': 'wrte', 'TE': 'wrte',
    'DT': 'front', 'DE': 'front', 'DL': 'front', 'NT': 'front',
    'LB': 'lb', 'ILB': 'lb', 'OLB': 'lb', 'MLB': 'lb',
    'CB': 'secondary', 'DB': 'secondary', 'S': 'secondary', 'FS': 'secondary', 'SS': 'secondary',
    'K': 'special_teams', 'PK': 'special_teams', 'P': 'special_teams', 'LS': 'special_teams',
}
SPECIAL_METRICS = {'field_goals', 'field_goal_attempts', 'extra_points', 'extra_point_attempts',
                   'punts', 'punt_yards', 'punt_touchbacks', 'punts_inside20',
                   'kick_returns', 'kick_return_yards', 'kick_return_td',
                   'punt_returns', 'punt_return_yards', 'punt_return_td'}


def resolve_unit_production(history: pd.DataFrame, frozen_rosters: pd.DataFrame, cutoffs: dict):
    """Join stable IDs within team/season; return additive unit-game totals.

    `cutoffs` is keyed by (season, team). Callers must provide source-backed
    roster availability evidence. This function never relabels the acquisition
    timestamp as a historical Week-0 snapshot.
    """
    required = {'season', 'team', 'athlete_id', 'position', 'available_utc', 'availability_evidence'}
    if not required <= set(frozen_rosters):
        raise ValueError('Missing frozen roster identity or availability fields')
    roster = frozen_rosters.copy()
    if roster.season.gt(2025).any() or history.season.gt(2025).any():
        raise ValueError('Quarantined unit source season')
    roster['athlete_id'] = roster.athlete_id.map(stable_id)
    for (year, team), rows in roster.groupby(['season', 'team']):
        cutoff = pd.Timestamp(cutoffs[(int(year), team)])
        available = pd.to_datetime(rows.available_utc, utc=True)
        if cutoff.tzinfo is None or available.isna().any() or not available.lt(cutoff).all():
            raise ValueError('Unit roster not available before Week 0')
        if not rows.availability_evidence.map(lambda x: isinstance(x,str) and bool(x.strip())).all():
            raise ValueError('Missing unit roster availability evidence')
    roster['unit'] = roster.position.astype(str).str.strip().str.upper().map(POSITION_UNIT)
    keys = ['season', 'team', 'athlete_id']
    identified = roster.loc[roster.athlete_id.notna()]
    # Even identical duplicate roster records require explicit upstream review.
    if identified.duplicated(keys).any():
        raise ValueError('Ambiguous roster player identity')
    source = history.copy()
    source['athlete_id'] = source.athlete_id.map(stable_id)
    joined = source.merge(identified[keys+['unit']], on=keys, how='left', validate='many_to_one')
    # Kicking/return production belongs to ST irrespective of the player's room.
    joined.loc[joined.metric.isin(SPECIAL_METRICS), 'unit'] = 'special_teams'
    audit = {'observation_rows': len(joined), 'assigned_rows': int(joined.unit.notna().sum()),
             'unassigned_rows': int(joined.unit.isna().sum()),
             'assigned_fraction': float(joined.unit.notna().mean()) if len(joined) else None}
    grouped = joined.dropna(subset=['unit']).groupby(['game_id', 'season', 'team', 'unit', 'metric'])
    # Unknown observations poison the total rather than disappearing in sum().
    totals = grouped.value.agg(lambda x: x.sum(min_count=len(x))).rename('value').reset_index()
    return totals, audit
