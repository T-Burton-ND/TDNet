"""Regular-season coaching history reconstructed without provider season ratings.

Coach IDs are join keys only. Never expose identity as a learned predictor or
attribute a split-coach season to one coach without game-level assignments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regular_coach_seasons(coaches: pd.DataFrame, schedule: pd.DataFrame):
    """Use only unambiguous completed head-coach/team seasons and fresh scores."""
    if schedule.season.gt(2025).any() or coaches.year.gt(2025).any():
        raise ValueError('Quarantined coaching source season')
    if schedule.id.duplicated().any():
        raise ValueError('Duplicate schedule identity')
    games = schedule.loc[schedule.season_type.eq('regular') & schedule.completed.eq(True)].copy()
    games['kickoff'] = pd.to_datetime(games.start_date, utc=True)
    if games.kickoff.isna().any():
        raise ValueError('Missing regular-game kickoff')
    identities = coaches[['year', 'team.school', 'coach.id', 'attribution_complete']].copy()
    if identities.duplicated(['year', 'team.school', 'coach.id']).any():
        raise ValueError('Duplicate coach/team-season observation')
    rows, excluded = [], []
    for (year, team), group in identities.groupby(['year', 'team.school']):
        if len(group) != 1 or group['coach.id'].isna().any() or not group.attribution_complete.eq(True).all():
            excluded.append({'season': int(year), 'team': team, 'reason': 'ambiguous_or_incomplete_coach_attribution'})
            continue
        subset = games.loc[games.season.eq(year) & (games.home_team.eq(team) | games.away_team.eq(team))]
        if subset.empty:
            continue
        home = subset.home_team.eq(team)
        points_for = pd.to_numeric(subset.home_points.where(home, subset.away_points), errors='coerce')
        points_against = pd.to_numeric(subset.away_points.where(home, subset.home_points), errors='coerce')
        if points_for.isna().any() or points_against.isna().any():
            excluded.append({'season': int(year), 'team': team, 'reason': 'missing_regular_game_score'})
            continue
        margin = points_for-points_against
        rows.append({'coach_id': group.iloc[0]['coach.id'], 'team': team, 'season': int(year),
                     'games': len(subset), 'wins': int(margin.gt(0).sum()), 'ties': int(margin.eq(0).sum()),
                     'points_for': float(points_for.sum()), 'points_against': float(points_against.sum()),
                     'close_games': int(margin.abs().le(8).sum()),
                     'close_wins': int((margin.gt(0) & margin.le(8)).sum()),
                     'latest_source_game_utc': subset.kickoff.max(),
                     'available': subset.kickoff.max()+pd.Timedelta(hours=48)})
    return pd.DataFrame(rows), {'excluded_team_seasons': excluded}


def coach_history_features(seasons: pd.DataFrame, coach_id, *, target_season: int, cutoff):
    """Aggregate history for an externally verified pre-target coach assignment.

    History scope starts at the acquired schedule's first season; quantities are
    observed-history totals, not claims of complete lifetime career coverage.
    """
    when = pd.Timestamp(cutoff)
    if when.tzinfo is None or target_season > 2025:
        raise ValueError('Timezone-aware pre-2026 coaching cutoff required')
    prior = seasons.loc[seasons.coach_id.eq(coach_id) & seasons.season.lt(target_season) & seasons.available.lt(when)].sort_values('season')
    names = ('regular_win_fraction', 'regular_points_for_per_game', 'regular_points_against_per_game',
             'regular_close_win_fraction', 'observed_regular_games', 'prior_program_count',
             'regular_margin_trend')
    if prior.empty:
        return {name: np.nan for name in names}
    games = prior.games.sum()
    close = prior.close_games.sum()
    margins = (prior.points_for-prior.points_against)/prior.games
    years = prior.season.to_numpy(float)
    trend = float(np.polyfit(years-years.min(), margins, 1)[0]) if len(np.unique(years)) >= 3 else np.nan
    return {'regular_win_fraction': float((prior.wins.sum()+.5*prior.ties.sum())/games),
            'regular_points_for_per_game': float(prior.points_for.sum()/games),
            'regular_points_against_per_game': float(prior.points_against.sum()/games),
            'regular_close_win_fraction': float(prior.close_wins.sum()/close) if close else np.nan,
            'observed_regular_games': int(games), 'prior_program_count': int(prior.team.nunique()),
            'regular_margin_trend': trend}
