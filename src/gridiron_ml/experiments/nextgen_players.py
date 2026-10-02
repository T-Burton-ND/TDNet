"""Structured player box-score primitives for F10/F12 prior production.

These are source observations, not canonical model features. Callers must apply
fresh regular-season schedule membership and pre-target availability before use.
"""
from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd

# Only additive observations. Rates, QBR and longest plays are not summed.
SCALARS = {
    'rushing': {'CAR': 'rush_attempts', 'YDS': 'rush_yards', 'TD': 'rush_td'},
    'passing': {'YDS': 'pass_yards', 'TD': 'pass_td', 'INT': 'pass_interceptions'},
    'receiving': {'REC': 'receptions', 'YDS': 'receive_yards', 'TD': 'receive_td'},
    'defensive': {'TOT': 'tackles', 'SOLO': 'solo_tackles', 'SACKS': 'sacks', 'TFL': 'tfl'},
    'interceptions': {'INT': 'defensive_interceptions', 'YDS': 'interception_yards'},
    'punting': {'NO': 'punts', 'YDS': 'punt_yards', 'TB': 'punt_touchbacks', 'In 20': 'punts_inside20'},
    'kickReturns': {'NO': 'kick_returns', 'YDS': 'kick_return_yards', 'TD': 'kick_return_td'},
    'puntReturns': {'NO': 'punt_returns', 'YDS': 'punt_return_yards', 'TD': 'punt_return_td'},
}
PAIRS = {('passing', 'C/ATT'): ('pass_completions', 'pass_attempts'),
         ('kicking', 'FG'): ('field_goals', 'field_goal_attempts'),
         ('kicking', 'XP'): ('extra_points', 'extra_point_attempts')}


def normalize_identity(value):
    if value is None or pd.isna(value):
        return ''
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower())


def stable_id(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return None if value.lower() in {'', 'nan', 'none', 'null'} else value


def flatten_player_boxes(frame: pd.DataFrame):
    """Keep explicit numeric athlete observations, including fractional tackles.

    Missing/invalid values remain missing and get counted in the audit. Never
    infer zero production from absence, nor injuries from nonparticipation.
    """
    observations, audit = [], {'unknown_stat_rows': 0, 'invalid_values': 0, 'missing_ids': 0}
    for game in frame.itertuples(index=False):
        for team in game.teams:
            for category in team.get('categories', []):
                cat = category['name']
                for stat in category.get('types', []):
                    key = (cat, stat['name'])
                    names = PAIRS.get(key)
                    if names is None:
                        name = SCALARS.get(cat, {}).get(stat['name'])
                        names = (name,) if name else ()
                    if not names:
                        audit['unknown_stat_rows'] += len(stat.get('athletes', []))
                        continue
                    for player in stat.get('athletes', []):
                        pid = stable_id(player.get('id'))
                        if pid is None:
                            audit['missing_ids'] += 1
                        raw = str(player.get('stat', '')).strip()
                        parts = raw.split('/') if len(names) == 2 else [raw]
                        values = pd.to_numeric(pd.Series(parts), errors='coerce').to_numpy(float)
                        if len(values) != len(names) or not np.isfinite(values).all():
                            audit['invalid_values'] += 1
                            values = np.full(len(names), np.nan)
                        for name, value in zip(names, values):
                            observations.append({'game_id': int(game.id), 'team': team['team'],
                                                 'athlete_id': pid, 'athlete_name': player.get('name'),
                                                 'metric': name, 'value': value})
    out = pd.DataFrame(observations, columns=['game_id', 'team', 'athlete_id', 'athlete_name', 'metric', 'value'])
    # Duplicate identities cannot silently double a player's production.
    if out.duplicated(['game_id', 'team', 'athlete_id', 'athlete_name', 'metric']).any():
        raise ValueError('Duplicate player-game metric observations')
    return out, audit


def match_players(players: pd.DataFrame, roster: pd.DataFrame):
    """Match within an already selected team/season/position cohort.

    Explicit contradictory IDs are never overridden by a name match. Exact
    names are fallback only when one side has no ID; ambiguity stays unmatched.
    Cross-team transfer identity must be resolved by stable ID by the caller.
    """
    required = {'athlete_id', 'name', 'team', 'season', 'position'}
    if not required <= set(players) or not required <= set(roster):
        raise ValueError('Player matching needs identity and team/season/position constraints')
    roster = roster.reset_index(drop=True).copy()
    roster['_id'] = roster.athlete_id.map(stable_id)
    roster['_name'] = roster.name.map(normalize_identity)
    results = []
    for row in players.itertuples(index=False):
        candidates = roster.loc[roster.team.eq(row.team) & roster.season.eq(row.season) & roster.position.eq(row.position)]
        pid = stable_id(row.athlete_id)
        matches = candidates.loc[candidates["_id"].eq(pid)] if pid is not None else candidates.iloc[:0]
        method = 'stable_id'
        if len(matches) == 0:
            method = 'exact_name'
            name = normalize_identity(row.name)
            matches = candidates.loc[candidates["_name"].eq(name)] if name else candidates.iloc[:0]
            if pid is not None:
                matches = matches.loc[matches["_id"].isna()]
        if len(matches) != 1:
            results.append({'roster_row': None, 'match_method': 'unmatched'})
        else:
            results.append({'roster_row': int(matches.index[0]), 'match_method': method})
    return pd.DataFrame(results, index=players.index, columns=['roster_row', 'match_method'])


USAGE_METRICS = ('pass_attempts', 'rush_attempts', 'receptions', 'tackles',
                 'field_goal_attempts', 'punts', 'kick_returns', 'punt_returns')


def checked_player_history(observations: pd.DataFrame, schedule: pd.DataFrame, *, lag_hours=48):
    """Bind observations to completed regular games and reconstructed availability."""
    if lag_hours < 0:
        raise ValueError('Reporting lag cannot be negative')
    if schedule.id.duplicated().any() or schedule.season.gt(2025).any():
        raise ValueError('Nonunique or quarantined schedule')
    if not schedule.season_type.eq('regular').all() or not schedule.completed.eq(True).all():
        raise ValueError('Player history requires completed regular-season authority')
    source = observations.merge(schedule[['id', 'season', 'start_date', 'home_team', 'away_team']],
                                left_on='game_id', right_on='id', how='left', validate='many_to_one')
    if source.id.isna().any():
        raise ValueError('Player observations outside regular-season schedule')
    if not (source.team.eq(source.home_team) | source.team.eq(source.away_team)).all():
        raise ValueError('Player team is not a scheduled participant')
    source['kickoff'] = pd.to_datetime(source.start_date, utc=True)
    if source.kickoff.isna().any():
        raise ValueError('Missing source kickoff')
    source['available'] = source.kickoff + pd.Timedelta(hours=lag_hours)
    source['athlete_id'] = source.athlete_id.map(stable_id)
    if source.loc[source.athlete_id.notna()].duplicated(['game_id', 'team', 'athlete_id', 'metric']).any():
        raise ValueError('Duplicate stable player-game metric')
    return source


def usage_concentration(prior: pd.DataFrame):
    """Shares describe observed usage, never assumed depth-chart availability.

    A missing identity or invalid count makes concentration unknown for that
    metric; aggregating anonymous athletes into one synthetic player is invalid.
    """
    result = {}
    for metric in USAGE_METRICS:
        part = prior.loc[prior.metric.eq(metric)]
        total = part.value.sum(min_count=1)
        valid = (not part.empty and part.athlete_id.notna().all()
                 and part.value.notna().all() and np.isfinite(part.value).all()
                 and part.value.ge(0).all() and total > 0)
        shares = part.groupby('athlete_id').value.sum()/total if valid else None
        result[metric+'_top_share'] = float(shares.max()) if valid else np.nan
        result[metric+'_hhi'] = float((shares**2).sum()) if valid else np.nan
    return result


def player_usage_state(observations: pd.DataFrame, schedule: pd.DataFrame, *, window=12, lag_hours=48):
    """Build paired pre-target usage states for subsequent canonical builders.

    Last available regular games, across seasons; no future roster is consulted.
    This is usage concentration, not a claim that those athletes remain active.
    """
    if window < 1:
        raise ValueError('Positive history window required')
    history = checked_player_history(observations, schedule, lag_hours=lag_hours)
    teams = {team: frame for team, frame in history.groupby('team')}
    targets = schedule.loc[schedule.home_classification.str.lower().eq('fbs') & schedule.away_classification.str.lower().eq('fbs')]
    rows, excluded = [], []
    for game in targets.itertuples(index=False):
        target = pd.Timestamp(game.start_date)
        if target.tzinfo is None:
            raise ValueError('Timezone-aware target required')
        pair = []
        for team in (game.home_team, game.away_team):
            h = teams.get(team)
            if h is None:
                break
            prior = h.loc[h.available.lt(target)]
            games = prior[['game_id', 'kickoff', 'available']].drop_duplicates().sort_values(['available', 'game_id']).tail(window)
            if games.empty:
                break
            prior = prior.loc[prior.game_id.isin(games.game_id)]
            latest = games.iloc[-1]
            values = usage_concentration(prior)
            values.update(season=int(game.season), season_type='regular', team=team,
                          target_game_id=int(game.id), target_start_utc=target,
                          feature_kind='dynamic', latest_source_game_id=int(latest.game_id),
                          latest_source_game_utc=latest.kickoff, latest_source_season_type='regular',
                          feature_available_utc=latest.available, static_availability_documentation=None,
                          source_game_count=len(games))
            pair.append(values)
        if len(pair) == 2:
            rows.extend(pair)
        else:
            excluded.append(int(game.id))
    return pd.DataFrame(rows), {'excluded_no_prior_player_coverage': excluded,
                                'window_games': window, 'reporting_lag_hours': lag_hours}


def returning_production(history: pd.DataFrame, frozen_roster: pd.DataFrame, *,
                         team: str, season: int, week0_cutoff):
    """Summarize prior-season observed production using verified frozen membership.

    `history` comes from checked_player_history. Roster availability must be
    supplied from documented source semantics; a season label alone is not
    evidence of a preseason snapshot. No unobserved player gets zero production.
    """
    cutoff = pd.Timestamp(week0_cutoff)
    if cutoff.tzinfo is None or season > 2025:
        raise ValueError('Pre-2026 season and timezone-aware Week-0 cutoff required')
    required = {'athlete_id', 'team', 'season', 'available_utc', 'availability_evidence'}
    if not required <= set(frozen_roster):
        raise ValueError('Frozen roster needs membership and availability evidence')
    roster = frozen_roster.loc[frozen_roster.team.eq(team) & frozen_roster.season.eq(season)].copy()
    if roster.empty:
        raise ValueError('Missing frozen team-season roster')
    available = pd.to_datetime(roster.available_utc, utc=True)
    if available.isna().any() or not available.lt(cutoff).all():
        raise ValueError('Roster was not available before Week 0')
    if not roster.availability_evidence.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Missing roster availability evidence')
    roster['athlete_id'] = roster.athlete_id.map(stable_id)
    ids = set(roster.athlete_id.dropna())
    if roster.athlete_id.dropna().duplicated().any():
        raise ValueError('Duplicate frozen roster identity')
    prior = history.loc[history.season.eq(season-1) & history.available.lt(cutoff)].copy()
    prior['athlete_id'] = prior.athlete_id.map(stable_id)
    home = prior.loc[prior.team.eq(team)]
    joined = prior.loc[prior.athlete_id.isin(ids)]
    matched_ids = set(joined.athlete_id.dropna())
    report = {'roster_rows': len(roster), 'roster_identified_count': len(ids),
              'roster_with_observed_prior_production': len(matched_ids),
              'prior_production_match_fraction': len(matched_ids)/len(roster),
              'unmatched_does_not_imply_zero_production': True}
    result = {}
    for metric in USAGE_METRICS:
        part = home.loc[home.metric.eq(metric)]
        # Unknown identities or malformed counts prevent claiming an exact share.
        valid = (not part.empty and part.athlete_id.notna().all() and part.value.notna().all()
                 and np.isfinite(part.value).all() and part.value.ge(0).all() and part.value.sum()>0)
        retained = part.loc[part.athlete_id.isin(ids)]
        result[metric+'_returning_share'] = float(retained.value.sum()/part.value.sum()) if valid else np.nan
        newcomers = joined.loc[joined.metric.eq(metric) & ~joined.team.eq(team)]
        # Only observed transfer production; a missing record is not zero.
        valid_incoming = (not newcomers.empty and newcomers.value.notna().all()
                          and np.isfinite(newcomers.value).all() and newcomers.value.ge(0).all())
        result[metric+'_incoming_prior_total'] = float(newcomers.value.sum()) if valid_incoming else np.nan
    return result, report
