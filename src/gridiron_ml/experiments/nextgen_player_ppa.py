"""Bind observed player-game PPA/positions without inventing roster membership."""
import numpy as np
import pandas as pd
from .nextgen_units import POSITION_UNIT
from .nextgen_players import stable_id


def bind_player_ppa(records, schedule):
    """Resolve provider season/week/team/opponent keys only when unique.

    PPA rows lack game IDs. Ambiguous doubleheaders and unmatched opponents
    remain excluded with an explicit audit; names and descriptions never repair
    their identities. Positions describe the historical source record only.
    """
    if (schedule.season.gt(2025).any() or records.season.gt(2025).any()
            or not schedule.season_type.eq('regular').all()
            or not schedule.completed.eq(True).all()
            or not records.season_type.eq('regular').all()):
        raise ValueError('Completed pre-2026 regular sources required')
    if schedule.id.duplicated().any():
        raise ValueError('Duplicate schedule identity')
    keys=['season','week','team','opponent']
    mapping=[]
    for g in schedule.itertuples(index=False):
        for team,opponent in [(g.home_team,g.away_team),(g.away_team,g.home_team)]:
            mapping.append(dict(season=g.season,week=g.week,team=team,opponent=opponent,
                                game_id=g.id,kickoff=g.start_date))
    mapping=pd.DataFrame(mapping)
    ambiguous=mapping.duplicated(keys,keep=False)
    unique=mapping.loc[~ambiguous]
    source=records.drop_duplicates().copy()
    exact_duplicate_rows=len(records)-len(source)
    source['athlete_id']=source.id.map(stable_id)
    joined=source.merge(unique,on=keys,how='left',validate='many_to_one')
    missing=joined.game_id.isna()
    audit={'source_rows':len(records),'exact_duplicate_rows_removed':exact_duplicate_rows,'unmatched_or_ambiguous_rows':int(missing.sum()),
           'unmatched_keys':joined.loc[missing,keys].drop_duplicates().to_dict(orient='records'),
           'missing_player_identity_rows':int(joined.athlete_id.isna().sum())}
    joined=joined.loc[~missing & joined.athlete_id.notna()].copy()
    joined['game_id']=joined.game_id.astype('int64')
    identity=['game_id','team','athlete_id']
    conflict=joined.duplicated(identity,keep=False)
    audit['conflicting_player_game_rows_excluded']=int(conflict.sum())
    audit['conflicting_player_game_keys']=joined.loc[conflict,identity].drop_duplicates().to_dict(orient='records')
    # Never choose between contradictory values, positions, or identities.
    joined=joined.loc[~conflict].copy()
    joined['unit']=joined.position.astype(str).str.strip().str.upper().map(POSITION_UNIT)
    joined['kickoff']=pd.to_datetime(joined.kickoff,utc=True)
    joined['available']=joined.kickoff+pd.Timedelta(hours=48)
    if joined.kickoff.isna().any():
        raise ValueError('Missing source kickoff')
    for column in ('average_p_p_a.all','average_p_p_a.pass','average_p_p_a.rush'):
        joined[column]=pd.to_numeric(joined[column],errors='coerce').replace([np.inf,-np.inf],np.nan)
    audit['matched_rows']=len(joined)
    audit['position_rows']=joined.position.value_counts(dropna=False).to_dict()
    audit['unassigned_unit_rows']=int(joined.unit.isna().sum())
    return joined,audit


def main():
    import json
    from pathlib import Path
    from .nextgen_f09 import verified_endpoint_records
    from gridiron_ml.pipeline.fetch.nextgen_acquisition import load_authoritative_schedule,atomic_json,sha256_file
    repo=Path(__file__).resolve().parents[3]
    root=Path(json.loads((repo/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    inventory=json.loads((repo/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule,digest=load_authoritative_schedule(root,inventory)
    sources=verified_endpoint_records(root,'/ppa/players/games')
    if not sources:
        raise ValueError('No verified player PPA records')
    raw=pd.concat([pd.read_parquet(r['cache_path']) for r in sources],ignore_index=True)
    joined,audit=bind_player_ppa(raw,schedule)
    path=root/'canonical/player_ppa_games.parquet'
    temporary=path.with_suffix('.tmp.parquet')
    joined.to_parquet(temporary,index=False,compression='zstd')
    temporary.replace(path)
    report={'scope':'source observations only, not a feature family','schedule_sha256':digest,
            'data_sha256':sha256_file(path),'code_sha256':sha256_file(Path(__file__)),
            'source_sha256':{r['request_id']:r['sha256'] for r in sources},'audit':audit}
    atomic_json(path.with_suffix('.provenance.json'),report)
    atomic_json(root/'results/player_ppa_game_binding.json',report)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
