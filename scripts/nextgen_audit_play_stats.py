#!/usr/bin/env python3
"""Audit bounded attributed-event samples against verified play partitions."""
import json
from pathlib import Path
import pandas as pd
from gridiron_ml.experiments.nextgen_microstructure import RUSH_TYPES
from gridiron_ml.pipeline.fetch.nextgen_acquisition import verify_cache,atomic_json
ROOT=Path(__file__).resolve().parents[1]


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    entries=[json.loads(p.read_text()) for p in (root/'request_ledger').glob('*/*.json')]
    samples=[r for r in entries if r['endpoint']=='/plays/stats' and r['status']=='success_complete']
    reports=[]
    for r in samples:
        if verify_cache(Path(r['cache_path']),r) is None:
            raise ValueError('Unverified sample cache')
        stats=pd.read_parquet(r['cache_path'])
        year=int(stats.season.iloc[0]);week=int(stats.week.iloc[0]);game=int(stats.game_id.iloc[0])
        sources=[p for p in entries if p['endpoint']=='/plays' and p.get('year')==year and p.get('week')==week and p['status']=='success_complete']
        if len(sources)!=1:
            raise ValueError('Unique verified play partition required')
        p=sources[0]
        if verify_cache(Path(p['cache_path']),p) is None:
            raise ValueError('Unverified play cache')
        plays=pd.read_parquet(p['cache_path'])
        plays=plays.loc[plays.game_id.eq(game)].copy()
        plays['play_key']=plays.id.astype(str)
        stats['play_key']=stats.play_id.astype(str)
        joined=stats.merge(plays[['play_key','offense','defense','play_type','yards_gained','period']],on='play_key',how='left',validate='many_to_one',suffixes=('_stat','_play'))
        rush=joined.loc[joined.stat_type.eq('Rush')]
        expected=set(plays.loc[plays.play_type.isin(RUSH_TYPES),'play_key'])
        observed=set(rush.play_key)
        game_report={'game_id':game,'season':year,'week':week,'sample_sha256':r['sha256'],'plays_sha256':p['sha256'],
            'rows':len(stats),'row_cap':2000,'at_cap':len(stats)>=2000,
            'unique_attributed_plays':int(stats.play_key.nunique()),'unmatched_attributed_rows':int(joined.play_type.isna().sum()),
            'missing_athlete_ids':int(stats.athlete_id.isna().sum()),
            'stat_types':stats.stat_type.value_counts().to_dict(),
            'stat_ranges':stats.groupby('stat_type').stat.agg(['min','max']).to_dict(orient='index'),
            'expected_rush_plays':len(expected),'attributed_rush_plays':len(observed),
            'rush_play_recall':len(expected&observed)/len(expected) if expected else None,
            'extra_attributed_rush_plays':len(observed-expected),
            'rush_offense_team_mismatches':int(rush.team.ne(rush.offense).sum()),
            'rush_yardage_mismatches':int(pd.to_numeric(rush.stat).ne(pd.to_numeric(rush.yards_gained)).sum()),
            'rush_period_mismatches':int(rush.period_stat.ne(rush.period_play).sum()),
            'duplicate_rush_attributions':int(rush.duplicated(['play_key','athlete_id']).sum())}
        reports.append(game_report)
    report={'scope':'bounded sample; not proof of universal historical coverage','samples':reports,
            'participation_inference_allowed':False,'position_field_available':False,
            'cap_policy':'responses at 2000 are suspected partial and halt acquisition pending subdivision/review'}
    atomic_json(root/'results/preflight/plays_stats_sample_audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
