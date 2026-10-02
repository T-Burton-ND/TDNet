#!/usr/bin/env python3
"""Compact measured coverage audit of existing canonical families only."""
import json
from pathlib import Path
import pandas as pd
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json,sha256_file
ROOT=Path(__file__).resolve().parents[1]


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    output=[]
    for path in sorted((root/'feature_families').glob('F*/*/canonical.parquet')):
        provenance=json.loads(path.with_suffix('.provenance.json').read_text())
        manifest=path.parent/'feature_manifest.json'
        if sha256_file(path)!=provenance['data_sha256'] or sha256_file(manifest)!=provenance['manifest_sha256']:
            raise ValueError('Canonical content changed: '+str(path))
        features=provenance['feature_columns']
        frame=pd.read_parquet(path,columns=['season','team','target_game_id',*features])
        if frame.season.isna().any() or frame.season.gt(2025).any():
            raise ValueError('Missing or quarantined coverage year')
        years=[]
        for year in range(2010,2026):
            part=frame.loc[frame.season.eq(year)]
            observed=part[features].notna()
            counts=observed.sum()
            years.append({'season':year,'team_target_rows':len(part),'target_games':int(part.target_game_id.nunique()),
                'teams':int(part.team.nunique()),'observed_feature_cells':int(counts.sum()),
                'total_feature_cells':int(len(part)*len(features)),
                'missing_fraction':float(1-observed.to_numpy().mean()) if len(part) else None,
                'features_with_any_observation':int(counts.gt(0).sum()),
                'all_missing_features':counts.index[counts.eq(0)].tolist(),
                'rows_with_any_feature':int(observed.any(axis=1).sum())})
        training=frame.loc[frame.season.between(2010,2023),features]
        output.append({'generation':path.parent.parent.name,'family':path.parent.name,
            'data_sha256':provenance['data_sha256'],'manifest_sha256':provenance['manifest_sha256'],
            'schedule_sha256':provenance['schedule_sha256'],'feature_count':len(features),
            'training_all_missing_features':training.columns[training.notna().sum().eq(0)].tolist(),'years':years})
    report={'scope':'built canonical component families, not complete fingerprint or model coverage',
            'coverage_does_not_authorize_training':True,'families':output}
    atomic_json(root/'results/canonical_family_coverage.json',report)
    for item in output:
        present=[y['season'] for y in item['years'] if y['team_target_rows']]
        print(item['generation'],item['family'],'years',min(present),max(present),
              'train_all_missing',len(item['training_all_missing_features']),flush=True)


if __name__=='__main__':main()
