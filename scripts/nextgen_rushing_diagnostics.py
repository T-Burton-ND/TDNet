#!/usr/bin/env python3
"""Required prior-state rushing-quarter diagnostics; no SHAP/ranking claims."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file,atomic_json


def main():
    config=json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())
    root=Path(config['artifact_root']); source=root/'fingerprints/F09_F_a'
    provenance=json.loads((source/'provenance.json').read_text())
    data=source/'values.parquet';manifest=source/'feature_manifest.json'
    if sha256_file(data)!=provenance['data_sha256'] or sha256_file(manifest)!=provenance['manifest_sha256']:
        raise ValueError('Fingerprint artifact changed')
    frame=pd.read_parquet(data)
    if frame.season.gt(2025).any():raise ValueError('Quarantined diagnostic data')
    records={r['name']:r for r in json.loads(manifest.read_text())}
    dest=root/'feature_diagnostics';dest.mkdir(parents=True,exist_ok=True)
    entries=[]
    for side in ('offense','defense'):
        name=side+'_rush_ypa_q4_minus_q1'; record=records[name]
        sample=frame[[name,'next_game_margin','next_game_win','season']].replace([np.inf,-np.inf],np.nan).dropna()
        sample['bin']=pd.qcut(sample[name],10,duplicates='drop')
        grouped=sample.groupby('bin',observed=True)
        bins=grouped.agg(x=(name,'mean'),margin=('next_game_margin','mean'),wins=('next_game_win','mean'),n=(name,'size'))
        pearson=float(sample[name].corr(sample.next_game_margin,method='pearson'))
        spearman=float(sample[name].corr(sample.next_game_margin,method='spearman'))
        plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
        fig,axes=plt.subplots(1,2,figsize=(12,5.1),constrained_layout=True)
        color='#176b87' if side=='offense' else '#925029'
        axes[0].scatter(sample[name],sample.next_game_margin,s=4,alpha=.045,color=color,rasterized=True)
        axes[0].plot(bins.x,bins.margin,'o-',color='#132d40',lw=2,label='Equal-count bin means')
        axes[0].axhline(0,color='#999999',lw=.7);axes[0].set_ylabel('Next-game team margin (points)')
        axes[0].legend(frameon=False,loc='upper left',fontsize=9)
        axes[1].plot(bins.x,100*bins.wins,'o-',color=color,lw=2)
        axes[1].axhline(50,color='#999999',lw=.7);axes[1].set_ylim(0,100)
        axes[1].set_ylabel('Next-game empirical win rate (%)')
        for ax in axes:
            ax.set_xlabel('Prior Q4 − Q1 rushing yards per attempt')
            ax.grid(axis='y',alpha=.15)
        label='Offensive rushing' if side=='offense' else 'Rushing allowed by defense'
        fig.suptitle(f'{label}: late-game change and the next game',fontsize=15,fontweight='bold')
        fig.supxlabel(f'2010–2025 design-informed corpus • {len(sample):,} team-target rows • Spearman ρ = {spearman:.3f}\nDescriptive association; repeated teams/games are not independent. SHAP and pruning not yet run.',fontsize=9)
        image=dest/(name+'.png');fig.savefig(image,dpi=160);plt.close(fig)
        meaning=('Positive: better late-game rushing gain per attempt.' if side=='offense' else 'Positive: more rushing yards allowed per attempt late in games.')
        md=f'''# {name}

Generation: F09. Required diagnostic, not a consensus-SHAP-selected feature.

- Equation: `{record['equation_excel']}`. Inputs: {', '.join(record['source_inputs'])}. Units: {record['units']}.
- Meaning: {meaning} Negative values indicate the opposite; zero indicates equal quarter averages.
- Football interpretation: {record['interpretation']}
- Film intuition: {record['film_intuition']}
- Source/inspiration: {record['source_inspiration']}
- Aggregation: {record['aggregation_window']}
- Coverage/sample rule: {record['minimum_sample_rule']}; {len(sample):,} finite paired-feature/outcome team-target rows out of {len(frame):,} assembled rows, 2010–2025.
- Garbage time: {record['garbage_time_handling']}
- Availability: {record['availability_rule']}
- Future targets: next-game team margin and empirical next-game win rate. These are not same-game outcome plots.
- Margin associations: Pearson {pearson:.4f}; Spearman {spearman:.4f}. Descriptive only, not a selection rule. Bin means pool design-informed 2010–2025 data; repeated teams/opponents are dependent.
- SHAP: not yet measured. Survival/pruning: present in full F09; reductions not run.
- Matchup counterpart: `{record['matchup_counterpart']}`.
- Source fingerprint/data hash: F09_F_a / `{provenance['data_sha256']}`.

![Prior-state versus next-game outcomes]({image.name})
'''
        (dest/(name+'.md')).write_text(md)
        entries.append({'feature':name,'generation':'F09','png':image.name,'markdown':name+'.md',
                        'selection_basis':'explicit_required_rushing_diagnostic','shap_available':False,
                        'data_sha256':provenance['data_sha256'],'rows':len(sample)})
    index=dest/'index.json'
    old=json.loads(index.read_text()) if index.exists() else []
    names={e['feature'] for e in entries};entries=[e for e in old if e['feature'] not in names]+entries
    if len(entries)>1000:raise ValueError('Global diagnostic feature cap exceeded')
    atomic_json(index,entries)
    print(json.dumps(entries,indent=2))


if __name__=='__main__':main()
