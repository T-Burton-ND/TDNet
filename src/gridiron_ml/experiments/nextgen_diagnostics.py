"""Global advanced-feature diagnostic ranking and compact future-target plots.

Callers must supply verified, terminal screening contexts. Ranking is descriptive
and does not authorize feature pruning or replace measured acceptance.
"""
import hashlib
import json
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd

from .nextgen_contract import parse_fingerprint_id


def advanced_record(record):
    """Exclude direct atomic baseline fields from the advanced-feature budget."""
    return (record['generation'] != 'F06' or record.get('operation') != 'identity'
            or record['name'].startswith(('opp_adj_', 'time_adj_', 'f06_', 'graph_', 'schedule_'))
            or (record['name'].startswith(('offense_', 'defense_'))
                and not record['name'].endswith(('_plays', '_drives', '_yards', '_points'))))


def feature_identity(record):
    """Share identical scientific definitions across design labels, never equations."""
    annotations = {'designs', 'interpretation', 'film_intuition', 'source_inspiration',
                   'provenance', 'code_path'}
    definition = {k: v for k, v in record.items() if k not in annotations}
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def rank_diagnostic_features(contexts, *, limit=1000, required_names=()):
    """Equal lineage, design and generation weight among contexts containing a feature.

    Each context's M2/M4 importance must already be the equal-run architecture
    consensus. Absent/pruned features are not treated as observed zero effects.
    """
    if not 0 < limit <= 1000:
        raise ValueError('Global diagnostic limit must be between one and 1000')
    required_names = set(required_names)
    if len(required_names) > limit:
        raise ValueError('Required diagnostics exceed the global limit')
    rows, records, locations, seen = [], {}, {}, set()
    for context in contexts:
        fingerprint = context['fingerprint_id']
        generation, lineage, design = parse_fingerprint_id(fingerprint)
        if fingerprint in seen:
            raise ValueError('Duplicate diagnostic screening context')
        seen.add(fingerprint)
        manifest = context['records']
        table = context['consensus'].set_index('source_feature')
        names = [r['name'] for r in manifest]
        if (len(names) != len(set(names)) or table.index.duplicated().any()
                or set(names) != set(table.index)):
            raise ValueError('Diagnostic SHAP schema differs from the manifest')
        values = table[['M2_importance', 'M4_importance']].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid diagnostic importance')
        totals = values.sum(axis=0)
        if not (np.isclose(totals, 1., atol=1e-6) | np.isclose(totals, 0.)).all():
            raise ValueError('Diagnostic architecture importance must be normalized')
        for record in manifest:
            if not advanced_record(record):
                continue
            identity = feature_identity(record)
            records.setdefault(identity, record)
            locations.setdefault(identity, []).append(fingerprint)
            rows.append({'identity': identity, 'generation': generation, 'design': design,
                         'lineage': lineage, **table.loc[record['name'], ['M2_importance', 'M4_importance']].to_dict()})
    if not rows:
        raise ValueError('No measured advanced-feature contexts')
    columns = ['M2_importance', 'M4_importance']
    table = pd.DataFrame(rows)
    by_design = table.groupby(['identity', 'generation', 'design'])[columns].mean()
    by_generation = by_design.groupby(['identity', 'generation']).mean()
    scores = by_generation.groupby('identity').mean()
    scores['joint_importance'] = scores[columns].mean(axis=1)
    scores = scores.reset_index().sort_values(['joint_importance', 'identity'], ascending=[False, True])
    scores['rank'] = range(1, len(scores)+1)
    chosen = set()
    for name in sorted(required_names):
        matches = [identity for identity in scores.identity if records[identity]['name'] == name]
        if not matches:
            raise ValueError(f'Required diagnostic lacks measured advanced-feature evidence: {name}')
        chosen.add(matches[0])
    for identity in scores.identity:
        if len(chosen) >= min(limit, len(scores)):
            break
        chosen.add(identity)
    result = []
    for row in scores.loc[scores.identity.isin(chosen)].to_dict(orient='records'):
        identity = row['identity']
        result.append({**row, 'record': records[identity],
                       'required_diagnostic': records[identity]['name'] in required_names,
                       'screening_contexts': sorted(locations[identity]),
                       'ranking_scope': 'conditional on feature presence; equal lineage/design/generation, then equal architecture'})
    return result


def render_diagnostic(destination, item, frame, *, fingerprint, data_sha256, survival_status,
                      source_documentation=None):
    """Render pre-target state versus future margin and empirical wins, never selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    record = item['record']
    documentation = source_documentation or {}
    units = documentation.get('units', record['units'])
    name = record['name']
    required = {name, 'season', 'next_game_margin', 'next_game_win'}
    if (not required <= set(frame) or frame.season.isna().any()
            or frame.season.gt(2025).any()):
        raise ValueError('Diagnostic requires explicit pre-2026 future-target data')
    sample = frame[[name, 'next_game_margin', 'next_game_win']].replace([np.inf, -np.inf], np.nan).dropna()
    if not sample.next_game_win.isin([0, 1]).all():
        raise ValueError('Future win target must be binary')
    pearson = spearman = None
    if len(sample) >= 3 and sample[name].nunique() > 1 and sample.next_game_margin.nunique() > 1:
        pearson = float(sample[name].corr(sample.next_game_margin, method='pearson'))
        spearman = float(sample[name].corr(sample.next_game_margin, method='spearman'))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if len(sample):
        axes[0].hexbin(sample[name], sample.next_game_margin, gridsize=40, mincnt=1, cmap='Blues')
        sample = sample.copy()
        sample['bin'] = pd.qcut(sample[name], min(10, sample[name].nunique()), duplicates='drop') if sample[name].nunique() > 1 else 0
        grouped = sample.groupby('bin', observed=True).agg(
            value=(name, 'mean'), margin=('next_game_margin', 'mean'), wins=('next_game_win', 'mean'))
        axes[0].plot(grouped.value, grouped.margin, 'o-', color='#ab4d28', label='Equal-count bin means')
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(grouped.value, grouped.wins*100, 'o-', color='#176b87')
    else:
        for ax in axes:
            ax.text(.5, .5, 'No finite paired observations', transform=ax.transAxes, ha='center')
    axes[0].axhline(0, color='grey', lw=.6)
    axes[0].set_ylabel('Next-game team margin (points)')
    axes[1].axhline(50, color='grey', lw=.6)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Next-game empirical win rate (%)')
    for ax in axes:
        ax.set_xlabel(fill(f"Prior feature value ({units})", width=48))
        ax.grid(axis='y', alpha=.15)
    fig.suptitle(name.replace('_', ' '), fontsize=13)
    years = f'{int(frame.season.min())}–{int(frame.season.max())}'
    fig.supxlabel(f'{years} design-informed corpus • {len(sample):,} finite team-target rows\nDescriptive association; repeated teams and games are dependent.', fontsize=9)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stem = name+'--'+item['identity'][:12]
    image = destination/(stem+'.png')
    fig.savefig(image, dpi=150)
    plt.close(fig)
    positive_negative = ('Positive means the first input exceeds the second; negative means the reverse.'
                         if record.get('operation') == 'difference'
                         else 'Positive/negative values are above/below zero in the declared equation and units; desirability follows the football interpretation, not the sign alone.')
    positive_negative = documentation.get('positive_negative_meaning', positive_negative)
    text = f"""# {name}

- Generation: {record['generation']}. Equation: `{record['equation_excel']}`.
- Inputs: {', '.join(record['source_inputs'])}. Units: {units}.
- Positive/negative meaning: {positive_negative} Direction: {documentation.get('direction', record.get('direction', 'unspecified'))}.
- Football interpretation: {documentation.get('interpretation', record['interpretation'])}
- Film intuition: {documentation.get('film_intuition', record['film_intuition'])}
- Source/inspiration: {record['source_inspiration']}
- Aggregation: {record['aggregation_window']}. Availability: {record['availability_rule']}.
- Coverage/sample rule: {record['minimum_sample_rule']}; {len(sample):,} finite rows of {len(frame):,} in {years}.
- Missingness: {record['missingness_policy']}
- Garbage time: {record['garbage_time_handling']}
- Future targets: next-game team margin and empirical next-game win rate.
- Associations: Pearson {pearson}; Spearman {spearman}. Descriptive only; curved relationships may be informative and correlation does not authorize pruning.
- Consensus SHAP: M2 {item['M2_importance']:.8f}; M4 {item['M4_importance']:.8f}; joint {item['joint_importance']:.8f}. Global advanced-feature rank {item['rank']}. {item['ranking_scope']}.
- Screening contexts: {', '.join(item['screening_contexts'])}.
- Survival/pruning: {survival_status}
- Matchup counterpart: `{record['matchup_counterpart']}`. Formula: `{record['matchup_formula']}`.
- Plot source: {fingerprint}, data SHA256 `{data_sha256}`. Both development years informed design; this is not unbiased holdout evidence.

![Prior-state relationship to future outcomes]({image.name})
"""
    if source_documentation:
        text += '\n## Source computation supplement\n\n'
        text += 'The frozen manifest above identifies the trained state column. This supplement documents its upstream computation without changing training inputs.\n\n'
        text += '\n'.join(f'- {key.replace("_", " ")}: {value}'
                          for key, value in source_documentation.items())+'\n'
    (destination/(stem+'.md')).write_text(text)
    return {'feature': name, 'identity': item['identity'], 'generation': record['generation'],
            'png': image.name, 'markdown': stem+'.md', 'rank': item['rank'],
            'joint_importance': item['joint_importance'], 'rows': len(sample),
            'data_sha256': data_sha256, 'source_fingerprint': fingerprint}
