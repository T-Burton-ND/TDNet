"""Assemble accumulated fingerprints from checked canonical family artifacts."""
import json
from pathlib import Path
import pandas as pd
from .nextgen_artifacts import NextgenModelBoundary
from .nextgen_contract import parse_fingerprint_id, parent_of, validate_feature_manifest
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file

ROOT=Path(__file__).resolve().parents[3]


def join_family(parent, family):
    keys=['target_game_id','team']
    if parent.duplicated(keys).any() or family.index.duplicated().any():
        raise ValueError('Duplicate assembly identity')
    if set(family.columns)&set(parent.columns):
        raise ValueError('Family would overwrite inherited values')
    joined=parent.merge(family.reset_index(),on=keys,how='inner',validate='one_to_one')
    counts=joined.groupby('target_game_id').size()
    paired=counts.index[counts.eq(2)]
    joined=joined.loc[joined.target_game_id.isin(paired)].copy()
    if joined.empty:
        raise ValueError('No paired targets after family assembly')
    return joined, sorted(set(parent.target_game_id)-set(joined.target_game_id))


def assert_family_values(frame, family):
    keyed=frame.set_index(['target_game_id','team'])
    names=[n for n in family if n in keyed]
    expected=family.reindex(keyed.index)
    if not keyed.index.isin(family.index).all():
        raise ValueError('Fingerprint rows absent from canonical family')
    for name in names:
        left,right=keyed[name],expected[name]
        if not (left.eq(right)|(left.isna()&right.isna())).all():
            raise ValueError(f'Fingerprint value differs from canonical family: {name}')


def assemble_full(root, generation, design, families):
    """Assemble a finalized full family list; never infer that list from disk.

    Callers choose the complete generation scope. Each supplied canonical family
    and every inherited family is revalidated before writing the child.
    """
    fingerprint=generation+'_F_'+design
    parse_fingerprint_id(fingerprint)
    if generation == 'F06':
        raise ValueError('F06 uses the exact baseline builder')
    if not families or len(families) != len(set(families)):
        raise ValueError('Explicit unique generation family list required')
    if any(not family.endswith('_'+design) for family in families):
        raise ValueError('Cross-design family ancestry is forbidden')
    parent_id=parent_of(fingerprint)
    parent=root/'fingerprints'/parent_id
    pp=json.loads((parent/'provenance.json').read_text())
    for name,key in [('values.parquet','data_sha256'),('feature_manifest.json','manifest_sha256')]:
        if sha256_file(parent/name)!=pp[key]:
            raise ValueError('Parent fingerprint provenance mismatch')
    frame=pd.read_parquet(parent/'values.parquet')
    records=json.loads((parent/'feature_manifest.json').read_text())
    inherited=pp.get('canonical_families', [])
    if parent_id[:3] != 'F06' and not inherited:
        raise ValueError('Parent lacks canonical ancestry')
    for item in inherited:
        boundary=NextgenModelBoundary.from_canonical(root,item['generation'],item['family'],Path(item['manifest_path']))
        assert_family_values(frame,boundary.for_fit())
    canonical=list(inherited)
    excluded=set()
    family_exclusions={}
    for family in families:
        manifest=root/'feature_families'/generation/family/'feature_manifest.json'
        boundary=NextgenModelBoundary.from_canonical(root,generation,family,manifest)
        values=boundary.for_fit()
        frame,lost=join_family(frame,values)
        excluded.update(lost)
        family_exclusions[family]=lost
        assert_family_values(frame,values)
        new_records=json.loads(manifest.read_text())
        if any(r['generation'] != generation for r in new_records):
            raise ValueError('New family has wrong generation')
        records+=new_records
        canonical.append({'generation':generation,'family':family,'manifest_path':str(manifest)})
    schema=json.loads((ROOT/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    validate_feature_manifest(records,schema)
    if any(design not in r['designs'] for r in records):
        raise ValueError('Cross-design ancestry is forbidden')
    if frame.season.gt(2025).any():
        raise ValueError('Quarantined assembly row')
    dest=root/'fingerprints'/fingerprint
    dest.mkdir(parents=True,exist_ok=True)
    feature_manifest=dest/'feature_manifest.json'
    content=json.dumps(records,indent=2)+'\n'
    if feature_manifest.exists() and feature_manifest.read_text()!=content:
        raise ValueError('Version fingerprint before changing its manifest')
    feature_manifest.write_text(content)
    tmp=dest/'values.tmp.parquet';data=dest/'values.parquet'
    frame.to_parquet(tmp,index=False,compression='zstd');tmp.replace(data)
    provenance={'fingerprint_id':fingerprint,'ancestry':[parent_id],
                'parent_data_sha256':pp['data_sha256'],'source_sha256':pp['source_sha256'],
                'schedule_sha256':pp['schedule_sha256'],'data_sha256':sha256_file(data),
                'manifest_sha256':sha256_file(feature_manifest),'feature_count':len(records),
                'max_design_year':int(frame.season.max()),'rows':len(frame),
                'excluded_parent_game_ids':sorted(excluded),'family_exclusions':family_exclusions,'source_semantics_audit_required_before_training':True,
                'canonical_families':canonical}
    atomic_json(dest/'provenance.json',provenance)
    return {'fingerprint':fingerprint,'rows':len(frame),'features':len(records),'excluded_games':len(excluded)}


def assemble_f09_full(root, design):
    return assemble_full(root, 'F09', design, ['play_microstructure_'+design])


if __name__=='__main__':
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    for design in 'abc':print(json.dumps(assemble_f09_full(root,design)),flush=True)
