"""Finalize F06 only from verified full/reduced screening and reduction evidence."""
import json
from pathlib import Path
import pandas as pd
from .nextgen_results import collect_results
from .nextgen_screening import checked_fingerprint
from .nextgen_reduction import check_survivors, measured_acceptance, source_importance_consensus
from .nextgen_recommendations import rank_lineages
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file

ROOT=Path(__file__).resolve().parents[3]


def finalize_f06(root):
    root=Path(root)
    # Collection revalidates successful run computation, inputs and outputs.
    table=collect_results(root)
    runs=table.loc[table.row_type.eq('run')]
    summaries=table.loc[table.row_type.eq('fingerprint_architecture_summary')]
    points=json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    evidence={}; recommendations=[];acceptances={}
    def bind(path):
        evidence[str(path)]=sha256_file(path)
    for design in 'abc':
        names=[f'F06_{variant}_{design}' for variant in ['F','R']]
        records_by_id={};frames={}
        for name in names:
            _,_,records,provenance=checked_fingerprint(root,name)
            records_by_id[name]=records
            folder=root/'fingerprints'/name
            frames[name]=pd.read_parquet(folder/'values.parquet')
            for filename in ['values.parquet','feature_manifest.json','provenance.json']:
                bind(folder/filename)
            cells=runs.loc[runs.fingerprint_id.eq(name)]
            shap={}
            for row in cells.itertuples(index=False):
                directory=root/'experiments/F06'/row.run_id
                # Progress files never satisfy the finalization evidence list.
                bind(directory/'result.json')
                if row.status=='success':
                    shap[(row.model,row.hyperparameter_setpoint)]=pd.read_parquet(directory/'source_shap.parquet')
                    bind(directory/'source_shap.parquet');bind(directory/'predictions.parquet')
            source_importance_consensus(shap,points,records)
        full,reduced=names
        full_records=records_by_id[full];reduced_records=records_by_id[reduced]
        survivors=[r['name'] for r in reduced_records]
        check_survivors(survivors,full_records)
        if reduced_records!=[r for r in full_records if r['name'] in set(survivors)]:
            raise ValueError('Reduced feature definitions differ from full reference')
        if len(survivors)>=len(full_records):
            raise ValueError('F06 reduction removed no features')
        # Same paired targets, outcomes, metadata and retained predictor values.
        full_names={r['name'] for r in full_records}
        columns=[c for c in frames[full] if c not in full_names or c in survivors]
        pd.testing.assert_frame_equal(frames[full][columns],frames[reduced][columns])
        reference=runs.loc[runs.fingerprint_id.eq(full)]
        candidate=runs.loc[runs.fingerprint_id.eq(reduced)]
        acceptance=measured_acceptance(reference,candidate,points,design=design)
        if not acceptance['accepted']:
            raise ValueError(f'{reduced} fails measured reduction acceptance')
        acceptances[reduced]=acceptance
        diagnostic=root/'results/F06'/full
        provenance=json.loads((diagnostic/'redundancy.provenance.json').read_text())
        if (provenance['source_data_sha256']!=sha256_file(root/'fingerprints'/full/'values.parquet')
            or provenance['source_manifest_sha256']!=sha256_file(root/'fingerprints'/full/'feature_manifest.json')
            or provenance['output_sha256']!=sha256_file(diagnostic/'redundancy.parquet')):
            raise ValueError('Full redundancy evidence is stale')
        bind(diagnostic/'redundancy.provenance.json');bind(diagnostic/'redundancy.parquet')
        ranked=rank_lineages(summaries,generation='F06',design=design,eligible_fingerprints=set(names))
        ranked['status']='success';ranked['max_design_year_used']=2025
        ranked['prospective_boundary_year']=2026
        ranked['run_id']=ranked.fingerprint_id+'__recommendation'
        recommendations.extend(json.loads(ranked.to_json(orient='records')))
    bind(root/'results/source_semantics_audit.json')
    output=root/'results/F06'
    output.mkdir(parents=True,exist_ok=True)
    report={'generation':'F06','rows':recommendations,'reduction_acceptance':acceptances,
            'evidence':[{'path':p,'sha256':h} for p,h in sorted(evidence.items())]}
    atomic_json(output/'recommendations.json',report)
    collect_results(root)
    final={'generation':'F06','training_finalized':True,'shap_finalized':True,
           'reduction_finalized':True,'lineages_compared':True,'recommendations_finalized':True,
           'recommendations_sha256':sha256_file(output/'recommendations.json'),
           'incomplete_coverage':any(r['incomplete_coverage'] for r in recommendations),
           'evidence':report['evidence']}
    atomic_json(output/'finalization.json',final)
    return final


if __name__=='__main__':
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    print(json.dumps(finalize_f06(root),indent=2))
