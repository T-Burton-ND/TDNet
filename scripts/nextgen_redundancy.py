#!/usr/bin/env python3
"""Measured, hash-bound redundancy diagnostics for a checked fingerprint."""
import argparse
import json
from pathlib import Path
import pandas as pd
from gridiron_ml.experiments.nextgen_screening import checked_fingerprint
from gridiron_ml.experiments.nextgen_reduction import redundancy_report
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fingerprints',nargs='+')
    args=parser.parse_args()
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    for fingerprint in args.fingerprints:
        _,_,records,provenance=checked_fingerprint(root,fingerprint)
        source=root/'fingerprints'/fingerprint/'values.parquet'
        frame=pd.read_parquet(source)
        report=redundancy_report(frame,records)
        if sha256_file(source)!=provenance['data_sha256']:
            raise ValueError('Fingerprint changed during diagnostic computation')
        dest=root/'results'/fingerprint.split('_')[0]/fingerprint
        dest.mkdir(parents=True,exist_ok=True)
        path=dest/'redundancy.parquet'
        temporary=path.with_suffix('.tmp.parquet')
        report.to_parquet(temporary,index=False);temporary.replace(path)
        summary={'fingerprint_id':fingerprint,'source_data_sha256':provenance['data_sha256'],
                 'source_manifest_sha256':provenance['manifest_sha256'],
                 'output_sha256':sha256_file(path),'rows':len(frame),
                 'source_seasons':sorted(int(y) for y in frame.season.unique()),
                 'minimum_joint_rows':100,'category_counts':report.category.value_counts().to_dict(),
                 'removal_authorized':False,
                 'code_sha256':sha256_file(ROOT/'src/gridiron_ml/experiments/nextgen_reduction.py')}
        atomic_json(dest/'redundancy.provenance.json',summary)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
