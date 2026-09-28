from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
root=Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen');f=root/'results/snapshots/2026-09-28-paused/results.parquet'
assert hashlib.sha256(f.read_bytes()).hexdigest()=='8f8935f77bd39a6b185642b9bf20fca8f87293cd5387c0f44b3c9ffccee69dd4'
t=pd.read_parquet(f);runs=t[t.row_type.eq('run')&t.status.eq('success')];cache={};ev=[]
for row in runs.itertuples():
 p=root/'experiments'/row.generation/row.run_id/'predictions.parquet'
 expected=json.loads(row.output_sha256)['predictions.parquet'];assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 x=pd.read_parquet(p);assert not x.duplicated(['season','target_game_id']).any();cache[(row.fingerprint_id,row.model,row.hyperparameter_setpoint)]=x
 ev.append({'path':str(p),'sha256':expected})
pairs=[(f'F06_F_{d}',f'F06_R_{d}') for d in 'abc']+[(f'F06_F_{d}',f'F09_F_{d}') for d in 'abc']+[('F09_F_a','F10_F_a'),('F10_F_a','F11_F_a'),('F11_F_a','F12_F_a')]
usable=set(map(tuple,t[t.row_type.eq('fingerprint_architecture_summary')&t.screening_usable.eq(True)][['fingerprint_id','model']].to_numpy()))
rows=[]
for a,b in pairs:
 for m in ('M2','M4'):
  if (a,m) not in usable or (b,m) not in usable:continue
  keys=sorted({k[2] for k in cache if k[:2]==(a,m)} & {k[2] for k in cache if k[:2]==(b,m)})
  for year in (2024,2025):
   am=[];bm=[];d=[];n=None;same=True
   for k in keys:
    aa=cache[a,m,k].query('season==@year');bb=cache[b,m,k].query('season==@year');same &= set(aa.target_game_id)==set(bb.target_game_id)
    z=aa.merge(bb,on=['season','target_game_id'],suffixes=('_a','_b'),validate='one_to_one');assert np.array_equal(z.actual_margin_a,z.actual_margin_b)
    assert n is None or n==len(z);n=len(z)
    ma=np.abs(z.actual_margin_a-z.predicted_margin_a).mean();mb=np.abs(z.actual_margin_b-z.predicted_margin_b).mean();am.append(ma);bm.append(mb);d.append(mb-ma)
   rows.append(dict(reference=a,candidate=b,model=m,year=year,games=n,identical_evaluation_ids=bool(same),matched_configs=len(keys),reference_mae_median=float(np.median(am)),candidate_mae_median=float(np.median(bm)),difference_of_medians=float(np.median(bm)-np.median(am)),median_paired_delta=float(np.median(d)),improving_configs=int(np.sum(np.array(d)<0)),min_paired_delta=float(min(d)),max_paired_delta=float(max(d))))
out={'scope':'Read-only analysis of previously fitted, hash-verified predictions from frozen 2026-09-28 12:52 UTC snapshot; no retraining. Intersection evaluation games and shared successful setpoint IDs per comparison.','snapshot_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'comparisons':rows,'prediction_evidence':ev}
Path('/tmp/tdnet_empirical_lessons.json').write_text(json.dumps(out,indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
