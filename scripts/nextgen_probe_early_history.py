"""Bounded early-history probes; empty responses remain failed, never complete."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from gridiron_ml.pipeline.fetch.nextgen_acquisition import AcquisitionLedger
root=Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen')
ledger=AcquisitionLedger(root)
plan=[json.loads(line) for line in (root/'results/preflight/cfbd_request_manifest_v1.jsonl').read_text().splitlines()]
selected=[]
for r in plan:
 if ((r['endpoint']=='/ppa/players/games' and r['year'] in [2010,2011,2012]) or
     (r['endpoint']=='/stats/player/success/game' and r['year']==2012)):
  actual=ledger.read(r['request_id']) or r
  if actual['status']=='planned':selected.append(r)
print('Bounded probe count',len(selected),flush=True)
for item in selected:
 print('Probe',item['endpoint'],item['year'],item['week'],flush=True)
 result=subprocess.run([sys.executable,'scripts/nextgen_cfbd_acquire.py','--execute-stage','D',
                        '--endpoint',item['endpoint'],'--year',str(item['year']),'--max-requests','1'])
 if result.returncode:raise SystemExit(result.returncode)
 observed=ledger.read(item['request_id'])
 if observed is None:raise SystemExit('Expected request was not executed')
 if observed['status']=='success_complete':continue
 # Only the already inspected, exact empty-200 condition is covered by this
 # bounded review. Any different anomaly stops and returns to the operator.
 if not (observed['status']=='needs_review' and observed.get('http_status')==200
         and observed.get('error_summary')=='Empty 200 response requires coverage review'):
  raise SystemExit('Unreviewed anomaly: '+str(observed.get('error_summary')))
 observed.setdefault('reviews',[]).append({'at_utc':datetime.now(timezone.utc).isoformat(),
  'prior_status':'needs_review','prior_error':observed['error_summary'],'disposition':'failed_final',
  'reason':'Bounded early-history coverage probe returned no rows. Record failed coverage, not complete data or proven structural absence.',
  'evidence':{'endpoint':item['endpoint'],'parameters':observed['parameters'],'http_status':200,
              'review_scope':'Exact empty-200 response only; valid documented weekly regular-season query; every other anomaly halts.'}})
 observed['status']='failed_final';observed['completeness_status']='reviewed_unavailable_not_complete'
 ledger.write(observed)
 print('Reviewed empty response as failed coverage',item['request_id'],flush=True)
print('Bounded probes finished',flush=True)
