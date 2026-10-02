import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

from gridiron_ml.pipeline.fetch.nextgen_acquisition import AcquisitionLedger, verify_cache

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec = importlib.util.spec_from_file_location('archive_batch',SCRIPTS/'nextgen_archive_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def source(tmp_path, conf, game=100):
    p = tmp_path/(conf+'.parquet')
    pd.DataFrame(dict(season=[2020],week=[1],conference=[conf],game_id=[game],
                      play_id=['123'],athlete_id=[conf+'-player'],stat_type=['Rush'],stat=[3])).to_parquet(p)
    return dict(request_id=conf,cache_path=str(p),status='success_complete',week=1,
                conference_name=conf,**verify_cache(p))


def item(tmp_path):
    return dict(request_id='c'*64,endpoint='/plays/stats',game_id=100,year=2020,
                cache_path=str(tmp_path/'game.parquet'))


def test_disjoint_conference_union_preserves_events_without_http(tmp_path):
    ledger = AcquisitionLedger(tmp_path)
    result = batch.assemble_game(item(tmp_path),[source(tmp_path,'A'),source(tmp_path,'B')],ledger)
    assert result['status']=='success_complete' and result['row_count']==2
    assert result['attempt_count']==0 and result['http_status'] is None
    assert len(result['source_partitions'])==2
    assert verify_cache(Path(result['cache_path']),result)


def test_capped_source_cannot_become_complete_game(tmp_path):
    rec = source(tmp_path,'A')
    rec['status']='success_suspected_partial'
    with pytest.raises(ValueError,match='incomplete batch'):
        batch.assemble_game(item(tmp_path),[rec],AcquisitionLedger(tmp_path))


def test_conference_filter_mismatch_fails_closed(tmp_path):
    rec = source(tmp_path,'A')
    rec['conference_name']='B'
    with pytest.raises(ValueError,match='scope'):
        batch.assemble_game(item(tmp_path),[rec],AcquisitionLedger(tmp_path))


def test_missing_game_is_recorded_without_invented_zero_events(tmp_path):
    result = batch.assemble_game(item(tmp_path),[source(tmp_path,'A',game=101)],AcquisitionLedger(tmp_path))
    assert result['status']=='needs_review'
    assert result['completeness_status']=='empty_partition_union'
    assert not Path(result['cache_path']).exists()
