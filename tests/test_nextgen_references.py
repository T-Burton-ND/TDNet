import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_references import annual_reference, freeze_reference


def test_equal_team_then_season_weight_and_no_target_year():
    frame = pd.DataFrame([
        (1,2023,1,'A',0.), (2,2023,2,'A',0.), (3,2023,1,'B',10.),
        (4,2024,1,'A',20.), (5,2025,1,'A',999.)],
        columns=['target_game_id','season','week','team','x'])
    result = annual_reference(frame, ['x'], 2025)
    assert result['values']['x'] == 12.5
    assert result['source_seasons'] == [2023,2024]
    assert result['support']['x'] == dict(team_weeks=4, team_seasons=3, seasons=2)
    assert annual_reference(frame, ['x'], 2026)['values']['x'] == (5+20+999)/3


def test_duplicate_week_missing_values_and_freeze(tmp_path):
    frame = pd.DataFrame([(1,2025,1,'A',0.),(2,2025,1,'A',10.),(3,2025,2,'A',20.)],
        columns=['target_game_id','season','week','team','x'])
    frame['missing'] = float('nan')
    result = annual_reference(frame, ['x','missing'], 2026)
    assert result['values'] == {'x':12.5,'missing':None}
    path = tmp_path/'reference.json'
    freeze_reference(path,result)
    freeze_reference(path,result)
    with pytest.raises(ValueError,match='Frozen'):
        freeze_reference(path,{**result,'reference_season':2025})
    frame.loc[0,'season'] = 2026
    with pytest.raises(ValueError,match='Quarantined'):
        annual_reference(frame,['x'],2026)
