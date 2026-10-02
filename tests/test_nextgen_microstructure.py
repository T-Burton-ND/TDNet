import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_microstructure import play_flags, game_sufficient_statistics, drive_sufficient_statistics


def plays(periods, scores, kinds=None, yards=None, minutes=None):
    n = len(periods)
    return pd.DataFrame({
        "game_id": [1]*n, "id": [str(i) for i in range(n)],
        "drive_number": [1]*n, "play_number": range(1, n+1),
        "offense": ["A"]*n, "defense": ["B"]*n, "home": ["A"]*n, "away": ["B"]*n,
        "offense_score": scores, "defense_score": [0]*n, "period": periods,
        "down": [1]*n, "distance": [10]*n, "yards_to_goal": [50]*n,
        "yards_gained": yards or [5]*n, "play_type": kinds or ["Rush"]*n,
        "clock.minutes": minutes or [10]*n, "clock.seconds": [0]*n,
    })


def test_postplay_score_does_not_make_scoring_play_garbage():
    p = plays([1, 1, 1], [0, 35, 35])
    flags = play_flags(p)
    assert flags.preplay_absolute_lead.tolist() == [0, 0, 35]
    assert flags.garbage.tolist() == [False, False, True]


def test_q4_whole_quarter_requires_every_qualifying_play_above_16():
    p = plays([3, 4, 4, 4], [17, 17, 10, 10])
    flags = play_flags(p)
    assert not flags.loc[flags.period.eq(4), "garbage"].any()
    p.offense_score = [17, 17, 17, 17]
    flags = play_flags(p)
    assert flags.loc[flags.period.eq(4), "garbage"].all()
    # Exact 16 is competitive, independent of later play outcomes.
    p.offense_score = [16]*4
    assert not play_flags(p).garbage.any()


def test_middle_eight_and_garbage_override_location_exception():
    p = plays([1, 2, 2, 3, 3], [0, 0, 0, 0, 0], minutes=[15, 4, 5, 12, 10])
    f = play_flags(p)
    assert f.middle_eight.tolist() == [False, True, False, True, False]
    p.offense_score = [35]*5
    p.yards_to_goal = 10
    f = play_flags(p)
    assert not f.middle_eight.any()
    assert f.red_zone.all()


def test_sufficient_stats_use_rush_attempts_and_reciprocal_defense():
    p = plays([1, 1, 1, 1], [0]*4, kinds=["Rush", "Rush", "Sack", "Pass Incompletion"], yards=[4, 8, -6, 0])
    d = game_sufficient_statistics(p).set_index("team")
    assert d.loc["A", "offense_rush_ypa__sum"] == 12
    assert d.loc["A", "offense_rush_ypa__n"] == 2
    assert d.loc["B", "defense_rush_ypa__sum"] == 12
    assert d.loc["A", "offense_sack_rate__sum"] == 1
    assert d.loc["A", "offense_sack_rate__n"] == 2


def test_no_description_or_probability_dependence_and_duplicate_rejection():
    p = plays([1, 1], [0, 0])
    enriched = p.assign(play_text="fictional injury or scoring description", win_probability=1.0)
    pd.testing.assert_frame_equal(play_flags(p), play_flags(enriched))
    with pytest.raises(ValueError, match="Duplicate"):
        play_flags(pd.concat([p, p]))


def test_legacy_pass_labels_preserve_dropback_statistics():
    # A completion, incompletion, interception and sack must all contribute to
    # the dropback denominator, in old and mixed-era source vocabularies alike.
    legacy = plays([1]*4, [0]*4,
                   kinds=["Pass Completion", "Pass Incompletion", "Pass Interception", "Sack"],
                   yards=[12, 0, 0, -4])
    modern = legacy.replace({"play_type": {"Pass Completion": "Pass Reception",
                                           "Pass Interception": "Interception"}})
    assert play_flags(legacy).dropback.all()
    pd.testing.assert_frame_equal(game_sufficient_statistics(legacy), game_sufficient_statistics(modern))
    stats = game_sufficient_statistics(legacy).set_index("team")
    assert stats.loc["A", "offense_sack_rate__n"] == 4
    assert stats.loc["A", "offense_sack_rate__sum"] == 1
    assert stats.loc["B", "defense_sack_rate__n"] == 4


def test_period_precedes_reused_drive_positions():
    p = plays([5, 4, 1], [24, 17, 0]).assign(drive_number=[1, 2, 1], play_number=[1, 1, 1])
    f = play_flags(p)
    assert f.period.tolist() == [1, 4, 5]
    assert f.preplay_absolute_lead.tolist() == [0, 0, 17]


def test_ambiguous_sequence_cannot_invent_preplay_scores():
    p = plays([1]*5, [0, 7, 14, 14, 14]).assign(play_number=[1, 2, 2, 3, 4])
    f = play_flags(p)
    assert f.preplay_absolute_lead.isna().tolist() == [False, True, True, True, False]
    assert not f.loc[1:3, "time_eligible"].any()
    # Renaming tied IDs must not change which observations have usable context.
    p.loc[1:2, "id"] = ["z", "a"]
    assert play_flags(p).preplay_absolute_lead.isna().tolist() == [False, True, True, True, False]


def test_drive_opportunity_net_points_and_pace_have_explicit_denominators():
    p = plays([1, 1], [0, 7], yards=[20, 30]).assign(drive_id="d1", yards_to_goal=[50, 30])
    drives = pd.DataFrame([dict(id="d1", game_id=1, offense="A", defense="B", start_period=1,
                                end_period=1, start_yards_to_goal=50, yards=50,
                                start_offense_score=0, end_offense_score=7,
                                start_defense_score=0, end_defense_score=0,
                                **{"elapsed.minutes": 0, "elapsed.seconds": 30})])
    d = drive_sufficient_statistics(drives, p).set_index("team")
    assert d.loc["A", "offense_points_per_opportunity__sum"] == 7
    assert d.loc["A", "offense_points_per_opportunity__n"] == 1
    assert d.loc["B", "defense_net_points_per_drive__sum"] == 7
    assert d.loc["A", "offense_drive_seconds_per_play__sum"] == 30
    assert d.loc["A", "offense_drive_seconds_per_play__n"] == 2
    assert d.loc["A", "offense_quality_drive_rate__sum"] == 1
