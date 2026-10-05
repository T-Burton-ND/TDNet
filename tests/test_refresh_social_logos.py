import pytest

from gridiron_ml.cli.publication.refresh_social_logos import _read_team_list


def test_read_team_list_ignores_comments_and_blank_lines(tmp_path):
    source = tmp_path / "teams.txt"
    source.write_text("# FBS teams\nIowa\n\nWashington\n", encoding="utf-8")
    assert _read_team_list(source) == ["Iowa", "Washington"]


def test_read_team_list_rejects_empty_and_duplicate_lists(tmp_path):
    source = tmp_path / "teams.txt"
    source.write_text("# no teams\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no team names"):
        _read_team_list(source)
    source.write_text("Iowa\nIowa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _read_team_list(source)
