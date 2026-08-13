import pytest

from gridiron_ml.publication.chart_contracts import validate_chart_domains


def test_probability_domain_is_bounded():
    assert validate_chart_domains(chart_kind="probability", y_domain=(0, 1))["y_domain"] == (0.0, 1.0)


def test_probability_domain_rejects_truncated_axis():
    with pytest.raises(ValueError, match="require y_domain"):
        validate_chart_domains(chart_kind="accuracy", y_domain=(0.4, 0.9))


def test_difference_domain_crosses_zero():
    validate_chart_domains(chart_kind="difference", y_domain=(-2, 3))
    with pytest.raises(ValueError, match="crossing zero"):
        validate_chart_domains(chart_kind="difference", y_domain=(1, 3))


def test_bar_domain_includes_zero():
    validate_chart_domains(chart_kind="bar", x_domain=(-1, 4))
    with pytest.raises(ValueError, match="include zero"):
        validate_chart_domains(chart_kind="bar", x_domain=(1, 4))


def test_margin_error_cannot_use_negative_floor():
    with pytest.raises(ValueError, match="start at zero"):
        validate_chart_domains(chart_kind="margin_error", y_domain=(-1, 10))
