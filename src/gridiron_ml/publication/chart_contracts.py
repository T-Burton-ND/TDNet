"""Backend-independent contracts for publication chart axis domains."""

from __future__ import annotations

from collections.abc import Sequence


def _domain(values: Sequence[float] | None, *, name: str) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two bounds.")
    low, high = (float(value) for value in values)
    if not low < high:
        raise ValueError(f"{name} must be strictly increasing.")
    return low, high


def validate_chart_domains(
    *,
    chart_kind: str,
    y_domain: Sequence[float] | None = None,
    x_domain: Sequence[float] | None = None,
) -> dict[str, tuple[float, float] | None]:
    """Validate declared domains for common publication chart types.

    Domains are intentionally explicit.  A bounded chart must declare the
    complete scientific range rather than relying on backend autoscaling.
    ``difference`` charts must cross zero and ``bar`` charts must include zero
    on the value axis.
    """
    kind = str(chart_kind).lower()
    x = _domain(x_domain, name="x_domain")
    y = _domain(y_domain, name="y_domain")
    if kind in {"probability", "proportion", "accuracy", "brier", "calibration"}:
        if y != (0.0, 1.0):
            raise ValueError(f"{kind} charts require y_domain=(0, 1), got {y}.")
    elif kind == "difference":
        if y is None or not (y[0] <= 0.0 <= y[1]):
            raise ValueError("difference charts must declare a y-domain crossing zero.")
    elif kind == "bar":
        value_domain = x if x is not None else y
        if value_domain is None or not (value_domain[0] <= 0.0 <= value_domain[1]):
            raise ValueError("bar charts must include zero on their value axis.")
    elif kind in {"margin_error", "margin"}:
        if y is not None and y[0] < 0.0:
            raise ValueError("margin-error domains must start at zero or above.")
    else:
        raise ValueError(f"Unknown chart kind: {chart_kind}")
    return {"x_domain": x, "y_domain": y}
