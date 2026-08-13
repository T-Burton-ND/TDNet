#!/usr/bin/env python3
"""Build a reviewable Sunday retrospective publication bundle.

This command scores an already immutable Thursday prediction bundle after the
source completeness report passes. It writes metrics, cumulative summaries,
baseline/Vegas comparison tables, a compact figure, and draft-only copy into a
new output directory. It never edits the prediction bundle and never sends a
public post.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.publication.bundles import (  # noqa: E402
    score_prediction_bundle,
    sha256_file,
    verify_prediction_bundle,
)
from gridiron_ml.publication.chart_contracts import validate_chart_domains  # noqa: E402
from gridiron_ml.publication.figure_theme import MODEL_COLORS, TDNET_COLORS, apply_tdnet_theme  # noqa: E402


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    probability = pd.to_numeric(frame.get("pred_home_win_probability"), errors="coerce")
    actual_win = pd.to_numeric(frame.get("actual_home_win"), errors="coerce")
    margin_error = pd.to_numeric(frame.get("absolute_margin_error"), errors="coerce")
    valid_probability = probability.notna() & actual_win.notna()
    valid_margin = margin_error.notna()
    output: dict[str, float | int] = {"games": int(frame["game_id"].nunique())}
    output["winner_accuracy"] = float(frame.loc[valid_probability, "winner_correct"].mean()) if valid_probability.any() else float("nan")
    output["margin_mae"] = float(margin_error.loc[valid_margin].mean()) if valid_margin.any() else float("nan")
    if valid_probability.any():
        clipped = probability.loc[valid_probability].clip(1e-8, 1 - 1e-8)
        y = actual_win.loc[valid_probability]
        output["brier_score"] = float(((clipped - y) ** 2).mean())
        output["log_loss"] = float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())
    else:
        output["brier_score"] = float("nan")
        output["log_loss"] = float("nan")
    return output


def _weekly_and_cumulative(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = scored.copy()
    if "week" not in frame:
        frame["week"] = 0
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").fillna(0).astype(int)
    weekly_rows = []
    cumulative_rows = []
    for (model, objective), group in frame.groupby(["model_name", "objective"], dropna=False):
        prior = []
        for week, week_frame in group.groupby("week", sort=True):
            prior.append(week_frame)
            weekly_rows.append({"model_name": model, "objective": objective, "week": int(week), **_metrics(week_frame)})
            cumulative_rows.append({"model_name": model, "objective": objective, "week": int(week), **_metrics(pd.concat(prior, ignore_index=True))})
    return pd.DataFrame(weekly_rows), pd.DataFrame(cumulative_rows)


def _figure(cumulative: pd.DataFrame, path: Path) -> None:
    apply_tdnet_theme()
    if cumulative.empty:
        raise ValueError("No cumulative metrics are available for the Sunday figure.")
    ranked = cumulative.groupby("model_name", as_index=False)["margin_mae"].last().sort_values("margin_mae").head(12)["model_name"]
    plot = cumulative[cumulative["model_name"].isin(ranked)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    max_mae = float(pd.to_numeric(plot["cumulative_margin_mae"], errors="coerce").max()) if "cumulative_margin_mae" in plot else float(pd.to_numeric(plot["margin_mae"], errors="coerce").max())
    validate_chart_domains(chart_kind="margin_error", y_domain=(0.0, max(1.0, max_mae * 1.05)))
    for model, group in plot.groupby("model_name"):
        color = MODEL_COLORS.get(str(model), TDNET_COLORS["slate"])
        axes[0].plot(group["week"], group["margin_mae"], marker="o", lw=1.5, label=str(model), color=color)
        axes[1].plot(group["week"], group["winner_accuracy"], marker="o", lw=1.5, label=str(model), color=color)
    axes[0].set(xlabel="Week", ylabel="Margin MAE (points)", title="Weekly margin error")
    axes[1].set(xlabel="Week", ylabel="Winner accuracy", title="Weekly winner accuracy", ylim=(0, 1))
    axes[1].axhline(0.5, color=TDNET_COLORS["medium_gray"], ls=":", lw=1)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.4)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="Completed CFBD results parquet/CSV.")
    parser.add_argument("--snapshot-completeness", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Sunday output: {args.output_root}")
    snapshot = json.loads(args.snapshot_completeness.read_text(encoding="utf-8"))
    if snapshot.get("status") != "pass" or snapshot.get("certification") != "weekly_snapshot_certified":
        raise RuntimeError("Sunday publication requires a certified source snapshot.")
    verification = verify_prediction_bundle(args.bundle)
    if not verification["valid"]:
        raise RuntimeError(f"Cannot score invalid prediction bundle: {verification['failures']}")
    results = pd.read_parquet(args.results) if args.results.suffix == ".parquet" else pd.read_csv(args.results)
    args.output_root.mkdir(parents=True, exist_ok=False)
    scored_root = args.output_root / "scoring"
    tables = score_prediction_bundle(args.bundle, results, output_root=scored_root)
    scored = tables["scored_predictions"]
    weekly, cumulative = _weekly_and_cumulative(scored)
    weekly.to_csv(args.output_root / "weekly_metrics.csv", index=False)
    cumulative.to_csv(args.output_root / "cumulative_metrics.csv", index=False)
    tables["scorecard"].to_csv(args.output_root / "scorecard.csv", index=False)
    model_labels = tables["scorecard"].loc[:, ["model_name", "objective"]].drop_duplicates()
    model_labels["baseline_or_model"] = model_labels["model_name"].astype(str).str.contains("vegas|random|knn|point|prior", case=False, regex=True).map({True: "baseline", False: "model"})
    model_labels.to_csv(args.output_root / "baseline_comparison.csv", index=False)
    _figure(weekly, args.output_root / "sunday_performance.png")
    (args.output_root / "blog_summary.md").write_text(
        f"# TDNet Sunday retrospective — {datetime.now(timezone.utc).date()}\n\n"
        "This draft was generated from an immutable prediction bundle after certified source checks. "
        "It is reviewable output, not an automatic publication.\n\n"
        f"Scored games: {int(scored['game_id'].nunique())}.\n",
        encoding="utf-8",
    )
    x_root = args.output_root / "x_post_package"
    x_root.mkdir()
    (x_root / "post.txt").write_text("TDNet Sunday scorecard draft — review required before posting.\n", encoding="utf-8")
    (x_root / "alt_text.md").write_text("# Alt text\n\nWeekly TDNet margin and winner-accuracy scorecards by model.\n", encoding="utf-8")
    (x_root / "manifest.json").write_text(json.dumps({"send_status": "draft_only_requires_explicit_approval"}, indent=2) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name != "sunday_publication_manifest.json":
            files[str(path.relative_to(args.output_root))] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sunday_review_bundle_ready",
        "prediction_bundle": str(args.bundle),
        "prediction_bundle_verification": verification,
        "results_path": str(args.results),
        "results_sha256": sha256_file(args.results),
        "snapshot_completeness": snapshot,
        "files": files,
        "external_send": "disabled",
    }
    (args.output_root / "sunday_publication_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "games": int(scored["game_id"].nunique()), "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
