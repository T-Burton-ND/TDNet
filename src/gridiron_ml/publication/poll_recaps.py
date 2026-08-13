"""Repeatable weekly TDNet poll and model-disagreement publication outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd

from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid, plot_weekly_top25_table
from gridiron_ml.td_run.poll_viz import (
    abbreviate_team_name, draw_team_logo, resolve_team_logo_path,
)
from .polls import add_team_records, load_postweek_ap_top25


def build_season_poll_recaps(
    poll_tables_root: str | Path,
    output_root: str | Path,
    *,
    objective: str,
    logo_dir: str | Path | None = None,
    top_n: int = 25,
    ap_rankings_path: str | Path | None = None,
    games_path: str | Path | None = None,
    season: int = 2025,
) -> pd.DataFrame:
    """Render consensus, per-model ballots, and disagreement for every week."""
    source = Path(poll_tables_root)
    poll = pd.read_csv(source / "weekly_poll_top25.csv")
    ballots = pd.read_csv(source / "weekly_poll_ballots.csv")
    output = Path(output_root)
    games = (
        pd.read_parquet(games_path) if games_path and str(games_path).endswith(".parquet")
        else pd.read_csv(games_path) if games_path else pd.DataFrame()
    )
    summaries = []
    for week in sorted(pd.to_numeric(poll["week"], errors="coerce").dropna().astype(int).unique()):
        week_poll = poll[pd.to_numeric(poll["week"], errors="coerce").eq(week)].copy()
        week_ballots = ballots[pd.to_numeric(ballots["week"], errors="coerce").eq(week)].copy()
        ap = load_postweek_ap_top25(ap_rankings_path, season=season, completed_week=week) if ap_rankings_path else pd.DataFrame()
        if not ap.empty:
            ap_rank = ap.set_index("team")["rank"]
            week_poll["ap_rank"] = week_poll["keys_team"].map(ap_rank).astype("Int64")
            week_poll["tdnet_minus_ap"] = week_poll["rank"] - week_poll["ap_rank"]
            week_poll["ap_snapshot_week"] = int(ap["week"].dropna().iloc[0]) if "week" in ap and ap["week"].notna().any() else max(1, week)
        else:
            week_poll["ap_rank"] = pd.Series(pd.NA, index=week_poll.index, dtype="Int64")
            week_poll["tdnet_minus_ap"] = pd.Series(pd.NA, index=week_poll.index, dtype="Int64")
        if not games.empty:
            week_poll = add_team_records(week_poll, games, completed_week=week)
        receiving_votes = aggregate_receiving_votes(week_poll, week_ballots, top_n=top_n)
        directory = output / f"week_{week:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        week_poll.to_csv(directory / "tdnet_top25.csv", index=False)
        receiving_votes.to_csv(directory / "tdnet_receiving_votes.csv", index=False)
        receiving_text = format_receiving_votes(receiving_votes)
        (directory / "receiving_votes.txt").write_text(receiving_text + "\n", encoding="utf-8")
        top_ballots = week_ballots[pd.to_numeric(week_ballots["ballot_rank"], errors="coerce").between(1, top_n)].copy()
        top_ballots.to_csv(directory / "per_model_top25_long.csv", index=False)
        ballot_wide = top_ballots.pivot_table(index="ballot_rank", columns="ballot_model", values="keys_team", aggfunc="first").sort_index()
        ballot_wide.to_csv(directory / "per_model_top25.csv")

        plot_consensus_poll_table(
            week_poll, directory / "tdnet_top25.png",
            title=f"2025 Week {week}: TDNet {objective.title()}-Model Top 25",
            receiving_votes=receiving_text,
            logo_dir=logo_dir,
        )
        plot_weekly_top25_table(
            week_poll, directory / "tdnet_top25_logo_column.png", top_n=top_n, logo_dir=logo_dir
        )
        plot_ballot_logo_grid(
            top_ballots, directory / "per_model_top25.png", top_n=top_n,
            logo_dir=logo_dir, title=f"2025 Week {week}: {objective.title()}-Model Top 25 Ballots",
        )
        if not ap.empty:
            plot_tdnet_vs_ap_poll(
                week_poll.rename(columns={"keys_team": "team"}), ap,
                directory / "tdnet_vs_ap_top25.png",
                title=f"2025 Week {week}: TDNet vs. AP Top 25",
                logo_dir=logo_dir,
            )
        disagreement = model_consensus_disagreement(week_poll, week_ballots, top_n=top_n)
        disagreement.to_csv(directory / "model_consensus_disagreement.csv", index=False)
        plot_model_disagreement(
            disagreement, directory / "model_consensus_disagreement.png",
            title=f"2025 Week {week}: Models Farthest from the {objective.title()} Consensus",
        )
        summaries.extend(disagreement.assign(week=week, objective=objective).to_dict("records"))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "model_consensus_disagreement_all_weeks.csv", index=False)
    plot_full_season_poll_grid(
        poll, output / "full_season_poll_grid.png", objective=objective,
        season=season, logo_dir=logo_dir, top_n=top_n,
    )
    return summary


def plot_consensus_poll_table(
    poll: pd.DataFrame, path: str | Path, *, title: str,
    receiving_votes: str | None = None, logo_dir: str | Path | None = None,
    dpi: int = 180, reference_label: str = "AP",
) -> Path:
    frame = poll.sort_values("rank").head(25).copy()
    ap_rank = frame.get(
        "reference_rank", frame.get("ap_rank", pd.Series(pd.NA, index=frame.index))
    ).map(
        lambda value: "NR" if pd.isna(value) else str(int(value))
    )
    delta = frame.get(
        "tdnet_minus_reference", frame.get("tdnet_minus_ap", pd.Series(pd.NA, index=frame.index))
    ).map(
        lambda value: "—" if pd.isna(value) else f"{int(value):+d}"
    )
    table = pd.DataFrame({
        "Rank": frame["rank"].astype(int),
        str(reference_label): ap_rank,
        "Δ": delta,
        "Logo": ["" if resolve_team_logo_path(team, logo_dir) else str(team)
                 for team in frame["keys_team"]],
        "Record": frame.get("record", pd.Series("—", index=frame.index)).fillna("—"),
        "Points": frame["poll_points"].astype(int),
        "Top-25 votes": frame["top25_votes"].astype(int).astype(str) + "/" + frame["ballots_seen"].astype(int).astype(str),
        "Firsts": frame["first_place_votes"].astype(int),
        "Avg. rank": frame["average_rank"].map(lambda value: f"{value:.1f}"),
        "Ballot range": frame["best_rank"].astype(int).astype(str) + "–" + frame["worst_rank"].astype(int).astype(str),
    })
    fig, axis = plt.subplots(figsize=(12.5, 10.2))
    fig.patch.set_facecolor("#F7F4ED")
    axis.axis("off")
    plotted = axis.table(
        cellText=table.values, colLabels=table.columns, loc="center", cellLoc="left", colLoc="left",
        bbox=[0.0, 0.085, 1.0, 0.82],
        colWidths=[0.05, 0.05, 0.05, 0.225, 0.075, 0.08, 0.13, 0.07, 0.09, 0.105],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(9.2)
    plotted.scale(1, 1.35)
    for (row, _), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor("#22324A")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
            if row <= 4:
                cell.set_text_props(weight="bold")
    fig.canvas.draw()
    team_column = table.columns.get_loc("Logo")
    for row_number, team in enumerate(frame["keys_team"].astype(str), start=1):
        logo = resolve_team_logo_path(team, logo_dir)
        if logo is None:
            continue
        cell = plotted[(row_number, team_column)]
        draw_team_logo(axis, logo, cell.get_x() + cell.get_width()/2,
                       cell.get_y() + cell.get_height()/2, target_px=20)
    axis.set_title(title, fontsize=17, weight="bold", pad=18, color="#17263C")
    footer = (
        f"Δ = TDNet rank − {reference_label} rank; negative values mean TDNet ranks the team higher."
    )
    if receiving_votes:
        footer += "\n" + textwrap.fill("Receiving votes: " + receiving_votes, width=145)
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=8.5, color="#555B63")
    fig.tight_layout(rect=[0, 0.055, 1, 0.985])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def aggregate_receiving_votes(poll: pd.DataFrame, ballots: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    """Aggregate positive ballot points for teams outside the consensus Top 25."""
    top_teams = set(poll.sort_values("rank").head(top_n)["keys_team"].astype(str))
    frame = ballots.copy()
    frame["poll_points"] = pd.to_numeric(frame["poll_points"], errors="coerce").fillna(0)
    outside = frame[~frame["keys_team"].astype(str).isin(top_teams)]
    result = outside.groupby("keys_team", as_index=False).agg(
        poll_points=("poll_points", "sum"),
        top25_votes=("top25_vote", "sum"),
        ballots_seen=("ballot_model", "nunique"),
        best_ballot_rank=("ballot_rank", "min"),
    )
    result = result[result["poll_points"].gt(0)].sort_values(
        ["poll_points", "top25_votes", "keys_team"], ascending=[False, False, True]
    ).reset_index(drop=True)
    return result


def format_receiving_votes(receiving_votes: pd.DataFrame) -> str:
    if receiving_votes.empty:
        return "None"
    return ", ".join(
        f"{row.keys_team} ({int(row.poll_points)})" for row in receiving_votes.itertuples()
    )


def plot_full_season_poll_grid(
    poll: pd.DataFrame, path: str | Path, *, objective: str, season: int,
    logo_dir: str | Path | None = None, top_n: int = 25, dpi: int = 180,
) -> Path:
    """Render every weekly Top 25 using the classic logo-column poll grid."""
    return plot_weekly_top25_table(poll, path, top_n=top_n, logo_dir=logo_dir)


def season_poll_podium_summary(poll: pd.DataFrame) -> pd.DataFrame:
    """Return the weekly top three and first-to-second point gap."""
    frame = poll.copy()
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["poll_points"] = pd.to_numeric(frame["poll_points"], errors="coerce")
    rows = []
    for week, week_poll in frame.dropna(subset=["week"]).groupby("week", sort=True):
        top = week_poll.sort_values(["rank", "keys_team"], kind="stable").head(3)
        if len(top) < 3:
            continue
        first, second, third = list(top.itertuples(index=False))
        rows.append({
            "week": int(week),
            "leader": str(first.keys_team),
            "leader_points": int(first.poll_points),
            "runner_up": str(second.keys_team),
            "runner_up_points": int(second.poll_points),
            "third_place": str(third.keys_team),
            "third_place_points": int(third.poll_points),
            "leader_gap": int(first.poll_points - second.poll_points),
            "ballots_seen": int(first.ballots_seen),
        })
    return pd.DataFrame(rows).sort_values("week", kind="stable").reset_index(drop=True)


def plot_season_poll_race(
    poll: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    objective: str,
    top_teams: int = 5,
    dpi: int = 180,
) -> Path:
    """Plot final-leading teams across the season and shade the weekly lead."""
    frame = poll.copy()
    for column in ("week", "rank", "poll_points"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["week", "rank", "poll_points", "keys_team"])
    final_week = int(frame["week"].max())
    selected = (
        frame[frame["week"].eq(final_week)]
        .sort_values(["rank", "keys_team"], kind="stable")
        .head(top_teams)["keys_team"]
        .astype(str)
        .tolist()
    )
    weeks = sorted(frame["week"].astype(int).unique())
    top_two = frame[frame["rank"].between(1, 2)].pivot(
        index="week", columns="rank", values="poll_points"
    ).reindex(weeks)

    fig, axis = plt.subplots(figsize=(14.8, 7.8))
    fig.patch.set_facecolor("#F7F4ED")
    axis.set_facecolor("#FCFBF7")
    axis.fill_between(
        weeks, top_two[2].to_numpy(), top_two[1].to_numpy(),
        color="#E8B84A", alpha=0.22, label="Weekly first-to-second gap", zorder=1,
    )
    colors = ["#A6192E", "#1E5A86", "#2F7D4A", "#8A5A9B", "#D07A2D"]
    for color, team in zip(colors, selected):
        team_frame = frame[frame["keys_team"].astype(str).eq(team)].sort_values("week")
        axis.plot(
            team_frame["week"], team_frame["poll_points"], marker="o", markersize=4.8,
            linewidth=2.8 if team == selected[0] else 2.0, color=color, label=team, zorder=3,
        )
    for week in weeks:
        first = float(top_two.loc[week, 1])
        second = float(top_two.loc[week, 2])
        axis.text(
            week, first + 7, f"+{int(first - second)}", ha="center", va="bottom",
            fontsize=7.5, weight="bold", color="#7A5711", zorder=4,
        )
    axis.set_xlim(min(weeks) - 0.35, max(weeks) + 0.35)
    axis.set_ylim(0, max(850, float(frame["poll_points"].max()) + 70))
    axis.set_xticks(weeks, ["Pre" if week == 0 else str(week) for week in weeks])
    axis.set_xlabel("2025 poll week")
    axis.set_ylabel("TDNet consensus poll points")
    axis.set_title(
        f"{season} TDNet Poll Race — {objective.title()} Roster",
        fontsize=18, weight="bold", pad=18, color="#17263C",
    )
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.19))
    ballots = int(pd.to_numeric(frame.get("ballots_seen"), errors="coerce").max())
    fig.text(
        0.5, 0.012,
        f"Corrected-F6 wide margin roster · {ballots} poll-enabled models · Gold band and +labels show the weekly leader's margin over second place.",
        ha="center", va="bottom", fontsize=9, color="#555B63",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_season_podium_gaps(
    poll: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    objective: str,
    logo_dir: str | Path | None = None,
    dpi: int = 180,
) -> Path:
    """Render a common-scale weekly podium for the full season."""
    summary = season_poll_podium_summary(poll)
    if summary.empty:
        raise ValueError("poll must contain at least three ranked teams per week")
    point_ceiling = int(summary["ballots_seen"].max()) * 25
    fig, axes = plt.subplots(3, 6, figsize=(18, 10.8), sharey=True)
    fig.patch.set_facecolor("#F7F4ED")
    podium_colors = ["#B7BDC6", "#D8AD3D", "#A96E43"]
    for axis, row in zip(axes.flat, summary.itertuples(index=False)):
        teams = [row.runner_up, row.leader, row.third_place]
        points = [row.runner_up_points, row.leader_points, row.third_place_points]
        ranks = ["2nd", "1st", "3rd"]
        axis.set_facecolor("#FCFBF7")
        bars = axis.bar([0, 1, 2], points, width=0.72, color=podium_colors, edgecolor="#7D858F", linewidth=0.8)
        for x, bar, team, value, rank in zip([0, 1, 2], bars, teams, points, ranks):
            axis.text(x, max(32, value * 0.48), f"{rank}\n{value}", ha="center", va="center", fontsize=8, weight="bold", color="#17263C")
            logo = resolve_team_logo_path(team, logo_dir)
            if logo is not None:
                draw_team_logo(axis, logo, x, value + 34, target_px=25)
            else:
                axis.text(x, value + 34, abbreviate_team_name(team, 10), ha="center", va="center", fontsize=7.5)
        axis.set_xticks([0, 1, 2], [abbreviate_team_name(team, 11) for team in teams], fontsize=7.5)
        axis.set_ylim(0, point_ceiling + 105)
        week_label = "Preseason" if row.week == 0 else f"Week {row.week}"
        axis.set_title(f"{week_label}\nLead: +{row.leader_gap}", fontsize=10.5, weight="bold", pad=5, color="#17263C")
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#AEB4BC")
    extra_axes = list(axes.flat[len(summary):])
    if extra_axes:
        axis = extra_axes.pop(0)
        final = summary.iloc[-1]
        peak = summary.loc[summary["leader_gap"].idxmax()]
        weeks_led = int(summary["leader"].eq(final["leader"]).sum())
        axis.set_facecolor("#FCFBF7")
        axis.set_xlim(0, 1)
        axis.axis("off")
        logo = resolve_team_logo_path(str(final["leader"]), logo_dir)
        if logo is not None:
            draw_team_logo(axis, logo, 0.5, 0.76 * (point_ceiling + 105), target_px=48)
        axis.text(0.5, 0.93, "Season snapshot", transform=axis.transAxes, ha="center", va="top", fontsize=12, weight="bold", color="#17263C")
        axis.text(0.5, 0.56, f"{final['leader']} finished No. 1", transform=axis.transAxes, ha="center", fontsize=10, weight="bold", color="#17263C")
        axis.text(0.5, 0.43, f"{int(final['leader_points'])} points · final lead +{int(final['leader_gap'])}", transform=axis.transAxes, ha="center", fontsize=9, color="#17263C")
        axis.text(0.5, 0.31, f"Led {weeks_led} of {len(summary)} polls", transform=axis.transAxes, ha="center", fontsize=9, color="#17263C")
        axis.text(0.5, 0.19, f"Largest lead: +{int(peak['leader_gap'])} in Week {int(peak['week'])}", transform=axis.transAxes, ha="center", fontsize=9, color="#17263C")
    for axis in extra_axes:
        axis.axis("off")
    fig.suptitle(
        f"{season} Weekly Poll Podiums — {objective.title()} Roster",
        fontsize=19, weight="bold", color="#17263C", y=0.985,
    )
    fig.text(
        0.5, 0.017,
        f"All podiums share a 0–{point_ceiling} point scale ({int(summary['ballots_seen'].max())} ballots × 25 points). Lead is first-place points minus second-place points.",
        ha="center", va="bottom", fontsize=9, color="#555B63",
    )
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.07, top=0.88, hspace=0.36, wspace=0.05)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def model_consensus_disagreement(poll: pd.DataFrame, ballots: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    """Rank ballots by rank distance and Top-25 membership disagreement."""
    consensus = poll.sort_values("rank").head(top_n).set_index("keys_team")["rank"]
    consensus_teams = set(consensus.index.astype(str))
    rows = []
    for model, ballot in ballots.groupby("ballot_model", sort=True):
        ranks = ballot.set_index("keys_team")["ballot_rank"].apply(pd.to_numeric, errors="coerce")
        aligned = pd.DataFrame({"consensus_rank": consensus, "model_rank": ranks.reindex(consensus.index)})
        aligned["model_rank"] = aligned["model_rank"].fillna(top_n + 1)
        aligned["absolute_rank_delta"] = (aligned["model_rank"] - aligned["consensus_rank"]).abs()
        model_top = set(ballot[pd.to_numeric(ballot["ballot_rank"], errors="coerce").between(1, top_n)]["keys_team"].astype(str))
        worst_team = aligned["absolute_rank_delta"].idxmax()
        rows.append({
            "ballot_model": model,
            "mean_absolute_rank_delta": float(aligned["absolute_rank_delta"].mean()),
            "median_absolute_rank_delta": float(aligned["absolute_rank_delta"].median()),
            "top25_overlap": int(len(consensus_teams & model_top)),
            "top25_membership_disagreements": int(top_n - len(consensus_teams & model_top)),
            "largest_rank_outlier_team": worst_team,
            "largest_rank_outlier_delta": float(aligned.loc[worst_team, "absolute_rank_delta"]),
            "rank_correlation": float(aligned[["consensus_rank", "model_rank"]].corr(method="spearman").iloc[0, 1]),
        })
    return pd.DataFrame(rows).sort_values(
        ["mean_absolute_rank_delta", "top25_membership_disagreements"], ascending=False
    ).reset_index(drop=True)


def plot_model_disagreement(disagreement: pd.DataFrame, path: str | Path, *, title: str, dpi: int = 180) -> Path:
    frame = disagreement.sort_values("mean_absolute_rank_delta", ascending=True).copy()
    labels = frame["ballot_model"].astype(str).str.replace(r"^(winner|margin|balanced)_", "", regex=True)
    fig_height = max(5.0, 0.38 * len(frame) + 1.8)
    fig, axis = plt.subplots(figsize=(11, fig_height))
    colors = ["#9F3A38" if value >= 4 else "#D19A3D" if value >= 2.5 else "#3E7C59" for value in frame["mean_absolute_rank_delta"]]
    bars = axis.barh(labels, frame["mean_absolute_rank_delta"], color=colors)
    for bar, (_, row) in zip(bars, frame.iterrows()):
        axis.text(
            bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
            f"{row['mean_absolute_rank_delta']:.1f} ranks • {int(row['top25_overlap'])}/25 overlap",
            va="center", fontsize=8.5,
        )
    axis.set_xlabel("Mean absolute rank difference from consensus (consensus Top 25)")
    axis.set_title(title, fontsize=15, weight="bold", pad=14)
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tdnet_vs_ap_poll(
    tdnet: pd.DataFrame, ap: pd.DataFrame, path: str | Path, *, title: str,
    logo_dir: str | Path | None = None, dpi: int = 180,
    model_set_sha256: str | None = None, checkpoint_count: int | None = None,
    generated_at_utc: str | None = None,
    reference_label: str = "AP",
) -> Path:
    """Render a rank dumbbell chart without exporting the source AP rows."""
    left = tdnet.rename(columns={"team": "team", "rank": "tdnet_rank"})[["team", "tdnet_rank"]]
    right = ap.rename(columns={"rank": "ap_rank"})[["team", "ap_rank"]]
    frame = left.merge(right, on="team", how="outer")
    frame["sort_rank"] = frame[["tdnet_rank", "ap_rank"]].min(axis=1)
    frame = frame.sort_values(["sort_rank", "team"]).head(35).reset_index(drop=True)
    y = range(len(frame))
    fig, axis = plt.subplots(figsize=(11, max(7, 0.35 * len(frame) + 1.8)))
    for yi, row in zip(y, frame.itertuples()):
        if pd.notna(row.tdnet_rank) and pd.notna(row.ap_rank):
            axis.plot([row.tdnet_rank, row.ap_rank], [yi, yi], color="#BBC2C9", lw=1.4, zorder=1)
    axis.scatter(frame["tdnet_rank"], list(y), color="#274C77", s=45, label="TDNet", zorder=2)
    axis.scatter(
        frame["ap_rank"], list(y), color="#A44A3F", s=45, marker="s",
        label=reference_label, zorder=2,
    )
    axis.set_yticks(list(y), ["" for _ in y])
    for yi, team in zip(y, frame["team"].astype(str)):
        logo = resolve_team_logo_path(team, logo_dir)
        if logo is not None:
            draw_team_logo(axis, logo, 25.35, yi, target_px=18)
        else:
            axis.text(25.35, yi, abbreviate_team_name(team, 12), ha="center", va="center", fontsize=6)
    axis.invert_yaxis()
    axis.invert_xaxis()
    axis.set_xlim(26, 0)
    axis.set_xticks(range(1, 26, 2))
    axis.set_xlabel("Rank (1 is best)")
    axis.set_title(title, fontsize=16, weight="bold", pad=14)
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    if model_set_sha256:
        count = f" · {int(checkpoint_count)} checkpoints" if checkpoint_count else ""
        generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
        fig.text(0.5, 0.014, f"Frozen model-set SHA-256: {model_set_sha256}{count} · Generated UTC: {generated}", ha="center", va="bottom", fontsize=9, weight="bold", color="#28323C", family="monospace", bbox={"facecolor": "#FFFFFF", "edgecolor": "#AAB5C1", "boxstyle": "round,pad=0.35"})
    fig.tight_layout(rect=[0, 0.052 if model_set_sha256 else 0, 1, 1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path
