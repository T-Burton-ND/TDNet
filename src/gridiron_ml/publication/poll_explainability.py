"""Public, descriptive explainability graphics for frozen TDNet polls.

These charts deliberately do not claim to explain AP voters or causally
decompose a rank gap.  They make the frozen-model ballot dispersion visible
and summarize measurable fingerprint differences from AP-rank peer teams.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure_theme import TDNET_COLORS


TDNET = TDNET_COLORS["signal_orange"]
AP = TDNET_COLORS["ion_blue"]
NAVY = TDNET_COLORS["midnight_gridiron"]
PINK = TDNET_COLORS["edge_pink"]
SLATE = TDNET_COLORS["slate"]
MIST = TDNET_COLORS["polar_mist"]
GRAY = TDNET_COLORS["medium_gray"]


def plot_top25_consensus_spread(
    poll: pd.DataFrame,
    ballots: pd.DataFrame,
    path: str | Path,
    *,
    reference_poll: pd.DataFrame | None = None,
    reference_label: str = "AP",
    title: str = "TDNet Top 25: Model Ballot Spread",
    display_max_rank: int = 40,
    dpi: int = 200,
) -> Path:
    """Plot compact ballot intervals for each consensus Top-25 team.

    The underlying matrix retains every ballot. This companion view caps the
    rank axis so a handful of extreme ballots cannot obscure the ensemble's
    central spread; a right-edge arrow reports the hidden count.
    """
    top = poll.sort_values("rank").head(25).copy()
    top["keys_team"] = top["keys_team"].astype(str)
    ap_rank: dict[str, float] = {}
    if reference_poll is not None and not reference_poll.empty:
        team_col = "team" if "team" in reference_poll else "keys_team"
        ap_rank = dict(zip(reference_poll[team_col].astype(str), pd.to_numeric(reference_poll["rank"], errors="coerce")))
    ranks = ballots.copy()
    ranks["ballot_rank"] = pd.to_numeric(ranks["ballot_rank"], errors="coerce")
    cap = max(25, int(display_max_rank))
    fig, axis = plt.subplots(figsize=(15.0, 13.2))
    for y, row in enumerate(top.itertuples(index=False)):
        values = ranks.loc[ranks["keys_team"].astype(str).eq(row.keys_team), "ballot_rank"].dropna().to_numpy()
        if len(values):
            shown = np.clip(values, 1, cap)
            low, high = float(shown.min()), float(shown.max())
            q25, median, q75 = np.percentile(shown, [25, 50, 75])
            axis.hlines(y, low, high, color=GRAY, lw=1.1, zorder=1)
            axis.hlines(y, q25, q75, color=SLATE, lw=7.2, alpha=.42, zorder=2)
            axis.scatter([median], [y], s=27, color=NAVY, edgecolor="white", linewidth=.55, zorder=4)
            hidden = int((values > cap).sum())
            if hidden:
                axis.annotate(f"→ {hidden}", xy=(cap, y), xytext=(4, 0), textcoords="offset points", va="center", ha="left", fontsize=9.5, color=SLATE, clip_on=False)
        axis.vlines(float(row.rank), y - .26, y + .26, color=TDNET, lw=3.0, zorder=5)
        if row.keys_team in ap_rank and pd.notna(ap_rank[row.keys_team]):
            axis.scatter([ap_rank[row.keys_team]], [y], marker="D", s=35, color=AP, edgecolor="white", linewidth=.6, zorder=5)
    axis.set_yticks(range(len(top)), [f"{int(r.rank):>2}. {r.keys_team}" for r in top.itertuples(index=False)], fontsize=11.5)
    axis.set_xlim(.5, cap + .9)
    axis.set_xticks(range(1, cap + 1, 5))
    axis.set_xlabel(f"Model ballot rank (1 is best; values above {cap} are capped and counted)")
    axis.set_title(title, fontsize=21, weight="bold", loc="left", pad=20)
    axis.text(0, 1.01, f"Thin whisker: displayed ballot range   |   thick bar: middle 50%   |   dot: median   |   orange tick: TDNet   |   blue diamond: {reference_label}", transform=axis.transAxes, fontsize=11, color=SLATE)
    axis.grid(axis="x", color=MIST, lw=.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.invert_yaxis()
    fig.tight_layout()
    return _save(fig, path, dpi)


def build_ap_peer_signal_proxy(
    poll: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    *,
    reference_poll: pd.DataFrame | None = None,
    top_n: int = 3,
    peer_window: int = 3,
    prioritize_tdnet_top_n: int = 10,
) -> pd.DataFrame:
    """Return the largest grouped fingerprint differences from AP-rank peers.

    This is a transparent descriptive proxy, not SHAP: for each largest
    rank-gap team, group-wise standardized values are compared with teams
    whose AP rank lies within ``peer_window``.  It intentionally has no claim
    about what *caused* an AP/TDNet disagreement.
    """
    if "ap_rank" not in poll and "reference_rank" in poll:
        poll = poll.rename(columns={"reference_rank": "ap_rank"})
    required = {"keys_team", "ap_rank", "rank"}
    if not required.issubset(poll):
        return pd.DataFrame()
    candidates = poll.dropna(subset=["ap_rank"]).copy()
    candidates["gap"] = pd.to_numeric(candidates["rank"], errors="coerce") - pd.to_numeric(candidates["ap_rank"], errors="coerce")
    candidates["absolute_gap"] = candidates["gap"].abs()
    candidates = candidates.sort_values(["absolute_gap", "rank", "keys_team"], ascending=[False, True, True], kind="stable")
    # Preserve one high-interest disagreement from TDNet's own Top 10 before
    # filling the remaining panels from the largest gaps overall.
    priority = candidates.loc[pd.to_numeric(candidates["rank"], errors="coerce").le(prioritize_tdnet_top_n)].head(1)
    selected = pd.concat([priority, candidates], ignore_index=False)
    candidates = selected.drop_duplicates("keys_team", keep="first").head(top_n).drop(columns="absolute_gap")
    if candidates.empty or "keys_team" not in feature_frame:
        return pd.DataFrame()
    meta = feature_metadata.copy()
    if not {"feature", "family"}.issubset(meta):
        return pd.DataFrame()
    usable = [c for c in meta["feature"].astype(str) if c in feature_frame and pd.api.types.is_numeric_dtype(feature_frame[c])]
    if not usable:
        return pd.DataFrame()
    state = feature_frame.drop_duplicates("keys_team", keep="last").set_index("keys_team")
    standardized = state[usable].apply(pd.to_numeric, errors="coerce")
    standardized = (standardized - standardized.median()) / standardized.std(ddof=0).replace(0, np.nan)
    families = meta.set_index("feature").loc[usable, "family"].astype(str)
    if reference_poll is not None and not reference_poll.empty:
        reference_team_col = "team" if "team" in reference_poll else "keys_team"
        team_to_ap = dict(zip(reference_poll[reference_team_col].astype(str), pd.to_numeric(reference_poll["rank"], errors="coerce")))
    else:
        team_to_ap = candidates.set_index("keys_team")["ap_rank"].to_dict()
    rows = []
    for team_row in candidates.itertuples(index=False):
        if team_row.keys_team not in standardized.index:
            continue
        peer_teams = [name for name, ap in team_to_ap.items() if name != team_row.keys_team and abs(float(ap) - float(team_row.ap_rank)) <= peer_window]
        # The top-25 comparison alone is too thin; use the AP-ranked candidate
        # set when possible, otherwise compare to the national fingerprint median.
        peer = standardized.reindex(peer_teams).mean() if peer_teams else pd.Series(0.0, index=usable)
        difference = standardized.loc[team_row.keys_team] - peer
        grouped = difference.groupby(families).mean().dropna()
        for family, value in grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(3).items():
            rows.append({"team": team_row.keys_team, "tdnet_rank": int(team_row.rank), "ap_rank": int(team_row.ap_rank), "gap": int(team_row.gap), "family": _friendly_family(family), "signal_z": float(value)})
    return pd.DataFrame(rows)


def plot_top25_discrepancy_features(signals: pd.DataFrame, path: str | Path, *, title: str = "Why TDNet Differs: Top Fingerprint Signals", dpi: int = 200) -> Path | None:
    """Plot the descriptive AP-peer fingerprint proxy, one panel per team."""
    if signals.empty:
        return None
    teams = list(dict.fromkeys(signals["team"].astype(str)))
    fig, axes = plt.subplots(len(teams), 1, figsize=(15.0, max(8.0, 3.2 * len(teams))), squeeze=False)
    fig.suptitle(title, fontsize=21, weight="bold", x=.07, ha="left", y=.985)
    for axis, team in zip(axes[:, 0], teams):
        frame = signals.loc[signals["team"].astype(str).eq(team)].copy().sort_values("signal_z")
        colors = [NAVY if value >= 0 else PINK for value in frame["signal_z"]]
        axis.barh(frame["family"], frame["signal_z"], color=colors, height=.58)
        axis.axvline(0, color=SLATE, lw=.8)
        row = frame.iloc[0]
        direction = "below" if row.gap > 0 else "above"
        axis.set_title(f"{team}  •  TDNet #{int(row.tdnet_rank)} vs AP #{int(row.ap_rank)} ({abs(int(row.gap))} places {direction} AP)", loc="left", fontsize=14.5, weight="bold", pad=7)
        axis.set_xlabel("Standardized fingerprint difference from AP-rank peers")
        axis.grid(axis="x", color=MIST, lw=.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.07, .012, "Midnight gridiron = signal associated with TDNet ranking the team higher than AP peers; edge pink = lower. Descriptive AP-peer proxy only: it does not explain voters, establish causation, or infer unencoded context such as coaching changes.", fontsize=10.3, color=SLATE)
    fig.tight_layout(rect=[0, .045, 1, .95])
    return _save(fig, path, dpi)


def _friendly_family(value: str) -> str:
    labels = {"roster": "Returning production / roster", "talent": "Roster talent", "coaching": "Coaching history", "efficiency": "Play efficiency", "box_score": "Prior-year team production", "box_score_offense": "Prior-year offensive production", "box_score_defense": "Prior-year defensive production", "box_score_general": "Prior-year team production", "box_score_special_teams": "Prior-year special teams", "preseason_prior": "Preseason prior", "opponent_adjusted": "Opponent-adjusted strength", "temporal": "Recent trajectory", "schedule_graph": "Schedule / network context", "sample_size": "Prior sample context", "situational": "Situational context"}
    return labels.get(value, value.replace("_", " ").title())


def _save(fig: plt.Figure, path: str | Path, dpi: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target
