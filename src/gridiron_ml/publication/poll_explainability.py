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


TDNET = "#E4572E"
AP = "#3569A8"
DOT = "#718096"
HIGH = "#167C5A"
LOW = "#B5473C"


def plot_top25_consensus_spread(
    poll: pd.DataFrame,
    ballots: pd.DataFrame,
    path: str | Path,
    *,
    reference_poll: pd.DataFrame | None = None,
    reference_label: str = "AP",
    title: str = "TDNet Top 25: Model Ballot Spread",
    dpi: int = 200,
) -> Path:
    """Plot every frozen-model rank for each consensus Top-25 team."""
    top = poll.sort_values("rank").head(25).copy()
    top["keys_team"] = top["keys_team"].astype(str)
    ap_rank: dict[str, float] = {}
    if reference_poll is not None and not reference_poll.empty:
        team_col = "team" if "team" in reference_poll else "keys_team"
        ap_rank = dict(zip(reference_poll[team_col].astype(str), pd.to_numeric(reference_poll["rank"], errors="coerce")))
    ranks = ballots.copy()
    ranks["ballot_rank"] = pd.to_numeric(ranks["ballot_rank"], errors="coerce")
    max_rank = max(25, int(ranks["ballot_rank"].max()))
    fig, axis = plt.subplots(figsize=(12.8, 10.8))
    rng = np.random.default_rng(2026)
    for y, row in enumerate(top.itertuples(index=False)):
        values = ranks.loc[ranks["keys_team"].astype(str).eq(row.keys_team), "ballot_rank"].dropna().to_numpy()
        if len(values):
            jitter = rng.uniform(-0.15, 0.15, size=len(values))
            axis.scatter(values, y + jitter, s=13, color=DOT, alpha=.45, linewidths=0, zorder=2)
        axis.vlines(float(row.rank), y - .34, y + .34, color=TDNET, lw=3, zorder=4)
        if row.keys_team in ap_rank and pd.notna(ap_rank[row.keys_team]):
            axis.scatter([ap_rank[row.keys_team]], [y], marker="D", s=35, color=AP, edgecolor="white", linewidth=.6, zorder=5)
    axis.set_yticks(range(len(top)), [f"{int(r.rank):>2}. {r.keys_team}" for r in top.itertuples(index=False)], fontsize=9)
    axis.set_xlim(max_rank + 1, 0)
    axis.set_xticks(range(1, max_rank + 1, 5))
    axis.set_xlabel("Rank (1 is best; dots can extend below the Top 25)")
    axis.set_title(title, fontsize=17, weight="bold", loc="left", pad=16)
    axis.text(0, 1.01, f"Dots: individual frozen-model ballots   |   orange line: TDNet consensus   |   blue diamond: {reference_label} rank", transform=axis.transAxes, fontsize=9.5, color="#44546A")
    axis.grid(axis="x", color="#D9E1E8", lw=.7)
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
    candidates = candidates.reindex(candidates["gap"].abs().sort_values(ascending=False).index).head(top_n)
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
    fig, axes = plt.subplots(len(teams), 1, figsize=(12.8, max(6.2, 2.45 * len(teams))), squeeze=False)
    fig.suptitle(title, fontsize=17, weight="bold", x=.07, ha="left", y=.985)
    for axis, team in zip(axes[:, 0], teams):
        frame = signals.loc[signals["team"].astype(str).eq(team)].copy().sort_values("signal_z")
        colors = [HIGH if value >= 0 else LOW for value in frame["signal_z"]]
        axis.barh(frame["family"], frame["signal_z"], color=colors, height=.58)
        axis.axvline(0, color="#607080", lw=.8)
        row = frame.iloc[0]
        direction = "below" if row.gap > 0 else "above"
        axis.set_title(f"{team}  •  TDNet #{int(row.tdnet_rank)} vs AP #{int(row.ap_rank)} ({abs(int(row.gap))} places {direction} AP)", loc="left", fontsize=11.5, weight="bold", pad=5)
        axis.set_xlabel("Standardized fingerprint difference from AP-rank peers")
        axis.grid(axis="x", color="#D9E1E8", lw=.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.07, .012, "Descriptive proxy: grouped measurable fingerprint values compared with AP-rank peers. It does not explain AP voters, establish causation, or infer unencoded context such as coaching changes.", fontsize=8.3, color="#44546A")
    fig.tight_layout(rect=[0, .045, 1, .95])
    return _save(fig, path, dpi)


def _friendly_family(value: str) -> str:
    labels = {"roster": "Returning production / roster", "talent": "Roster talent", "coaching": "Coaching history", "efficiency": "Play efficiency", "box_score": "Prior-year team production", "box_score_offense": "Prior-year offensive production", "box_score_defense": "Prior-year defensive production", "box_score_general": "Prior-year team production", "box_score_special_teams": "Prior-year special teams", "preseason_prior": "Preseason prior", "opponent_adjusted": "Opponent-adjusted strength", "temporal": "Recent trajectory", "schedule_graph": "Schedule / network context", "sample_size": "Prior sample context", "situational": "Situational context"}
    return labels.get(value, value.replace("_", " ").title())


def _save(fig: plt.Figure, path: str | Path, dpi: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return target
