"""Materialize auditable Week-0 team states for preseason prediction."""

from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.pipeline.contracts.features import is_feature_column
from .bundles import sha256_file


def build_preseason_state_frame(
    frame: pd.DataFrame,
    *,
    season: int,
    project_root: str | Path | None = None,
    cold_start_policy: str = "conference_mean",
    k_neighbors: int = 5,
    donor_week: int = 4,
    shrinkage: float = 0.20,
) -> pd.DataFrame:
    """Carry each team's latest prior state into its target-season Week-0 row."""
    required = {"keys_season", "keys_week", "keys_team"}
    if not required.issubset(frame):
        raise ValueError(f"Fingerprint frame lacks {sorted(required - set(frame))}")
    out = frame.copy()
    seasons = pd.to_numeric(out["keys_season"], errors="coerce")
    weeks = pd.to_numeric(out["keys_week"], errors="coerce")
    target = seasons.eq(int(season)) & weeks.eq(0)
    if not target.any():
        raise ValueError(f"No Week-0 rows exist for season {season}.")
    prior = out.loc[seasons.lt(int(season))].copy()
    prior["__season"] = pd.to_numeric(prior["keys_season"], errors="coerce")
    prior["__week"] = pd.to_numeric(prior["keys_week"], errors="coerce")
    prior = prior.sort_values(["keys_team", "__season", "__week"]).drop_duplicates(
        "keys_team", keep="last"
    ).set_index("keys_team")
    feature_columns = [
        column
        for column in out.columns
        if is_feature_column(column)
        and not str(column).startswith(("next_", "keys_", "fp_", "y_", "market_"))
    ]
    target_rows = out.loc[target].copy()
    current_preseason = target_rows.copy()
    target_rows["preseason_prior_source_season"] = pd.NA
    target_rows["preseason_prior_source_week"] = pd.NA
    target_rows["preseason_prior_applied"] = False
    target_rows["preseason_prior_method"] = "missing"
    target_rows["preseason_prior_neighbors"] = ""
    for index in target_rows.index:
        team = target_rows.at[index, "keys_team"]
        if team not in prior.index:
            continue
        target_rows.loc[index, feature_columns] = prior.loc[team, feature_columns].to_numpy()
        target_rows.at[index, "preseason_prior_source_season"] = int(prior.at[team, "__season"])
        target_rows.at[index, "preseason_prior_source_week"] = int(prior.at[team, "__week"])
        target_rows.at[index, "preseason_prior_applied"] = True
        target_rows.at[index, "preseason_prior_method"] = "same_team_carry_forward"
    missing = ~target_rows["preseason_prior_applied"]
    if missing.any() and project_root is not None:
        if cold_start_policy == "conference_mean":
            target_rows = _apply_conference_mean(
                target_rows=target_rows, missing=missing, season=int(season),
                project_root=Path(project_root), feature_columns=feature_columns,
            )
        elif cold_start_policy == "first_time_fbs_knn":
            target_rows = _apply_newcomer_knn(
                frame=out, target_rows=target_rows, missing=missing, season=int(season),
                project_root=Path(project_root), feature_columns=feature_columns,
                k_neighbors=k_neighbors, donor_week=donor_week, shrinkage=shrinkage,
            )
        else:
            raise ValueError(f"Unknown cold_start_policy={cold_start_policy!r}.")
    overlay_columns = _allowed_preseason_columns(
        target_rows.columns, project_root=Path(project_root) if project_root is not None else None
    )
    target_rows["preseason_current_overlay_count"] = 0
    for column in overlay_columns:
        available = current_preseason[column].notna()
        target_rows.loc[available, column] = current_preseason.loc[available, column]
        target_rows.loc[available, "preseason_current_overlay_count"] += 1
    if project_root is not None:
        target_rows = _overlay_raw_preseason_context(
            target_rows, season=int(season), project_root=Path(project_root)
        )
    return target_rows.reset_index(drop=True)


def _overlay_raw_preseason_context(
    target_rows: pd.DataFrame, *, season: int, project_root: Path
) -> pd.DataFrame:
    """Overlay current-season CFBD context that may postdate the frozen fingerprint."""
    out = target_rows.copy()
    raw_root = project_root / "data/raw/cfbd/v2"
    mappings = {
        "talent": {"talent": "roster_talent"},
        "returning": {
            "total_rushing_p_p_a": "roster_return_total_rushing_p_p_a",
            "percent_p_p_a": "roster_return_percent_p_p_a",
            "percent_passing_p_p_a": "roster_return_percent_passing_p_p_a",
            "percent_receiving_p_p_a": "roster_return_percent_receiving_p_p_a",
            "percent_rushing_p_p_a": "roster_return_percent_rushing_p_p_a",
            "rushing_usage": "roster_return_rushing_usage",
        },
    }
    team_key = out["keys_team"].astype(str)
    for endpoint, mapping in mappings.items():
        path = raw_root / endpoint / f"{season}.parquet"
        if not path.exists():
            continue
        raw = pd.read_parquet(path)
        if raw.empty or "team" not in raw:
            continue
        raw = raw.drop_duplicates("team", keep="last").set_index(raw["team"].astype(str))
        for source, target in mapping.items():
            if source not in raw or target not in out:
                continue
            values = team_key.map(raw[source])
            available = values.notna()
            out.loc[available, target] = values.loc[available].to_numpy()
            if "preseason_current_overlay_count" in out:
                out.loc[available, "preseason_current_overlay_count"] += 1
    return out


def _allowed_preseason_columns(columns, *, project_root: Path | None) -> list[str]:
    """Resolve current-season values that may overlay the carried Week-0 state."""
    if project_root is None:
        return []
    registry_path = project_root / "configs/features/feature_registry.yaml"
    if not registry_path.exists():
        return []
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    exact = {
        str(name)
        for name, metadata in dict(registry.get("features", {})).items()
        if bool((metadata or {}).get("allowed_preseason", False))
    }
    patterns = [
        str(pattern)
        for pattern, metadata in dict(registry.get("patterns", {})).items()
        if bool((metadata or {}).get("allowed_preseason", False))
    ]
    return [
        str(column)
        for column in columns
        if str(column) in exact or any(fnmatchcase(str(column), pattern) for pattern in patterns)
    ]


def _apply_conference_mean(*, target_rows, missing, season, project_root, feature_columns):
    """Use the owner-approved mean of carried-forward peers in the new FBS conference."""
    membership_path = project_root / f"data/raw/cfbd/v2/teams_fbs/{season}.parquet"
    if not membership_path.exists():
        return target_rows
    membership = pd.read_parquet(membership_path)
    conferences = dict(zip(membership["school"].astype(str), membership["conference"].astype(str)))
    peer_conference = target_rows["keys_team"].astype(str).map(conferences)
    for index in target_rows.index[missing]:
        team = str(target_rows.at[index, "keys_team"])
        conference = conferences.get(team)
        if not conference:
            continue
        peers = target_rows.loc[
            target_rows["preseason_prior_applied"]
            & peer_conference.eq(conference)
            & target_rows["keys_team"].astype(str).ne(team)
        ]
        if peers.empty:
            continue
        prior_values = peers[feature_columns].apply(pd.to_numeric, errors="coerce").mean()
        usable = prior_values.index[prior_values.notna()]
        target_rows.loc[index, usable] = prior_values.loc[usable].to_numpy()
        target_rows.at[index, "preseason_prior_applied"] = bool(len(usable))
        target_rows.at[index, "preseason_prior_method"] = "conference_mean"
        target_rows.at[index, "preseason_prior_neighbors"] = ";".join(
            sorted(peers["keys_team"].astype(str).tolist())
        )
    return target_rows


def _apply_newcomer_knn(*, frame, target_rows, missing, season, project_root, feature_columns, k_neighbors, donor_week, shrinkage):
    """Fill first-time FBS entrants using leakage-safe historical transitions."""
    teams_root = project_root / "data/raw/cfbd/v2/teams_fbs"
    rankings_root = project_root / "data/raw/cfbd/v2/rankings"
    entrants = _first_time_fbs_entrants(teams_root, max_season=season)
    donors = entrants[entrants["entry_season"].lt(season)].copy()
    queries = entrants[entrants["entry_season"].eq(season)].copy()
    if donors.empty or queries.empty:
        return target_rows
    donor_states = []
    for _, entrant in donors.iterrows():
        rows = frame[
            frame["keys_team"].astype(str).eq(entrant["team"])
            & pd.to_numeric(frame["keys_season"], errors="coerce").eq(entrant["entry_season"])
            & pd.to_numeric(frame["keys_week"], errors="coerce").between(1, donor_week)
        ].sort_values("keys_week")
        if not rows.empty:
            donor_states.append((entrant, rows.iloc[-1]))
    if not donor_states:
        return target_rows
    donor_meta = pd.DataFrame([x[0] for x in donor_states]).reset_index(drop=True)
    donor_values = pd.DataFrame([x[1][feature_columns] for x in donor_states]).reset_index(drop=True)
    donor_meta = _attach_fcs_poll_features(donor_meta, rankings_root)
    queries = _attach_fcs_poll_features(queries, rankings_root)
    numeric = ["fcs_rank", "fcs_points_share", "latitude", "longitude"]
    donor_numeric = donor_meta[numeric].apply(pd.to_numeric, errors="coerce")
    med = donor_numeric.median()
    scale = donor_numeric.std().replace(0, 1).fillna(1)
    global_prior = donor_values.apply(pd.to_numeric, errors="coerce").median()
    for index in target_rows.index[missing]:
        team = str(target_rows.at[index, "keys_team"])
        query = queries[queries["team"].eq(team)]
        if query.empty:
            continue
        q = query.iloc[0]
        dnum = donor_numeric.fillna(med)
        qnum = pd.to_numeric(q[numeric], errors="coerce").fillna(med)
        distance = np.sqrt((((dnum - qnum) / scale) ** 2).sum(axis=1))
        distance += donor_meta["conference"].ne(q["conference"]).astype(float) * 0.75
        chosen = distance.nsmallest(min(int(k_neighbors), len(distance)))
        weights = 1.0 / (chosen.to_numpy() + 0.25)
        weights /= weights.sum()
        values = donor_values.loc[chosen.index].apply(pd.to_numeric, errors="coerce")
        knn = values.mul(weights, axis=0).sum(axis=0, min_count=1)
        prior_values = (1.0 - float(shrinkage)) * knn + float(shrinkage) * global_prior
        usable = prior_values.index[prior_values.notna()]
        target_rows.loc[index, usable] = prior_values.loc[usable].to_numpy()
        target_rows.at[index, "preseason_prior_applied"] = bool(len(usable))
        target_rows.at[index, "preseason_prior_method"] = "first_time_fbs_knn"
        target_rows.at[index, "preseason_prior_neighbors"] = ";".join(
            f"{donor_meta.at[i, 'team']}:{donor_meta.at[i, 'entry_season']}:{distance.at[i]:.3f}"
            for i in chosen.index
        )
    return target_rows


def _first_time_fbs_entrants(root: Path, *, max_season: int) -> pd.DataFrame:
    seen: set[str] = set()
    rows = []
    for path in sorted(root.glob("*.parquet")):
        year = int(path.stem)
        if year > max_season:
            continue
        teams = pd.read_parquet(path)
        if teams.empty or "school" not in teams:
            continue
        for _, team in teams[~teams["school"].astype(str).isin(seen)].iterrows():
            if not seen:
                continue
            location = team.get("location") or {}
            if hasattr(location, "to_dict"):
                location = location.to_dict()
            rows.append({
                "team": str(team["school"]), "entry_season": year,
                "conference": team.get("conference"),
                "latitude": location.get("latitude") if isinstance(location, dict) else np.nan,
                "longitude": location.get("longitude") if isinstance(location, dict) else np.nan,
            })
        seen.update(teams["school"].astype(str))
    return pd.DataFrame(rows)


def _attach_fcs_poll_features(entrants: pd.DataFrame, rankings_root: Path) -> pd.DataFrame:
    out = entrants.copy()
    out["fcs_rank"] = np.nan
    out["fcs_points_share"] = np.nan
    for idx, row in out.iterrows():
        path = rankings_root / f"{int(row['entry_season']) - 1}.parquet"
        if not path.exists():
            continue
        rankings = pd.read_parquet(path)
        if rankings.empty or "polls" not in rankings:
            continue
        for snapshot in rankings.sort_values("week", ascending=False).itertuples():
            polls = snapshot.polls.tolist() if hasattr(snapshot.polls, "tolist") else snapshot.polls
            match = None
            for poll in polls:
                if str(poll.get("poll", "")).casefold() != "fcs coaches poll":
                    continue
                ranks = poll.get("ranks", [])
                ranks = ranks.tolist() if hasattr(ranks, "tolist") else ranks
                match = next((rank for rank in ranks if str(rank.get("school")) == row["team"]), None)
                if match:
                    max_points = max([float(rank.get("points") or 0) for rank in ranks] or [1.0])
                    out.at[idx, "fcs_rank"] = float(match["rank"])
                    out.at[idx, "fcs_points_share"] = float(match.get("points") or 0) / max(max_points, 1.0)
                    break
            if match:
                break
    return out


def materialize_preseason_state(
    *, source_path: str | Path, output_dir: str | Path, season: int, fingerprint: str
) -> dict:
    source = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[3]
    state = build_preseason_state_frame(pd.read_parquet(source), season=season, project_root=project_root)
    state_path = output / "preseason_state.parquet"
    state.to_parquet(state_path, index=False)
    feature_columns = [c for c in state if is_feature_column(c)]
    audit = pd.DataFrame(
        {
            "feature": feature_columns,
            "non_null_rows": [int(state[c].notna().sum()) for c in feature_columns],
            "unique_values": [int(state[c].nunique(dropna=True)) for c in feature_columns],
        }
    )
    audit.to_csv(output / "feature_availability_audit.csv", index=False)
    metadata = {
        "season": int(season),
        "fingerprint": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "same_team_carry_forward_then_conference_mean",
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "state_path": str(state_path.resolve()),
        "state_sha256": sha256_file(state_path),
        "rows": len(state),
        "teams": int(state["keys_team"].nunique()),
        "prior_applied_rows": int(state["preseason_prior_applied"].sum()),
        "prior_method_counts": state["preseason_prior_method"].value_counts(dropna=False).to_dict(),
    }
    (output / "state_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
