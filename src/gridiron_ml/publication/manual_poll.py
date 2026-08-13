"""Reusable weekly manual Top-25 ballot workflow for the publication notebook."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

from gridiron_ml.models import load_model_checkpoint
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.poll_viz import (
    plot_ballot_logo_grid,
    plot_weekly_top25_table,
)
from gridiron_ml.td_run.weekly_report import discover_latest_checkpoints
from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints


def find_latest_model_poll(project_root, *, season=None, week=None, top_n=25):
    """Find the newest model-produced poll and return ``(path, top25)``.

    Candidate locations cover the normal weekly report, comparison, frozen
    poll, and publication output layouts.  Rows are ordered by their recorded
    season/week first and file modification time second, so a stale file from
    a newer directory cannot displace a genuinely newer poll.
    """
    root = Path(project_root).resolve()
    patterns = (
        "data/publication/**/*.csv",
        "data/weekly_reports/**/tables/current_poll_top25.csv",
        "data/weekly_reports/**/tables/weekly_poll_top25.csv",
        "data/comparisons/**/polls/tables/weekly_poll_top25.csv",
        "reports/final_model_polls/**/*.csv",
        "publication/**/*.csv",
    )
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            if "manual_polls" in path.parts:
                continue
            seen.add(path)
            try:
                frame = pd.read_csv(path)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            normalized = _normalize_poll(frame, top_n=top_n)
            if normalized.empty:
                continue
            if season is not None and "season" in normalized:
                normalized = normalized.loc[
                    pd.to_numeric(normalized["season"], errors="coerce").eq(int(season))
                ]
            if week is not None and "week" in normalized:
                normalized = normalized.loc[
                    pd.to_numeric(normalized["week"], errors="coerce").eq(int(week))
                ]
            if normalized.empty:
                continue
            if len(normalized) < int(top_n) and (season is not None or week is not None):
                continue
            latest_season = _max_number(normalized.get("season"), default=-1)
            latest_week = _max_number(normalized.get("week"), default=-1)
            candidates.append((latest_season, latest_week, path.stat().st_mtime, path, normalized))
    if not candidates:
        raise FileNotFoundError(
            "No model-produced Top-25 poll was found. Build a weekly/frozen poll first."
        )
    _, _, _, path, frame = max(candidates, key=lambda row: row[:3])
    return path, _latest_poll_slice(frame, top_n=top_n)


def ballot_store_path(project_root, season):
    """Return the persistent per-season manual ballot store path."""
    return Path(project_root).resolve() / "data" / "publication" / str(int(season)) / "manual_polls" / "manual_ballots.csv"


def load_saved_ballot(project_root, *, season, week, ballot_name="manual_poll", top_n=25):
    """Load a previously submitted weekly ballot, or return an empty list."""
    path = ballot_store_path(project_root, season)
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return []
    required = {"season", "week", "rank", "keys_team"}
    if not required.issubset(frame.columns):
        return []
    ballot_models = frame["ballot_model"].astype(str) if "ballot_model" in frame.columns else pd.Series(str(ballot_name), index=frame.index)
    selected = frame.loc[
        pd.to_numeric(frame["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(frame["week"], errors="coerce").eq(int(week))
        & ballot_models.eq(str(ballot_name))
    ].sort_values("rank")
    return selected["keys_team"].astype(str).head(int(top_n)).tolist()


def save_manual_ballot(project_root, *, season, week, teams, ballot_name="manual_poll", top_n=25):
    """Validate and persist one season/week ballot, replacing that ballot only."""
    teams = validate_ballot(teams, top_n=top_n)
    path = ballot_store_path(project_root, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            existing = pd.DataFrame()
    if not existing.empty and {"season", "week", "ballot_model"}.issubset(existing.columns):
        keep = ~(
            pd.to_numeric(existing["season"], errors="coerce").eq(int(season))
            & pd.to_numeric(existing["week"], errors="coerce").eq(int(week))
            & existing["ballot_model"].astype(str).eq(str(ballot_name))
        )
        existing = existing.loc[keep]
    new = pd.DataFrame(
        {
            "season": int(season),
            "week": int(week),
            "ballot_model": str(ballot_name),
            "rank": range(1, len(teams) + 1),
            "keys_team": teams,
        }
    )
    out = pd.concat([existing, new], ignore_index=True, sort=False)
    out = out.sort_values(["season", "week", "ballot_model", "rank"]).reset_index(drop=True)
    out.to_csv(path, index=False)
    return path


def validate_ballot(teams, *, top_n=25):
    """Require exactly ``top_n`` non-empty, unique team names."""
    values = [str(team).strip() for team in list(teams or [])]
    if len(values) != int(top_n):
        raise ValueError(f"A Top-{top_n} ballot must contain exactly {top_n} teams; got {len(values)}.")
    if any(not value for value in values):
        raise ValueError("A ballot cannot contain blank team names.")
    if len(set(values)) != len(values):
        raise ValueError("A ballot cannot rank the same team twice.")
    return values


def _materialize_poll_frame(path: Path, *, season: int, project_root: Path) -> pd.DataFrame:
    """Load a roster fingerprint and apply the same preseason policy as weekly polling."""
    frame = pd.read_parquet(path)
    if int(season) == 2026 and {"keys_season", "keys_week"}.issubset(frame.columns):
        from gridiron_ml.publication.preseason_states import build_preseason_state_frame

        state = build_preseason_state_frame(frame, season=season, project_root=project_root)
        keep = ~(
            pd.to_numeric(frame["keys_season"], errors="coerce").eq(season)
            & pd.to_numeric(frame["keys_week"], errors="coerce").eq(0)
        )
        shared = [column for column in frame if column in state]
        frame = pd.concat([frame.loc[keep], state[shared]], ignore_index=True, sort=False)
    return frame


def _aggregate_poll_ballots(ballots: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Aggregate ballots produced against multiple exact fingerprint variants."""
    poll = (
        ballots.groupby("keys_team", as_index=False)
        .agg(
            poll_points=("poll_points", "sum"),
            ballots_seen=("ballot_model", "nunique"),
            top25_votes=("top25_vote", "sum"),
            first_place_votes=("first_place_vote", "sum"),
            average_rank=("ballot_rank", "mean"),
            best_rank=("ballot_rank", "min"),
            worst_rank=("ballot_rank", "max"),
        )
        .sort_values(
            ["poll_points", "average_rank", "best_rank", "keys_team"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )
    poll.insert(0, "rank", range(1, len(poll) + 1))
    poll["average_rank"] = poll["average_rank"].astype(float).round(3)
    return poll.head(int(top_n)).copy()


def run_manual_poll(
    project_root,
    *,
    season,
    week,
    teams,
    ballot_name="manual_poll",
    top_n=25,
    fingerprint_version=0,
    average_scope="season",
    models_root=None,
    inventory_path=None,
    logo_dir=None,
    output_dir=None,
    figure_output_dir=None,
    objective=None,
):
    """Run one objective's models plus the named manual ballot for one week."""
    root = Path(project_root).resolve()
    teams = validate_ballot(teams, top_n=top_n)
    poll_objective = str(objective).strip().lower() if objective is not None else "all"
    if poll_objective not in {"all", "winner", "margin"}:
        raise ValueError("objective must be one of: winner, margin, or None.")
    save_manual_ballot(root, season=season, week=week, teams=teams, ballot_name=ballot_name, top_n=top_n)
    models_root = Path(models_root) if models_root is not None else root / "models"
    if inventory_path is not None:
        from gridiron_ml.publication.weekly import _load_inventory_models

        inventory = pd.read_csv(inventory_path)
        if poll_objective != "all":
            if "objective" not in inventory:
                raise ValueError("Objective-specific polling requires an objective column in the inventory.")
            inventory = inventory.loc[
                inventory["objective"].astype(str).str.lower().eq(poll_objective)
            ].copy()
        entries, _ = _load_inventory_models(inventory, root)
        if "use_in_tdnet_poll" in inventory.columns:
            enabled = inventory.loc[
                inventory["use_in_tdnet_poll"].astype(str).str.lower().isin({"1", "true", "yes"}),
                "checkpoint_path",
            ].astype(str)
            enabled_paths = {
                str((root / path).resolve()) if not Path(path).is_absolute() else str(Path(path).resolve())
                for path in enabled
            }
            entries = [
                entry for entry in entries
                if str(Path(entry["checkpoint_path"]).resolve()) in enabled_paths
            ]
    else:
        entries, inventory = discover_latest_checkpoints(models_root, include_models="all")
    if not entries:
        raise FileNotFoundError(f"No loadable model checkpoints found under {models_root}.")
    models = [entry["model"] for entry in entries]
    inventory_poll_failures = []
    if inventory_path is not None and "fingerprint_path" in inventory.columns:
        # Search/refit artifacts can legitimately use several fingerprint
        # variants. Poll each checkpoint against the exact variant it was
        # trained on, then combine the resulting ballots.
        entry_by_checkpoint = {
            str(Path(entry["checkpoint_path"]).resolve()): entry for entry in entries
        }
        grouped_entries = {}
        for _, row in inventory.iterrows():
            raw_checkpoint = row.get("checkpoint_path")
            raw_fingerprint = row.get("fingerprint_path")
            if pd.isna(raw_checkpoint) or pd.isna(raw_fingerprint):
                continue
            checkpoint = Path(str(raw_checkpoint))
            checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
            fingerprint = Path(str(raw_fingerprint))
            fingerprint = fingerprint if fingerprint.is_absolute() else root / fingerprint
            entry = entry_by_checkpoint.get(str(checkpoint.resolve()))
            if entry is not None:
                grouped_entries.setdefault(str(fingerprint.resolve()), []).append(entry)

        ballot_frames = []
        manual_added = False
        for raw_fingerprint, group_entries in sorted(grouped_entries.items()):
            fingerprint_path = Path(raw_fingerprint)
            if not fingerprint_path.exists():
                inventory_poll_failures.extend(
                    {"model": entry.get("model_name", "model"), "reason": f"missing fingerprint: {fingerprint_path}"}
                    for entry in group_entries
                )
                continue
            try:
                frame = _materialize_poll_frame(fingerprint_path, season=int(season), project_root=root)
                evaluator = TDEval(
                    {"fingerprints": {"version": int(fingerprint_version), "root": str(root)}},
                    fingerprints=StaticFrameFingerprints(frame),
                    matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
                    model=group_entries[0]["model"],
                )
                evaluator.poll(
                    models=[entry["model"] for entry in group_entries],
                    season=int(season),
                    week=int(week),
                    average_scope=average_scope,
                    top_n=int(top_n),
                    manual_ballots=(
                        {int(week): {"name": str(ballot_name), "teams": teams}}
                        if not manual_added else None
                    ),
                )
                ballot_frames.append(evaluator.poll_ballots_.copy())
                inventory_poll_failures.extend(getattr(evaluator, "poll_model_failures_", []))
                manual_added = True
            except Exception as exc:
                inventory_poll_failures.extend(
                    {"model": entry.get("model_name", "model"), "reason": str(exc)}
                    for entry in group_entries
                )
        if not ballot_frames:
            raise RuntimeError("No inventory model produced a valid manual-poll ballot.")
        ballots = pd.concat(ballot_frames, ignore_index=True)
        poll = _aggregate_poll_ballots(ballots, top_n=int(top_n))
        evaluator = None
    else:
        evaluator = TDEval(
            {"fingerprints": {"version": int(fingerprint_version), "root": str(root)}},
            fingerprints=Fingerprints(version=int(fingerprint_version), root=root),
            matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
            model=models[0],
        )
        poll = evaluator.poll(
            models=models,
            season=int(season),
            week=int(week),
            average_scope=average_scope,
            top_n=int(top_n),
            manual_ballots={int(week): {"name": str(ballot_name), "teams": teams}},
        ).head(int(top_n)).copy()
        ballots = evaluator.poll_ballots_.copy()
    ballots.insert(0, "poll_objective", poll_objective)
    poll.insert(2, "poll_objective", poll_objective)
    poll.insert(0, "week", int(week))
    poll.insert(0, "season", int(season))
    output = Path(output_dir) if output_dir is not None else root / "data" / "publication" / str(int(season)) / "manual_polls" / f"week_{int(week):02d}"
    tables = output / "tables"
    figures = Path(figure_output_dir) if figure_output_dir is not None else output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    poll.to_csv(tables / "manual_poll_top25.csv", index=False)
    ballots.to_csv(tables / "manual_poll_ballots.csv", index=False)
    inventory.to_csv(tables / "checkpoint_inventory.csv", index=False)
    poll_failures = pd.DataFrame(
        inventory_poll_failures if evaluator is None else getattr(evaluator, "poll_model_failures_", [])
    )
    poll_failures.to_csv(tables / "poll_model_failures.csv", index=False)
    week_poll = poll.copy()
    consensus_figure = plot_weekly_top25_table(
        week_poll, figures / "manual_poll_top25.png", top_n=top_n, logo_dir=logo_dir,
    )
    ballot_figure = plot_ballot_logo_grid(
        ballots, figures / "manual_poll_all_ballots.png", top_n=top_n, logo_dir=logo_dir,
        title=f"{season} Week {week}: {poll_objective.title()}-objective model ballots + {ballot_name}",
    )
    metadata = {
        "season": int(season),
        "week": int(week),
        "manual_ballot_name": str(ballot_name),
        "manual_ballot": teams,
        "poll_objective": poll_objective,
        "model_count": len(models),
        "models": [entry.get("name", entry.get("model_name", "unknown")) for entry in entries],
        "poll_model_failure_count": int(len(poll_failures)),
        "knn_participated": any("knn" in str(entry.get("name", entry.get("model_name", ""))).lower() for entry in entries),
        "output_dir": str(output),
    }
    (output / "manual_poll_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "poll": poll,
        "ballots": ballots,
        "figures": {"top25": consensus_figure, "all_ballots": ballot_figure},
        "output_dir": output,
        "metadata": metadata,
    }


def make_drag_poll_editor(teams, *, logo_dir=None, top_n=25, ballot_name="manual_poll"):
    """Create an HTML5 drag-and-drop editor and a Python submit button.

    The hidden ``Text`` widget is the small comms bridge from browser-side
    drag events back to Python.  A plain up/down fallback is intentionally not
    needed: users drag the visible logo tiles, then click the returned button.
    """
    import ipywidgets as widgets
    from IPython.display import HTML, Javascript, display

    teams = validate_ballot(teams, top_n=top_n)
    editor_id = "tdnet-manual-poll-editor"
    state = widgets.Text(value=json.dumps(teams), layout=widgets.Layout(width="1px", height="1px", opacity="0"))
    submit = widgets.Button(description=f"Submit {ballot_name}", button_style="primary", icon="check")
    output = widgets.Output()
    tiles = []
    for index, team in enumerate(teams):
        image = _logo_data_uri(team, logo_dir)
        image_html = f'<img src="{image}" alt="{html.escape(team)}" />' if image else ""
        tiles.append(
            f'<div class="tdnet-poll-tile" draggable="true" data-team="{html.escape(team)}">'
            f'<div class="tdnet-poll-rank">{index + 1}</div>{image_html}'
            f'<div class="tdnet-poll-name">{html.escape(team)}</div></div>'
        )
    markup = HTML(
        f'''<style>
        #{editor_id} {{ display:grid; grid-template-columns:repeat(5,minmax(125px,1fr)); gap:8px; max-width:900px; }}
        #{editor_id} .tdnet-poll-tile {{ border:1px solid #ccd4df; border-radius:8px; padding:7px; text-align:center; background:#f8fafc; cursor:grab; min-height:90px; }}
        #{editor_id} .tdnet-poll-tile.dragging {{ opacity:.35; }} #{editor_id} img {{ height:44px; max-width:72px; object-fit:contain; display:block; margin:auto; }}
        #{editor_id} .tdnet-poll-rank {{ font-weight:700; color:#526173; }} #{editor_id} .tdnet-poll-name {{ font-size:11px; margin-top:3px; }}
        </style><div id="{editor_id}">{''.join(tiles)}</div>'''
    )
    display(markup, state, submit, output)
    display(Javascript(
        f'''(() => {{ const root=document.getElementById("{editor_id}"); if(!root) return;
        let dragged=null; const sync=()=>{{ const value=[...root.querySelectorAll(".tdnet-poll-tile")].map(x=>x.dataset.team);
        const inputs=[...document.querySelectorAll("input")].filter(x=>x.value && x.value.startsWith("[\\\""));
        const input=inputs[inputs.length-1]; if(input){{input.value=JSON.stringify(value); input.dispatchEvent(new Event("input",{{bubbles:true}}));}};
        root.querySelectorAll(".tdnet-poll-tile").forEach((tile,index)=>tile.querySelector(".tdnet-poll-rank").textContent=index+1);}};
        root.addEventListener("dragstart",e=>{{dragged=e.target.closest(".tdnet-poll-tile"); dragged.classList.add("dragging");}});
        root.addEventListener("dragend",()=>{{if(dragged) dragged.classList.remove("dragging"); dragged=null; sync();}});
        root.addEventListener("dragover",e=>{{e.preventDefault(); const target=e.target.closest(".tdnet-poll-tile"); if(!dragged||!target||target===dragged)return;
        const box=target.getBoundingClientRect(); root.insertBefore(dragged, e.clientX < box.left+box.width/2 ? target : target.nextSibling);}}); sync(); }})();'''
    ))
    return state, submit, output


def _normalize_poll(frame, *, top_n):
    frame = pd.DataFrame(frame).copy()
    team_col = next((c for c in ["keys_team", "team", "school", "name"] if c in frame.columns), None)
    if team_col is None:
        return pd.DataFrame()
    frame["keys_team"] = frame[team_col].astype(str)
    if "rank" not in frame.columns:
        group_keys = [c for c in ["season", "week"] if c in frame.columns]
        if group_keys:
            frame["rank"] = frame.groupby(group_keys)["keys_team"].cumcount() + 1
        else:
            frame["rank"] = range(1, len(frame) + 1)
    return frame.loc[pd.to_numeric(frame["rank"], errors="coerce").le(int(top_n))].copy()


def _latest_poll_slice(frame, *, top_n):
    keys = [c for c in ["season", "week"] if c in frame.columns]
    if keys:
        for key in keys:
            frame[key] = pd.to_numeric(frame[key], errors="coerce")
        values = {key: frame[key].max() for key in keys}
        for key, value in values.items():
            frame = frame.loc[frame[key].eq(value)]
    return frame.sort_values("rank").head(int(top_n)).reset_index(drop=True)


def _max_number(series, *, default):
    if series is None:
        return default
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else default


def _logo_data_uri(team, logo_dir):
    if logo_dir is None:
        return None
    from gridiron_ml.td_run.poll_viz import resolve_team_logo_path

    path = resolve_team_logo_path(team, logo_dir)
    if path is None:
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
