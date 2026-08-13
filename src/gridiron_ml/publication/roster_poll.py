"""Build the weekly TDNet poll from every checkpoint in the frozen roster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.models import load_model_checkpoint
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid

from .poll_recaps import aggregate_receiving_votes, format_receiving_votes, plot_consensus_poll_table
from .preseason_states import build_preseason_state_frame


# KNN is a ballot-producing model in the weekly workflow.  Only explicit
# naive baselines are excluded from the model-produced Top-25 poll.
COMPARATIVE_BASELINE_FAMILIES = frozenset({"naive"})
# Market-bearing scientific tiers are retained for research comparisons but
# never contribute to TDNet predictions, consensus, or polls.
INVALID_POLL_FEATURE_CONFIGS = frozenset({"F7", "F8"})
POLL_EXCLUDED_MODEL_IDS = frozenset(
    {
        "winner_linear_ols",
        "winner_linear_huber",
        "winner_linear_ridge",
        "winner_linear_sgd",
    }
)


def build_frozen_roster_poll(
    inventory_path: str | Path,
    *,
    season: int,
    week: int,
    output_dir: str | Path,
    project_root: str | Path,
    logo_dir: str | Path | None = None,
    top_n: int = 25,
    objective: str | None = None,
    reference_poll: pd.DataFrame | None = None,
    reference_label: str = "AP",
    render_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    """Create one objective-specific ballot per registered model.

    ``objective=None`` is retained for compatibility with older rehearsals;
    production publication should pass ``winner`` or ``margin`` so the two
    objective polls remain separately auditable.
    """
    root = Path(project_root).resolve()
    inventory = pd.read_csv(inventory_path)
    required = {"checkpoint_path", "fingerprint_path"}
    if not required.issubset(inventory):
        raise ValueError(f"Frozen roster is missing {sorted(required - set(inventory))}.")
    ballots = []
    failures = []
    enabled = inventory.loc[
        inventory.get("use_in_tdnet_poll", True).astype(str).str.lower().isin({"1", "true", "yes", "y"})
    ].copy()
    family = enabled.get("model_family", enabled.get("family", pd.Series("", index=enabled.index))).astype(str).str.lower()
    enabled = enabled.loc[~family.isin(COMPARATIVE_BASELINE_FAMILIES)].copy()
    if "feature_config" in enabled:
        enabled = enabled.loc[~enabled["feature_config"].astype(str).isin(INVALID_POLL_FEATURE_CONFIGS)].copy()
    if "model_id" in enabled:
        enabled = enabled.loc[~enabled["model_id"].astype(str).isin(POLL_EXCLUDED_MODEL_IDS)].copy()
    poll_objective = str(objective).strip().lower() if objective is not None else "all"
    if poll_objective not in {"all", "winner", "margin"}:
        raise ValueError("objective must be one of: winner, margin, or None.")
    if poll_objective != "all":
        if "objective" not in enabled:
            raise ValueError("Objective-specific polling requires an objective column in the inventory.")
        enabled = enabled.loc[enabled["objective"].astype(str).str.lower().eq(poll_objective)].copy()
    if enabled.empty:
        raise ValueError(f"No enabled frozen-roster models exist for objective={poll_objective!r}.")
    # Group by fingerprint so the expensive preseason-state construction and
    # matchup matrix are done once per fingerprint, while every checkpoint in
    # that group still contributes its own ballot.
    for fingerprint_path, group in enabled.groupby("fingerprint_path", dropna=False, sort=True):
        try:
            fp = Path(str(fingerprint_path))
            if not fp.is_absolute():
                fp = root / fp
            frame = pd.read_parquet(fp)
            if int(season) == 2026:
                state = build_preseason_state_frame(frame, season=season, project_root=root)
                keep = ~(
                    pd.to_numeric(frame["keys_season"], errors="coerce").eq(season)
                    & pd.to_numeric(frame["keys_week"], errors="coerce").eq(0)
                )
                shared = [column for column in frame if column in state]
                frame = pd.concat([frame.loc[keep], state[shared]], ignore_index=True, sort=False)
            models = []
            for _, row in group.iterrows():
                checkpoint = Path(str(row["checkpoint_path"]))
                if not checkpoint.is_absolute():
                    checkpoint = root / checkpoint
                label = str(row.get("final_model_name", row.get("model_name", row.get("concrete_model_type", "model"))))
                model = load_model_checkpoint(checkpoint)
                model.model_name = label
                models.append(model)
            evaluator = TDEval(
                config={"eval": {"artifact_root": str(Path(output_dir) / "_private_eval")}},
                fingerprints=StaticFrameFingerprints(frame),
                matchup_builder=MatchupBuilder(representation="unit_matchup"),
                model=models[0],
            )
            evaluator.poll(models=models, season=season, week=week, top_n=top_n, average_scope="season")
            ballots.append(evaluator.poll_ballots_.copy())
            failures.extend(getattr(evaluator, "poll_model_failures_", []))
        except Exception as exc:
            failures.extend({"model": str(row.get("final_model_name", "model")), "reason": str(exc)} for _, row in group.iterrows())
    if not ballots:
        reasons = "; ".join(
            f"{item.get('model', 'model')}: {item.get('reason', 'unknown failure')}"
            for item in failures[:5]
        )
        raise RuntimeError(
            "No frozen-roster model produced a valid weekly ballot."
            + (f" First failures: {reasons}" if reasons else "")
        )
    ballots = pd.concat(ballots, ignore_index=True)
    ballots.insert(0, "poll_objective", poll_objective)
    poll = (
        ballots.groupby("keys_team", as_index=False)
        .agg(
            poll_points=("poll_points", "sum"), ballots_seen=("ballot_model", "nunique"),
            top25_votes=("top25_vote", "sum"), first_place_votes=("first_place_vote", "sum"),
            average_rank=("ballot_rank", "mean"), best_rank=("ballot_rank", "min"),
            worst_rank=("ballot_rank", "max"),
        )
        .sort_values(["poll_points", "average_rank", "best_rank", "keys_team"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    poll.insert(0, "rank", np.arange(1, len(poll) + 1))
    poll = poll.head(top_n).copy()
    poll.insert(0, "week", int(week))
    poll.insert(0, "season", int(season))
    poll.insert(2, "poll_objective", poll_objective)
    if reference_poll is not None and not reference_poll.empty:
        reference = reference_poll.copy()
        team_column = "team" if "team" in reference else "keys_team"
        rank_lookup = dict(zip(reference[team_column].astype(str), reference["rank"]))
        poll["reference_rank"] = poll["keys_team"].astype(str).map(rank_lookup)
        poll["tdnet_minus_reference"] = poll["rank"] - poll["reference_rank"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receiving = aggregate_receiving_votes(poll, ballots, top_n=top_n)
    receiving_text = format_receiving_votes(receiving)
    poll.to_csv(output / "tdnet_top25.csv", index=False)
    ballots.to_csv(output / "tdnet_model_ballots.csv", index=False)
    receiving.to_csv(output / "tdnet_receiving_votes.csv", index=False)
    (output / "receiving_votes.txt").write_text(receiving_text + "\n", encoding="utf-8")
    pd.DataFrame(failures).to_csv(output / "tdnet_poll_model_failures.csv", index=False)
    if render_figures:
        plot_consensus_poll_table(
            poll, output / "tdnet_top25.png", title=f"{season} Week {week}: TDNet {poll_objective.title()}-Objective Top 25",
            receiving_votes=receiving_text,
            logo_dir=logo_dir, reference_label=reference_label,
        )
        plot_ballot_logo_grid(
            ballots, output / "tdnet_model_ballots.png", top_n=top_n, logo_dir=logo_dir,
            title=f"{season} Week {week}: TDNet {poll_objective.title()}-Objective Model Ballots",
        )
    return {"poll": poll, "ballots": ballots, "failures": pd.DataFrame(failures)}
