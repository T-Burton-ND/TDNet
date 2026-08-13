"""src.gridiron_ml.td_sim.recursive_simulator.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.td_run.matchups import MatchupBuilder

from .bootstrap import append_bootstrap_week0
from .checkpoints import discover_model_checkpoints
from .probability import clip_probabilities, sigmoid_margin_to_prob
from .recursive_plots import save_recursive_top25_plot
from .schedule_loader import ScheduleLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MUTABLE_PREFIXES = ("offense_", "defense_", "statOff_", "statDef_", "statGen_", "statSpe_")


class RecursiveSeasonSimulator:
    """Week-by-week TD Sim engine with evolving synthetic fingerprints."""

    def __init__(self, config=None, schedule=None, fingerprint_frame=None, matchup_builder=None, model_specs=None):
        """Internal helper for the init__ step."""
        self.config = dict(config or {})
        self.schedule = schedule
        self.fingerprint_frame = fingerprint_frame
        self.matchup_builder = matchup_builder or MatchupBuilder(**self.config.get("matchup", {}))
        self.model_specs = model_specs

    def run(
        self,
        season,
        N=None,
        models=None,
        workflow="recursive_single_model",
        schedule_mode="full_schedule",
        as_of_week=None,
        output_dir=None,
        sim_start=0,
        shard_id=None,
        save_debug=None,
        show_progress=None,
    ):
        """Run the run step and return its normalized result."""
        season = int(season)
        n_sims = int(N or self.config.get("simulation", {}).get("n_sims", 10))
        show_progress = self._show_progress(show_progress)
        specs = self._select_model_specs(models=models, workflow=workflow)
        schedule = self._load_schedule(season)
        fingerprint_frame = self._load_fingerprint_frame()
        teams = self._scheduled_teams(schedule, season)
        fingerprint_frame = self._bootstrap_target_week0(fingerprint_frame=fingerprint_frame, season=season, teams=teams)
        state_factory = FingerprintStateFactory(self.config, fingerprint_frame=fingerprint_frame)
        initial_state = state_factory.initial_state(season=season, teams=teams)
        sampler = HistoricalGameSampler(
            config=self.config,
            fingerprint_frame=fingerprint_frame,
            feature_cols=initial_state.feature_cols,
            mutable_cols=initial_state.mutable_cols,
        )

        all_records = []
        all_ballots = []
        all_games = []
        final_states = []
        model_iter = self._progress(
            list(enumerate(specs)),
            desc="TD Sim models",
            enabled=show_progress and len(specs) > 1,
            leave=True,
        )
        for model_idx, spec in model_iter:
            sim_iter = self._progress(
                range(n_sims),
                desc=f"{spec['name']} simulations",
                enabled=show_progress,
                leave=True,
            )
            for offset in sim_iter:
                sim_id = int(sim_start) + offset
                rng = np.random.default_rng(self._seed(model_idx=model_idx, sim_id=sim_id))
                result = self._run_one_sim(
                    season=season,
                    sim_id=sim_id,
                    model_spec=spec,
                    schedule=schedule,
                    state=initial_state.copy(),
                    sampler=sampler,
                    rng=rng,
                    schedule_mode=schedule_mode,
                    as_of_week=as_of_week,
                    show_progress=show_progress,
                )
                all_records.append(result["records"])
                all_ballots.append(result["ballots"])
                all_games.append(result["games"])
                if self.config.get("recursive", {}).get("save_final_fingerprints", False):
                    final_states.append(result["final_state"])

        records = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
        ballots = pd.concat(all_ballots, ignore_index=True) if all_ballots else pd.DataFrame()
        game_predictions = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
        final_records = self._summarize_records(records)
        average_poll = self._summarize_poll(ballots=ballots, final_records=final_records, top_n=self._top_n())
        sim_accuracy = self._summarize_sim_accuracy(game_predictions)

        output_dir = self._output_dir(season=season, workflow=workflow, models=specs, output_dir=output_dir, shard_id=shard_id)
        saved = self._save_outputs(
            output_dir=output_dir,
            final_records=final_records,
            average_poll=average_poll,
            records=records,
            ballots=ballots,
            game_predictions=game_predictions,
            sim_accuracy=sim_accuracy,
            final_states=final_states,
            save_debug=bool(self.config.get("recursive", {}).get("save_debug", False) if save_debug is None else save_debug),
        )

        return {
            "final_records": final_records,
            "average_poll": average_poll,
            "simulation_records": records,
            "game_predictions": game_predictions,
            "simulation_accuracy": sim_accuracy,
            "poll_ballots": ballots,
            "output_dir": output_dir,
            "saved": saved,
            "models": pd.DataFrame([{k: v for k, v in spec.items() if k != "model"} for spec in specs]),
        }

    def _run_one_sim(self, season, sim_id, model_spec, schedule, state, sampler, rng, schedule_mode, as_of_week, show_progress=False):
        """Internal helper for the run_one_sim step."""
        model = model_spec["model"]
        model_name = model_spec["name"]
        games = self._simulation_games(schedule, season=season, schedule_mode=schedule_mode, as_of_week=as_of_week)
        wins = {team: 0 for team in state.teams}
        losses = {team: 0 for team in state.teams}
        game_rows = []

        for _, completed in games.loc[games["is_completed"]].iterrows():
            self._apply_game(
                row=completed,
                model_name=model_name,
                sim_id=sim_id,
                home_margin=float(completed["home_points"]) - float(completed["away_points"]),
                state=state,
                sampler=sampler,
                rng=rng,
                wins=wins,
                losses=losses,
            )

        pending = games.loc[~games["is_completed"]].sort_values(["week", "game_id", "home_team", "away_team"])
        weeks = sorted(pd.to_numeric(pending["week"], errors="coerce").dropna().astype(int).unique())
        week_iter = self._progress(
            weeks,
            desc=f"{model_name} sim {sim_id} weeks",
            enabled=show_progress,
            leave=False,
        )
        for week in week_iter:
            week_games = pending.loc[pd.to_numeric(pending["week"], errors="coerce") == int(week)].reset_index(drop=True)
            if week_games.empty:
                continue
            pred = self._predict_week(model=model, games=week_games, state=state, season=season)
            for game_idx, game in week_games.iterrows():
                pred_row = pred.iloc[game_idx]
                home_margin = self._sample_home_margin(pred_row=pred_row, rng=rng)
                game_rows.append(
                    self._game_prediction_row(
                        row=game,
                        model_name=model_name,
                        sim_id=sim_id,
                        pred_row=pred_row,
                        sampled_home_margin=home_margin,
                    )
                )
                self._apply_game(
                    row=game,
                    model_name=model_name,
                    sim_id=sim_id,
                    home_margin=home_margin,
                    state=state,
                    sampler=sampler,
                    rng=rng,
                    wins=wins,
                    losses=losses,
                )

        records = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "sim_id": sim_id,
                    "season": int(season),
                    "team": team,
                    "wins": int(wins.get(team, 0)),
                    "losses": int(losses.get(team, 0)),
                    "games": int(wins.get(team, 0) + losses.get(team, 0)),
                    "win_pct": float(wins.get(team, 0) / max(wins.get(team, 0) + losses.get(team, 0), 1)),
                }
                for team in state.teams
            ]
        )
        ballots = self._poll_ballot(model=model, model_name=model_name, sim_id=sim_id, season=season, state=state)
        final_state = state.frame.copy()
        final_state.insert(0, "model", model_name)
        final_state.insert(1, "sim_id", sim_id)
        final_state.insert(2, "season", int(season))
        games = pd.DataFrame(game_rows)
        return {"records": records, "ballots": ballots, "games": games, "final_state": final_state}

    def _show_progress(self, explicit):
        """Internal helper for the show_progress step."""
        if explicit is not None:
            return bool(explicit)
        runtime = self.config.get("runtime", {})
        if "show_progress" in runtime:
            return bool(runtime.get("show_progress"))
        return bool(self.config.get("recursive", {}).get("show_progress", False))

    def _progress(self, iterable, desc=None, enabled=True, leave=False):
        """Internal helper for the progress step."""
        if not enabled:
            return iterable
        try:
            from tqdm.auto import tqdm
        except Exception:
            return iterable
        return tqdm(iterable, desc=desc, leave=leave)

    def _predict_week(self, model, games, state, season):
        """Internal helper for the predict_week step."""
        home = state.frame.reindex(games["home_team"].astype(str)).reset_index(drop=True)
        away = state.frame.reindex(games["away_team"].astype(str)).reset_index(drop=True)
        if home.isna().all(axis=1).any() or away.isna().all(axis=1).any():
            missing = sorted(set(games["home_team"]).union(set(games["away_team"])) - set(state.frame.index))
            raise ValueError(f"Recursive TD Sim state is missing scheduled teams: {missing[:10]}")
        matchup = self.matchup_builder.build_many(home.loc[:, state.feature_cols], away.loc[:, state.feature_cols])
        meta = pd.DataFrame(
            {
                "keys_season": int(season),
                "keys_week": games["week"].astype(int).to_numpy(),
                "keys_game_id": games["game_id"].to_numpy(),
                "keys_team_home": games["home_team"].astype(str).to_numpy(),
                "keys_team_away": games["away_team"].astype(str).to_numpy(),
            }
        )
        return model.predict(matchup, meta_df=meta)

    def _apply_game(self, row, model_name, sim_id, home_margin, state, sampler, rng, wins, losses):
        """Internal helper for the apply_game step."""
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        if home_margin >= 0:
            wins[home_team] = wins.get(home_team, 0) + 1
            losses[away_team] = losses.get(away_team, 0) + 1
        else:
            wins[away_team] = wins.get(away_team, 0) + 1
            losses[home_team] = losses.get(home_team, 0) + 1

        home_vector = sampler.sample(
            team=home_team,
            opponent=away_team,
            is_home=True,
            state=state,
            team_margin=home_margin,
            rng=rng,
        )
        away_vector = sampler.sample(
            team=away_team,
            opponent=home_team,
            is_home=False,
            state=state,
            team_margin=-home_margin,
            rng=rng,
        )
        state.update_game(home_team, home_vector)
        state.update_game(away_team, away_vector)

    def _sample_home_margin(self, pred_row, rng):
        """Internal helper for the sample_home_margin step."""
        margin = float(pd.to_numeric(pd.Series([pred_row.get("pred_margin", pred_row.get("predicted_home_margin", 0.0))]), errors="coerce").fillna(0.0).iloc[0])
        prob = pred_row.get("pred_proba_home_win", pred_row.get("predicted_home_win_prob", np.nan))
        if pd.isna(prob):
            prob = sigmoid_margin_to_prob(
                [margin],
                scale=float(self.config.get("simulation", {}).get("sigmoid_scale", 13.5)),
                random_scaling_factor=float(self.config.get("simulation", {}).get("random_scaling_factor", 1.0)),
            ).iloc[0]
        prob = self._scale_probability(float(prob))
        home_win = bool(rng.random() < prob)
        noise_std = float(self.config.get("simulation", {}).get("game_noise_std") or self.config.get("simulation", {}).get("model_residual_std", 14.0))
        noise_std *= float(self.config.get("simulation", {}).get("residual_std_multiplier", 1.0))
        sampled = margin + rng.normal(0.0, noise_std)
        magnitude = max(abs(sampled), 0.1)
        return float(magnitude if home_win else -magnitude)

    def _game_prediction_row(self, row, model_name, sim_id, pred_row, sampled_home_margin):
        """Return one game-level simulation prediction row for retrospective scoring."""
        pred_margin = float(
            pd.to_numeric(
                pd.Series([pred_row.get("pred_margin", pred_row.get("predicted_home_margin", np.nan))]),
                errors="coerce",
            ).iloc[0]
        )
        pred_prob = pd.to_numeric(
            pd.Series([pred_row.get("pred_proba_home_win", pred_row.get("predicted_home_win_prob", np.nan))]),
            errors="coerce",
        ).iloc[0]
        home_points = pd.to_numeric(pd.Series([row.get("home_points", np.nan)]), errors="coerce").iloc[0]
        away_points = pd.to_numeric(pd.Series([row.get("away_points", np.nan)]), errors="coerce").iloc[0]
        actual_home_margin = float(home_points - away_points) if pd.notna(home_points) and pd.notna(away_points) else np.nan
        simulated_winner = "home" if float(sampled_home_margin) >= 0 else "away"
        predicted_favorite = "home" if pred_margin >= 0 else "away"
        actual_winner = "home" if actual_home_margin >= 0 else "away" if np.isfinite(actual_home_margin) else pd.NA
        has_actual_winner = isinstance(actual_winner, str) and actual_winner in ("home", "away")
        return {
            "model": model_name,
            "sim_id": int(sim_id),
            "season": int(row.get("season")),
            "week": int(row.get("week")),
            "game_id": row.get("game_id"),
            "home_team": str(row.get("home_team")),
            "away_team": str(row.get("away_team")),
            "pred_home_margin": pred_margin,
            "pred_home_win_probability": float(pred_prob) if pd.notna(pred_prob) else np.nan,
            "sampled_home_margin": float(sampled_home_margin),
            "actual_home_margin": actual_home_margin,
            "predicted_favorite": predicted_favorite,
            "simulated_winner": simulated_winner,
            "actual_winner": actual_winner,
            "simulated_called_upset": simulated_winner != predicted_favorite,
            "simulated_correct": simulated_winner == actual_winner if has_actual_winner else pd.NA,
        }

    def _scale_probability(self, probability):
        """Internal helper for the scale_probability step."""
        sim_cfg = self.config.get("simulation", {})
        p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        scaling = float(sim_cfg.get("random_scaling_factor", 1.0))
        if scaling != 1.0:
            logit = np.log(p / (1.0 - p))
            p = float(1.0 / (1.0 + np.exp(-logit / max(scaling, 1e-8))))
        return float(
            clip_probabilities(
                [p],
                min_prob=float(sim_cfg.get("clip_win_probability_min", 0.01)),
                max_prob=float(sim_cfg.get("clip_win_probability_max", 0.99)),
            ).iloc[0]
        )

    def _poll_ballot(self, model, model_name, sim_id, season, state):
        """Internal helper for the poll_ballot step."""
        meta = pd.DataFrame({"keys_team": state.frame.index.astype(str), "keys_season": int(season), "keys_week": state.final_week})
        average = pd.DataFrame([state.frame.loc[:, state.feature_cols].mean(numeric_only=True)], columns=state.feature_cols)
        rank_X, rank_meta, _ = self.matchup_builder.team_vs_average(
            state.frame.loc[:, state.feature_cols].reset_index(drop=True),
            meta_df=meta.reset_index(drop=True),
            average_team_df=average,
        )
        ranked = model.total_rank(rank_X, meta_df=rank_meta).reset_index(drop=True)
        team_col = "keys_team" if "keys_team" in ranked.columns else ranked.columns[0]
        score_col = "score" if "score" in ranked.columns else "pred_margin"
        ranked = ranked.sort_values(score_col, ascending=False).reset_index(drop=True)
        ranked["poll_rank"] = np.arange(1, len(ranked) + 1)
        top_n = self._top_n()
        ballot = ranked.head(top_n).loc[:, [team_col, "poll_rank"]].rename(columns={team_col: "team"})
        ballot.insert(0, "model", model_name)
        ballot.insert(1, "sim_id", int(sim_id))
        ballot.insert(2, "season", int(season))
        ballot["poll_points"] = np.maximum(top_n + 1 - ballot["poll_rank"], 0)
        ballot["top25_vote"] = True
        ballot["first_place_vote"] = ballot["poll_rank"] == 1
        return ballot

    def _summarize_records(self, records):
        """Internal helper for the summarize_records step."""
        if records.empty:
            return pd.DataFrame()
        model_rows = self._record_group(records, ["model", "team"])
        model_rows.insert(0, "summary_level", "model")
        aggregate_rows = self._record_group(records, ["team"])
        aggregate_rows.insert(0, "model", "ALL_MODELS")
        aggregate_rows.insert(0, "summary_level", "aggregate")
        out = pd.concat([aggregate_rows, model_rows], ignore_index=True, sort=False)
        return out.sort_values(["summary_level", "expected_wins", "team"], ascending=[True, False, True]).reset_index(drop=True)

    def _record_group(self, records, group_cols):
        """Internal helper for the record_group step."""
        rows = []
        for key, group in records.groupby(group_cols, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
            wins = pd.to_numeric(group["wins"], errors="coerce")
            losses = pd.to_numeric(group["losses"], errors="coerce")
            record_counts = (wins.astype(int).astype(str) + "-" + losses.astype(int).astype(str)).value_counts()
            most_likely = record_counts.index[0] if not record_counts.empty else ""
            row.update(
                {
                    "simulations": int(group["sim_id"].nunique()),
                    "models_used": int(group["model"].nunique()),
                    "expected_wins": float(wins.mean()),
                    "expected_losses": float(losses.mean()),
                    "median_wins": float(wins.median()),
                    "win_total_p10": float(np.percentile(wins, 10)),
                    "win_total_p90": float(np.percentile(wins, 90)),
                    "prob_10_plus_wins": float((wins >= 10).mean()),
                    "prob_11_plus_wins": float((wins >= 11).mean()),
                    "prob_undefeated": float((losses == 0).mean()),
                    "most_likely_record": most_likely,
                    "projected_record": f"{wins.mean():.1f}-{losses.mean():.1f}",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def _summarize_poll(self, ballots, final_records, top_n):
        """Internal helper for the summarize_poll step."""
        aggregate_records = final_records.loc[final_records["summary_level"] == "aggregate"].copy() if not final_records.empty else pd.DataFrame()
        if ballots.empty:
            return aggregate_records.head(top_n)
        total_ballots = int(ballots[["model", "sim_id"]].drop_duplicates().shape[0])
        rows = []
        for team, group in ballots.groupby("team", sort=False):
            rank_sum = float(pd.to_numeric(group["poll_rank"], errors="coerce").sum())
            missing = max(total_ballots - len(group), 0)
            rows.append(
                {
                    "team": team,
                    "ballots": total_ballots,
                    "top25_appearances": int(len(group)),
                    "top25_probability": float(len(group) / max(total_ballots, 1)),
                    "first_place_probability": float(pd.to_numeric(group["first_place_vote"], errors="coerce").fillna(0).sum() / max(total_ballots, 1)),
                    "average_poll_points": float(pd.to_numeric(group["poll_points"], errors="coerce").sum() / max(total_ballots, 1)),
                    "average_rank_when_ranked": float(pd.to_numeric(group["poll_rank"], errors="coerce").mean()),
                    "mean_rank_all_ballots": float((rank_sum + missing * (top_n + 1)) / max(total_ballots, 1)),
                }
            )
        poll = pd.DataFrame(rows)
        if not aggregate_records.empty:
            keep = ["team", "expected_wins", "expected_losses", "projected_record", "most_likely_record", "prob_10_plus_wins", "prob_undefeated"]
            poll = poll.merge(aggregate_records.loc[:, [c for c in keep if c in aggregate_records.columns]], on="team", how="left")
        poll = poll.sort_values(
            ["average_poll_points", "top25_probability", "mean_rank_all_ballots", "expected_wins", "team"],
            ascending=[False, False, True, False, True],
        ).reset_index(drop=True)
        poll.insert(0, "rank", np.arange(1, len(poll) + 1))
        return poll

    def _summarize_sim_accuracy(self, game_predictions):
        """Summarize retrospective simulation pick accuracy by model."""
        if game_predictions.empty or "actual_winner" not in game_predictions.columns:
            return pd.DataFrame()
        scored = game_predictions.loc[game_predictions["actual_winner"].isin(["home", "away"])].copy()
        if scored.empty:
            return pd.DataFrame()
        scored["simulated_correct"] = scored["simulated_winner"] == scored["actual_winner"]
        scored["simulated_called_upset"] = scored["simulated_winner"] != scored["predicted_favorite"]
        rows = []
        for model, group in scored.groupby("model", sort=False):
            chalk = group.loc[~group["simulated_called_upset"]]
            upset = group.loc[group["simulated_called_upset"]]
            rows.append(
                {
                    "model": model,
                    "games_scored": int(len(group)),
                    "simulations": int(group["sim_id"].nunique()),
                    "winner_accuracy": float(group["simulated_correct"].mean()),
                    "chalk_accuracy": float(chalk["simulated_correct"].mean()) if not chalk.empty else np.nan,
                    "upset_accuracy": float(upset["simulated_correct"].mean()) if not upset.empty else np.nan,
                    "called_upset_rate": float(group["simulated_called_upset"].mean()),
                    "called_upsets": int(upset.shape[0]),
                }
            )
        return pd.DataFrame(rows).sort_values("winner_accuracy", ascending=False).reset_index(drop=True)

    def _save_outputs(self, output_dir, final_records, average_poll, records, ballots, game_predictions, sim_accuracy, final_states, save_debug=False):
        """Internal helper for the save_outputs step."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        final_records_path = output_dir / "final_records.csv"
        average_poll_path = output_dir / "average_poll.csv"
        final_records.to_csv(final_records_path, index=False)
        average_poll.to_csv(average_poll_path, index=False)
        saved.extend([final_records_path, average_poll_path])
        plot_path = save_recursive_top25_plot(
            average_poll=average_poll,
            output_path=output_dir / "top25_projected_records.png",
            top_n=self._top_n(),
            logo_dir=self.config.get("figures", {}).get("logo_dir"),
            dpi=int(self.config.get("figures", {}).get("dpi", 220)),
        )
        if plot_path is not None:
            saved.append(plot_path)
        if not sim_accuracy.empty:
            sim_accuracy_path = output_dir / "simulation_accuracy.csv"
            sim_accuracy.to_csv(sim_accuracy_path, index=False)
            saved.append(sim_accuracy_path)
            from .recursive_plots import save_sim_accuracy_plot

            accuracy_plot_path = save_sim_accuracy_plot(
                sim_accuracy,
                output_dir / "simulation_winner_accuracy.png",
                dpi=int(self.config.get("figures", {}).get("dpi", 220)),
            )
            if accuracy_plot_path is not None:
                saved.append(accuracy_plot_path)
        if save_debug:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            records.to_csv(debug_dir / "simulation_records.csv", index=False)
            ballots.to_csv(debug_dir / "poll_ballots.csv", index=False)
            if not game_predictions.empty:
                game_predictions.to_csv(debug_dir / "game_predictions.csv", index=False)
            if final_states:
                pd.concat(final_states, ignore_index=False).to_csv(debug_dir / "synthetic_final_fingerprints.csv")
        return saved

    def _simulation_games(self, schedule, season, schedule_mode, as_of_week):
        """Internal helper for the simulation_games step."""
        games = schedule.loc[schedule["season"].astype(int) == int(season)].copy()
        games["is_completed"] = games["home_points"].notna() & games["away_points"].notna()
        if schedule_mode == "full_schedule":
            games["is_completed"] = False
        elif schedule_mode == "remaining_schedule":
            if as_of_week is None:
                raise ValueError("remaining_schedule requires as_of_week.")
            games["is_completed"] = games["is_completed"] & (pd.to_numeric(games["week"], errors="coerce") <= int(as_of_week))
        else:
            raise ValueError("schedule_mode must be 'full_schedule' or 'remaining_schedule'.")
        return games.reset_index(drop=True)

    def _select_model_specs(self, models, workflow):
        """Internal helper for the select_model_specs step."""
        specs = self.model_specs
        if specs is None:
            model_cfg = self.config.get("models", {})
            specs = discover_model_checkpoints(
                models_root=model_cfg.get("checkpoint_root", "models"),
                include_models=models or model_cfg.get("include_models", "all"),
                exclude_models=model_cfg.get("exclude_models", []),
            )
        elif models:
            include = {str(model).strip() for model in ([models] if isinstance(models, str) else models)}
            specs = [spec for spec in specs if spec["name"] in include]

        if not specs:
            raise ValueError("No model checkpoints were available for recursive TD Sim.")
        if workflow == "recursive_single_model":
            return [specs[0]]
        if workflow == "recursive_multi_model":
            return specs
        raise ValueError("workflow must be 'recursive_single_model' or 'recursive_multi_model'.")

    def _load_schedule(self, season):
        """Internal helper for the load_schedule step."""
        if self.schedule is not None:
            return self.schedule.copy()
        return ScheduleLoader(self.config.get("data", {})).load_schedule(season)

    def _load_fingerprint_frame(self):
        """Internal helper for the load_fingerprint_frame step."""
        if self.fingerprint_frame is not None:
            return self.fingerprint_frame.copy()
        fp_cfg = self.config.get("fingerprints", {})
        fp = Fingerprints(
            version=int(fp_cfg.get("version", 0)),
            postseason=bool(fp_cfg.get("postseason", False)),
            root=fp_cfg.get("root", PROJECT_ROOT),
            team_game_tables_dir=fp_cfg.get("team_game_tables_dir"),
        )
        return fp.frame()

    def _scheduled_teams(self, schedule, season):
        """Internal helper for the scheduled_teams step."""
        games = schedule.loc[schedule["season"].astype(int) == int(season)]
        return sorted(set(games["home_team"].astype(str)).union(set(games["away_team"].astype(str))))

    def _bootstrap_target_week0(self, fingerprint_frame, season, teams):
        """Internal helper for the bootstrap_target_week0 step."""
        fp_cfg = self.config.get("fingerprints", {})
        if not bool(fp_cfg.get("bootstrap_week0_if_missing", True)):
            return fingerprint_frame
        return append_bootstrap_week0(
            fingerprint_frame,
            season=int(season),
            teams=teams,
            seasons_back=int(fp_cfg.get("bootstrap_seasons_back", 3)),
            recency_halflife=float(fp_cfg.get("bootstrap_recency_halflife", 1.5)),
            persist=bool(fp_cfg.get("bootstrap_persist", False)),
            root=fp_cfg.get("root", PROJECT_ROOT),
            version=int(fp_cfg.get("version", 0)),
        )

    def _seed(self, model_idx, sim_id):
        """Internal helper for the seed step."""
        base = int(self.config.get("simulation", {}).get("random_seed", 42))
        return base + int(model_idx) * 1_000_003 + int(sim_id)

    def _output_dir(self, season, workflow, models, output_dir=None, shard_id=None):
        """Internal helper for the output_dir step."""
        if output_dir is not None:
            base = Path(output_dir)
            if not base.is_absolute():
                base = PROJECT_ROOT / base
        else:
            template = self.config.get("outputs", {}).get("output_dir", "data/td_sim/{season}/")
            base = Path(str(template).format(season=int(season)))
            if not base.is_absolute():
                base = PROJECT_ROOT / base
            td_sim_root = PROJECT_ROOT / "data" / "td_sim"
            if td_sim_root not in [base, *base.parents]:
                base = td_sim_root / str(season)
        model_label = models[0]["name"] if workflow == "recursive_single_model" and len(models) == 1 else "all_models"
        out = base / "recursive" / workflow.replace("recursive_", "") / _safe_name(model_label)
        if shard_id is not None:
            out = out / f"shard_{int(shard_id):04d}"
        return out

    def _top_n(self):
        """Internal helper for the top_n step."""
        return int(self.config.get("outputs", {}).get("top_n_teams", 25))


class FingerprintState:
    """Represent the FingerprintState component and its local behavior."""
    def __init__(self, frame, prior, feature_cols, mutable_cols, prior_games, final_week=16):
        """Internal helper for the init__ step."""
        self.frame = frame.copy()
        self.prior = prior.copy()
        self.feature_cols = list(feature_cols)
        self.mutable_cols = list(mutable_cols)
        self.prior_games = float(prior_games)
        self.final_week = int(final_week)
        self.teams = list(self.frame.index.astype(str))
        self.generated_totals = pd.DataFrame(0.0, index=self.teams, columns=self.mutable_cols)
        self.generated_games = pd.Series(0.0, index=self.teams)

    def copy(self):
        """Run the copy step and return its normalized result."""
        copied = FingerprintState(
            frame=self.frame.copy(),
            prior=self.prior.copy(),
            feature_cols=self.feature_cols,
            mutable_cols=self.mutable_cols,
            prior_games=self.prior_games,
            final_week=self.final_week,
        )
        copied.generated_totals = self.generated_totals.copy()
        copied.generated_games = self.generated_games.copy()
        return copied

    def update_game(self, team, game_vector):
        """Run the update_game step and return its normalized result."""
        team = str(team)
        if team not in self.frame.index:
            return
        values = pd.Series(game_vector, index=self.mutable_cols).astype(float)
        self.generated_totals.loc[team, self.mutable_cols] += values.reindex(self.mutable_cols).fillna(0.0)
        self.generated_games.loc[team] += 1.0
        games = float(self.generated_games.loc[team])
        denom = games + self.prior_games
        if denom <= 0:
            return
        updated = (self.generated_totals.loc[team, self.mutable_cols] + self.prior.loc[team, self.mutable_cols] * self.prior_games) / denom
        self.frame.loc[team, self.mutable_cols] = updated.to_numpy(dtype=float)


class FingerprintStateFactory:
    """Represent the FingerprintStateFactory component and its local behavior."""
    def __init__(self, config, fingerprint_frame):
        """Internal helper for the init__ step."""
        self.config = dict(config or {})
        self.fingerprint_frame = fingerprint_frame.copy()
        self.fp = Fingerprints(
            version=int(self.config.get("fingerprints", {}).get("version", 0)),
            root=self.config.get("fingerprints", {}).get("root", PROJECT_ROOT),
        )

    def initial_state(self, season, teams):
        """Run the initial_state step and return its normalized result."""
        feature_cols = self._feature_cols()
        prior = self._historical_prior(season=season, teams=teams, feature_cols=feature_cols)
        week0 = self._week0_state(season=season, teams=teams, feature_cols=feature_cols, fallback=prior)
        mutable_cols = [col for col in feature_cols if col.startswith(MUTABLE_PREFIXES)]
        prior_games = float(self.config.get("recursive", {}).get("historical_regression_games", 3.0))
        final_week = int(self.config.get("recursive", {}).get("final_week", 16))
        return FingerprintState(
            frame=week0,
            prior=prior,
            feature_cols=feature_cols,
            mutable_cols=mutable_cols,
            prior_games=prior_games,
            final_week=final_week,
        )

    def _feature_cols(self):
        """Internal helper for the feature_cols step."""
        x_df, _, _, _ = self.fp.split_frame(self.fingerprint_frame)
        return list(x_df.columns)

    def _historical_prior(self, season, teams, feature_cols):
        """Internal helper for the historical_prior step."""
        seasons_back = int(self.config.get("recursive", {}).get("historical_seasons", 3))
        frame = self.fingerprint_frame.copy()
        frame["keys_season"] = pd.to_numeric(frame["keys_season"], errors="coerce")
        hist = frame.loc[(frame["keys_season"] < int(season)) & (frame["keys_season"] >= int(season) - seasons_back)].copy()
        if hist.empty:
            hist = frame.loc[frame["keys_season"] < int(season)].copy()
        if hist.empty:
            hist = frame.copy()
        x_hist, _, meta_hist, _ = self.fp.split_frame(hist)
        x_hist = x_hist.reindex(columns=feature_cols)
        meta_hist = meta_hist.reset_index(drop=True)
        global_prior = x_hist.mean(numeric_only=True).reindex(feature_cols).fillna(0.0)
        by_team = x_hist.assign(keys_team=meta_hist["keys_team"].astype(str).to_numpy()).groupby("keys_team", sort=False).mean(numeric_only=True)
        rows = []
        for team in teams:
            if str(team) in by_team.index:
                row = by_team.loc[str(team)].reindex(feature_cols)
            else:
                row = global_prior
            rows.append(row.fillna(global_prior))
        return pd.DataFrame(rows, index=[str(team) for team in teams], columns=feature_cols).astype(float)

    def _week0_state(self, season, teams, feature_cols, fallback):
        """Internal helper for the week0_state step."""
        frame = self.fingerprint_frame.copy()
        mask = (pd.to_numeric(frame["keys_season"], errors="coerce") == int(season)) & (pd.to_numeric(frame["keys_week"], errors="coerce") == 0)
        week0 = frame.loc[mask].copy()
        if week0.empty:
            return fallback.copy()
        x_week0, _, meta_week0, _ = self.fp.split_frame(week0)
        x_week0 = x_week0.reindex(columns=feature_cols)
        x_week0.index = meta_week0["keys_team"].astype(str).to_numpy()
        rows = []
        for team in teams:
            if str(team) in x_week0.index:
                row = x_week0.loc[str(team)].reindex(feature_cols)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                row = row.fillna(fallback.loc[str(team)])
            else:
                row = fallback.loc[str(team)]
            rows.append(row)
        return pd.DataFrame(rows, index=[str(team) for team in teams], columns=feature_cols).astype(float)


class HistoricalGameSampler:
    """Represent the HistoricalGameSampler component and its local behavior."""
    def __init__(self, config, fingerprint_frame, feature_cols, mutable_cols):
        """Internal helper for the init__ step."""
        self.config = dict(config or {})
        self.feature_cols = list(feature_cols)
        self.mutable_cols = list(mutable_cols)
        recursive_cfg = self.config.get("recursive", {})
        self.mode = str(recursive_cfg.get("performance_sampler", "hybrid")).strip().lower()
        if self.mode not in {"historical", "knn", "hybrid"}:
            raise ValueError("recursive.performance_sampler must be one of: historical, knn, hybrid.")
        self.knn_neighbors = int(recursive_cfg.get("knn_neighbors", 40))
        self.knn_randomness = float(recursive_cfg.get("knn_randomness", 1.0))
        self.knn_margin_band = float(recursive_cfg.get("knn_margin_band", 17.5))
        self.knn_max_candidates = int(recursive_cfg.get("knn_max_candidates", 1500))
        self.hybrid_knn_weight = float(recursive_cfg.get("hybrid_knn_weight", 0.65))
        self.knn_noise = float(recursive_cfg.get("knn_noise", 0.05))
        self.neighbor_model = None
        self.fp = Fingerprints(
            version=int(self.config.get("fingerprints", {}).get("version", 0)),
            root=self.config.get("fingerprints", {}).get("root", PROJECT_ROOT),
        )
        self.stats = self._fit(fingerprint_frame)

    def sample(self, team=None, opponent=None, is_home=None, state=None, team_margin=None, rng=None, home_margin=None):
        """Run the sample step and return its normalized result."""
        rng = rng or np.random.default_rng()
        margin = float(team_margin if team_margin is not None else home_margin)
        historical = self._historical_sample(margin=margin, rng=rng)
        if self.mode == "historical":
            return historical

        knn = self._knn_sample(
            team=team,
            opponent=opponent,
            is_home=is_home,
            state=state,
            margin=margin,
            rng=rng,
        )
        if knn is None:
            return historical
        if self.mode == "knn":
            return knn

        weight = float(np.clip(self.hybrid_knn_weight, 0.0, 1.0))
        values = weight * knn.reindex(self.mutable_cols).to_numpy(dtype=float) + (1.0 - weight) * historical.reindex(self.mutable_cols).to_numpy(dtype=float)
        values = np.clip(values, self.stats["q01"], self.stats["q99"])
        return pd.Series(values, index=self.mutable_cols)

    def _historical_sample(self, margin, rng):
        """Internal helper for the historical_sample step."""
        mean = self.stats["mean"]
        slope = self.stats["slope"]
        std = self.stats["std"] * float(self.config.get("recursive", {}).get("stat_randomness", 1.0))
        values = mean + slope * (margin - float(self.stats["mean_margin"])) + rng.normal(0.0, std)
        values = np.clip(values, self.stats["q01"], self.stats["q99"])
        return pd.Series(values, index=self.mutable_cols)

    def _knn_sample(self, team, opponent, is_home, state, margin, rng):
        """Internal helper for the knn_sample step."""
        knn = self.stats.get("knn")
        if not knn or knn["query"].size == 0 or state is None:
            return None
        if team not in state.frame.index or opponent not in state.frame.index:
            return None

        query = self._query_from_state(
            team=str(team),
            opponent=str(opponent),
            is_home=bool(is_home),
            state=state,
            margin=float(margin),
            knn=knn,
        )
        n_neighbors = max(1, min(self.knn_neighbors, knn["query"].shape[0]))
        candidate_mask = knn["is_home"] == (1.0 if is_home else 0.0)
        margin_distance = np.abs(knn["margin"] - float(margin))
        margin_mask = candidate_mask & (margin_distance <= self.knn_margin_band)
        candidate_indices = np.flatnonzero(margin_mask)
        if len(candidate_indices) < n_neighbors:
            candidate_indices = np.flatnonzero(candidate_mask)
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(knn["query"].shape[0])
        if len(candidate_indices) > self.knn_max_candidates:
            narrowed = np.argsort(margin_distance[candidate_indices])[: self.knn_max_candidates]
            candidate_indices = candidate_indices[narrowed]

        diff = knn["query"][candidate_indices] - query.reshape(1, -1)
        distances = np.sqrt(np.nanmean(diff * diff, axis=1))
        order = np.argsort(distances)[:n_neighbors]
        indices = candidate_indices[order]
        distances = distances[order]

        choice_pos = self._weighted_neighbor_choice(distances=distances, rng=rng)
        values = knn["performance"][indices[choice_pos]].copy()
        if self.knn_noise > 0:
            values = values + rng.normal(0.0, self.stats["std"] * self.knn_noise, size=len(values))
        values = np.clip(values, self.stats["q01"], self.stats["q99"])
        return pd.Series(values, index=self.mutable_cols)

    def _query_from_state(self, team, opponent, is_home, state, margin, knn):
        """Internal helper for the query_from_state step."""
        team_vector = state.frame.loc[team, self.feature_cols].astype(float).to_numpy()
        opponent_vector = state.frame.loc[opponent, self.feature_cols].astype(float).to_numpy()
        query = np.concatenate([team_vector, opponent_vector, [1.0 if is_home else 0.0, float(margin)]])
        query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        return (query - knn["mean"]) / knn["std"]

    def _weighted_neighbor_choice(self, distances, rng):
        """Internal helper for the weighted_neighbor_choice step."""
        if len(distances) == 1 or self.knn_randomness <= 0:
            return 0
        distances = np.asarray(distances, dtype=float)
        scale = float(np.nanstd(distances)) * max(self.knn_randomness, 1e-6)
        if not np.isfinite(scale) or scale <= 1e-9:
            weights = np.ones(len(distances), dtype=float) / len(distances)
        else:
            weights = np.exp(-(distances - np.nanmin(distances)) / scale)
            weights = weights / weights.sum()
        return int(rng.choice(np.arange(len(distances)), p=weights))

    def _fit(self, fingerprint_frame):
        """Internal helper for the fit step."""
        frame = fingerprint_frame.copy()
        if "keys_week" in frame.columns:
            frame = frame.loc[pd.to_numeric(frame["keys_week"], errors="coerce") > 0].copy()
        x_df, _, meta_df, _ = self.fp.split_frame(frame)
        x_df = x_df.reindex(columns=self.feature_cols)
        meta_df = meta_df.reset_index(drop=True)
        games_played = pd.to_numeric(meta_df.get("games_played", pd.Series(1, index=meta_df.index)), errors="coerce").fillna(1.0)
        keys = [meta_df.get("keys_season"), meta_df.get("keys_team")]
        feature_work = x_df.reindex(columns=self.feature_cols).copy().reset_index(drop=True)
        work = x_df.loc[:, self.mutable_cols].copy().reset_index(drop=True)
        work["__season"] = pd.to_numeric(keys[0], errors="coerce").to_numpy() if keys[0] is not None else 0
        work["__team"] = keys[1].astype(str).to_numpy() if keys[1] is not None else ""
        work["__week"] = pd.to_numeric(meta_df.get("keys_week", pd.Series(0, index=meta_df.index)), errors="coerce").to_numpy()
        work["__game_id"] = meta_df.get("keys_game_id", pd.Series(pd.NA, index=meta_df.index)).to_numpy()
        work["__opponent"] = meta_df.get("keys_opponent", pd.Series(pd.NA, index=meta_df.index)).astype(str).to_numpy()
        work["__is_home"] = pd.Series(meta_df.get("game_is_home", pd.Series(False, index=meta_df.index))).astype("boolean").fillna(False).astype(float).to_numpy()
        work["__games"] = games_played.to_numpy(dtype=float)
        work["__margin"] = self._margin(frame).reset_index(drop=True).reindex(work.index).fillna(0.0)
        for col in ["__season", "__team", "__week", "__game_id", "__opponent", "__is_home", "__games", "__margin"]:
            feature_work[col] = work[col].to_numpy()
        feature_work = feature_work.sort_values(["__season", "__team", "__games"]).reset_index(drop=True)
        work = work.sort_values(["__season", "__team", "__games"]).reset_index(drop=True)

        per_game = pd.DataFrame(index=work.index, columns=self.mutable_cols, dtype=float)
        for _, group in work.groupby(["__season", "__team"], sort=False):
            prev_games = group["__games"].shift(1).fillna(0.0)
            game_delta = (group["__games"] - prev_games).replace(0.0, np.nan)
            for col in self.mutable_cols:
                current_total = pd.to_numeric(group[col], errors="coerce") * group["__games"]
                prev_total = pd.to_numeric(group[col], errors="coerce").shift(1).fillna(0.0) * prev_games
                per_game.loc[group.index, col] = (current_total - prev_total) / game_delta

        per_game = per_game.apply(pd.to_numeric, errors="coerce")
        margin = pd.to_numeric(work["__margin"], errors="coerce").fillna(0.0)
        mean_margin = float(margin.mean())
        centered_margin = margin - mean_margin
        margin_var = float(np.nanvar(centered_margin))
        mean = per_game.mean(numeric_only=True).reindex(self.mutable_cols).fillna(0.0)
        if margin_var <= 1e-12:
            slope = pd.Series(0.0, index=self.mutable_cols)
        else:
            slope = per_game.sub(mean, axis=1).mul(centered_margin, axis=0).mean(numeric_only=True) / margin_var
            slope = slope.reindex(self.mutable_cols).fillna(0.0)
        fitted_margin_effect = pd.DataFrame(
            np.outer(centered_margin.to_numpy(dtype=float), slope.to_numpy(dtype=float)),
            index=per_game.index,
            columns=self.mutable_cols,
        )
        residual = per_game.sub(mean, axis=1).sub(fitted_margin_effect, axis=1)
        std = residual.std(numeric_only=True).reindex(self.mutable_cols).replace(0.0, np.nan)
        fallback_std = per_game.std(numeric_only=True).median()
        std = std.fillna(float(fallback_std if pd.notna(fallback_std) else 1.0)).clip(lower=1e-6)
        q01 = per_game.quantile(0.01, numeric_only=True).reindex(self.mutable_cols).fillna(mean)
        q99 = per_game.quantile(0.99, numeric_only=True).reindex(self.mutable_cols).fillna(mean)
        knn = self._fit_knn_reference(work=work, feature_work=feature_work, per_game=per_game)
        return {
            "mean": mean.to_numpy(dtype=float),
            "slope": slope.to_numpy(dtype=float),
            "std": std.to_numpy(dtype=float),
            "q01": q01.to_numpy(dtype=float),
            "q99": q99.to_numpy(dtype=float),
            "mean_margin": mean_margin,
            "knn": knn,
        }

    def _fit_knn_reference(self, work, feature_work, per_game):
        """Internal helper for the fit_knn_reference step."""
        if self.mode == "historical" or work.empty:
            return {}

        pregame = feature_work.groupby(["__season", "__team"], sort=False)[self.feature_cols].shift(1)
        pregame = pregame.fillna(feature_work.loc[:, self.feature_cols])
        ref = work.loc[
            work["__opponent"].notna() & work["__game_id"].notna(),
            ["__season", "__week", "__game_id", "__team", "__opponent", "__is_home", "__margin"],
        ].copy()
        if ref.empty:
            return {}

        ref["__row"] = ref.index.to_numpy()
        lookup = ref.loc[:, ["__season", "__week", "__game_id", "__team", "__row"]].rename(
            columns={"__team": "__lookup_team", "__row": "__opponent_row"}
        )
        paired = ref.merge(
            lookup,
            left_on=["__season", "__week", "__game_id", "__opponent"],
            right_on=["__season", "__week", "__game_id", "__lookup_team"],
            how="inner",
        )
        paired = paired.loc[paired["__opponent_row"].notna()].copy()
        if paired.empty:
            return {}

        team_pre = pregame.loc[paired["__row"].to_numpy(), self.feature_cols].reset_index(drop=True)
        opponent_pre = pregame.loc[paired["__opponent_row"].to_numpy(dtype=int), self.feature_cols].reset_index(drop=True)
        query = np.column_stack(
            [
                team_pre.to_numpy(dtype=float),
                opponent_pre.to_numpy(dtype=float),
                pd.to_numeric(paired["__is_home"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                pd.to_numeric(paired["__margin"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            ]
        )
        performance = per_game.loc[paired["__row"].to_numpy(), self.mutable_cols].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(query).any(axis=1) & performance.notna().any(axis=1).to_numpy()
        query = query[valid]
        performance = performance.loc[valid].fillna(performance.mean(numeric_only=True)).fillna(0.0)
        if query.size == 0 or performance.empty:
            return {}

        query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        query_mean = np.nanmean(query, axis=0)
        query_std = np.nanstd(query, axis=0)
        query_std = np.where(query_std <= 1e-9, 1.0, query_std)
        query_scaled = (query - query_mean) / query_std
        try:
            from sklearn.neighbors import NearestNeighbors

            self.neighbor_model = NearestNeighbors(n_neighbors=min(self.knn_neighbors, len(query_scaled)), algorithm="auto")
            self.neighbor_model.fit(query_scaled)
        except Exception:
            self.neighbor_model = None

        return {
            "query": query_scaled,
            "performance": performance.to_numpy(dtype=float),
            "mean": query_mean,
            "std": query_std,
            "is_home": pd.to_numeric(paired.loc[valid, "__is_home"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "margin": pd.to_numeric(paired.loc[valid, "__margin"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        }

    def _margin(self, frame):
        """Internal helper for the margin step."""
        if "y_margin_this_week" in frame.columns:
            return pd.to_numeric(frame["y_margin_this_week"], errors="coerce")
        if "target_team_margin" in frame.columns:
            return pd.to_numeric(frame["target_team_margin"], errors="coerce")
        if {"target_points_for", "target_points_against"}.issubset(frame.columns):
            return pd.to_numeric(frame["target_points_for"], errors="coerce") - pd.to_numeric(frame["target_points_against"], errors="coerce")
        return pd.Series(0.0, index=frame.index)


def _safe_name(value):
    """Internal helper for the safe_name step."""
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "model"
