"""Checked F09–F12 canonical writes and keyed model/matchup reads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd

from .nextgen_contract import assert_pair_closed, assert_temporal_feature_rows, validate_feature_manifest
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    atomic_json, load_authoritative_schedule, sha256_file,
)
from gridiron_ml.td_run.matchups.builder import MatchupBuilder


REPO_ROOT = Path(__file__).resolve().parents[3]
SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_BOUNDARY_TOKEN = object()


def canonical_family_path(root: Path, generation: str, family: str) -> Path:
    if generation not in {"F09", "F10", "F11", "F12"} or not SAFE_NAME.fullmatch(family):
        raise ValueError("Invalid nextgen feature family identity")
    return Path(root) / "feature_families" / generation / family / "canonical.parquet"


def _manifest(manifest_path: Path, columns: tuple[str, ...], generation: str) -> str:
    records = json.loads(Path(manifest_path).read_text())
    schema = json.loads((REPO_ROOT / "configs/experiments/nextgen_feature_manifest_schema_v1.json").read_text())
    validate_feature_manifest(records, schema)
    if any(record["generation"] != generation for record in records):
        raise ValueError("Feature manifest generation disagrees with canonical artifact")
    if len(columns) != len(set(columns)) or set(columns) != {record["name"] for record in records}:
        raise ValueError("Canonical columns must exactly match the feature manifest")
    assert_pair_closed(columns, records)
    kinds = {record["static_or_dynamic"] for record in records}
    if len(kinds) != 1:
        raise ValueError("One canonical family artifact must have one temporal kind")
    return "static_week0" if kinds == {"static_preseason"} else "dynamic"


def _schedule_and_freeze(root: Path, *, create_freeze: bool) -> tuple[pd.DataFrame, str, dict]:
    inventory = json.loads((REPO_ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json").read_text())
    schedule, schedule_hash = load_authoritative_schedule(root, inventory)
    cutoffs: dict[str, dict[str, str]] = {}
    for game in schedule.itertuples(index=False):
        season = str(int(game.season))
        kickoff = pd.to_datetime(game.start_date, utc=True)
        for team in (str(game.home_team), str(game.away_team)):
            prior = cutoffs.setdefault(season, {}).get(team)
            if prior is None or kickoff < pd.Timestamp(prior):
                cutoffs[season][team] = kickoff.isoformat()
    expected = {"schema_version": 1, "schedule_sha256": schedule_hash,
                "rule": "first_regular_game_kickoff_exclusive", "team_season_cutoffs": cutoffs}
    freeze_path = Path(root) / "results/preflight/week0_freeze_v1.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != expected:
            raise ValueError("Week-0 freeze differs from the authoritative Stage-A schedule")
    elif create_freeze:
        atomic_json(freeze_path, expected)
    else:
        raise ValueError("Missing durable Week-0 freeze artifact")
    return schedule, schedule_hash, cutoffs


def _validate_rows(frame: pd.DataFrame, columns: tuple[str, ...], kind: str,
                   schedule: pd.DataFrame, cutoffs: dict) -> None:
    assert_temporal_feature_rows(frame, columns)
    if not frame.feature_kind.eq(kind).all():
        raise ValueError("Feature rows disagree with manifest temporal kind")
    by_id = {int(game.id): game for game in schedule.itertuples(index=False)}
    keys = list(zip(frame.target_game_id.astype(int), frame.team.astype(str)))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate target-game/team feature rows")
    target_groups: dict[int, list[str]] = {}
    for row in frame.itertuples(index=False):
        game_id = int(row.target_game_id)
        target = by_id.get(game_id)
        if (target is None or target.home_classification.lower() != "fbs"
                or target.away_classification.lower() != "fbs"):
            raise ValueError("Target game is absent from authoritative FBS schedule")
        if row.team not in {target.home_team, target.away_team} or int(row.season) != int(target.season):
            raise ValueError("Target team or season disagrees with authoritative schedule")
        target_time = pd.to_datetime(target.start_date, utc=True)
        if pd.to_datetime(row.target_start_utc, utc=True) != target_time:
            raise ValueError("Target kickoff disagrees with authoritative schedule")
        target_groups.setdefault(game_id, []).append(row.team)
        if row.feature_kind == "dynamic":
            source = by_id.get(int(row.latest_source_game_id))
            if (source is None or row.team not in {source.home_team, source.away_team}
                    or source.season_type.lower() != "regular" or not source.completed
                    or pd.to_datetime(row.latest_source_game_utc, utc=True) !=
                       pd.to_datetime(source.start_date, utc=True)
                    or pd.to_datetime(source.start_date, utc=True) >= target_time):
                raise ValueError("Source game provenance disagrees with authoritative schedule")
        else:
            cutoff = cutoffs.get(str(int(row.season)), {}).get(row.team)
            if cutoff is None or pd.to_datetime(row.feature_available_utc, utc=True) >= pd.Timestamp(cutoff):
                raise ValueError("Static feature was not available by the frozen Week-0 cutoff")
    for game_id, teams in target_groups.items():
        game = by_id[game_id]
        if len(teams) != 2 or set(teams) != {game.home_team, game.away_team}:
            raise ValueError("Target game lacks exactly one row for each scheduled team")
    static = frame.loc[frame.feature_kind.eq("static_week0")]
    for _, group in static.groupby(["season", "team"]):
        stable = list(columns) + ["feature_available_utc", "static_availability_documentation"]
        if group.loc[:, stable].nunique(dropna=False).gt(1).any():
            raise ValueError("Static Week-0 value changes within a team-season")


def _static_hashes(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for (season, team), group in frame.loc[frame.feature_kind.eq("static_week0")].groupby(["season", "team"]):
        row = group.iloc[0]
        payload = {name: row[name] for name in columns}
        payload["feature_available_utc"] = row.feature_available_utc
        result[f"{int(season)}:{team}"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return result


def write_canonical_feature_family(root: Path, generation: str, family: str,
                                   frame: pd.DataFrame, feature_columns: Iterable[str],
                                   manifest_path: Path) -> Path:
    """Validate schedule, Week-0 freeze, and manifest before an atomic write."""
    columns = tuple(feature_columns)
    kind = _manifest(manifest_path, columns, generation)
    schedule, schedule_hash, cutoffs = _schedule_and_freeze(root, create_freeze=True)
    _validate_rows(frame, columns, kind, schedule, cutoffs)
    path = canonical_family_path(root, generation, family)
    sidecar = path.with_suffix(".provenance.json")
    static_hashes = _static_hashes(frame, columns)
    if path.exists() or sidecar.exists():
        if not path.exists() or not sidecar.exists():
            raise ValueError("Existing canonical artifact lacks its provenance sidecar")
        prior = json.loads(sidecar.read_text())
        if prior.get("data_sha256") != sha256_file(path):
            raise ValueError("Existing canonical artifact fails its content hash")
        if prior.get("manifest_sha256") != sha256_file(Path(manifest_path)):
            raise ValueError("Existing canonical artifact uses a different feature manifest")
        if prior.get("schedule_sha256") != schedule_hash:
            raise ValueError("Existing canonical artifact uses a different authoritative schedule")
        for key, value in static_hashes.items():
            old = prior.get("static_team_season_hashes", {}).get(key)
            if old is not None and old != value:
                raise ValueError("Frozen static team-season value changed on rewrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".canonical_", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temp_name, index=False)
        data_hash = sha256_file(Path(temp_name))
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    atomic_json(sidecar, {"manifest_sha256": sha256_file(Path(manifest_path)),
                          "data_sha256": data_hash, "schedule_sha256": schedule_hash,
                          "feature_columns": columns, "static_team_season_hashes": static_hashes})
    return path


class NextgenFeatureBuilder(ABC):
    """Future F09–F12 builders supply rows; materialization remains checked."""

    generation: str
    family: str
    feature_columns: tuple[str, ...]
    manifest_path: Path

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "materialize" in cls.__dict__:
            raise TypeError("Nextgen builders cannot bypass canonical validation")

    @abstractmethod
    def build_frame(self) -> pd.DataFrame:
        """Return both target-team rows with schedule-backed provenance."""

    def materialize(self, root: Path) -> Path:
        return write_canonical_feature_family(
            root, self.generation, self.family, self.build_frame(),
            self.feature_columns, self.manifest_path)


class NextgenModelBoundary:
    """Expose only artifact- and schedule-validated rows to matchup, fit, SHAP."""

    def __init__(self, frame: pd.DataFrame, columns: tuple[str, ...],
                 schedule: pd.DataFrame, *, _token=None):
        if _token is not _BOUNDARY_TOKEN:
            raise ValueError("Load nextgen model inputs through from_canonical")
        self.feature_columns = columns
        self._frame = frame.copy()
        self._schedule = {int(game.id): game for game in schedule.itertuples(index=False)}
        self._rows = {(int(row.target_game_id), str(row.team)): row
                      for row in frame.itertuples(index=False)}

    @classmethod
    def from_canonical(cls, root: Path, generation: str, family: str,
                       manifest_path: Path) -> "NextgenModelBoundary":
        path = canonical_family_path(root, generation, family)
        sidecar = path.with_suffix(".provenance.json")
        if not path.exists() or not sidecar.exists():
            raise ValueError("Canonical artifact or provenance sidecar missing")
        provenance = json.loads(sidecar.read_text())
        columns = tuple(provenance["feature_columns"])
        kind = _manifest(manifest_path, columns, generation)
        if provenance.get("manifest_sha256") != sha256_file(Path(manifest_path)):
            raise ValueError("Canonical feature manifest hash mismatch")
        if provenance.get("data_sha256") != sha256_file(path):
            raise ValueError("Canonical feature data hash mismatch")
        schedule, schedule_hash, cutoffs = _schedule_and_freeze(root, create_freeze=False)
        if provenance.get("schedule_sha256") != schedule_hash:
            raise ValueError("Canonical artifact uses a different authoritative schedule")
        frame = pd.read_parquet(path)
        _validate_rows(frame, columns, kind, schedule, cutoffs)
        if provenance.get("static_team_season_hashes") != _static_hashes(frame, columns):
            raise ValueError("Canonical static snapshot hash mismatch")
        return cls(frame, columns, schedule, _token=_BOUNDARY_TOKEN)

    def matchup(self, home_key: tuple[int, str], away_key: tuple[int, str],
                *, representation: str = "diff") -> pd.DataFrame:
        """Resolve `(target_game_id, team)` keys against the fresh schedule."""
        if len(home_key) != 2 or len(away_key) != 2 or int(home_key[0]) != int(away_key[0]):
            raise ValueError("Home and away keys must identify the same target game")
        game_id = int(home_key[0])
        game = self._schedule.get(game_id)
        if (game is None or home_key != (game_id, game.home_team)
                or away_key != (game_id, game.away_team)):
            raise ValueError("Matchup keys disagree with the authoritative schedule")
        home = self._rows.get(home_key)
        away = self._rows.get(away_key)
        if home is None or away is None:
            raise ValueError("Home or away target-team row is missing")
        home_values = pd.DataFrame([{name: getattr(home, name) for name in self.feature_columns}])
        away_values = pd.DataFrame([{name: getattr(away, name) for name in self.feature_columns}])
        return MatchupBuilder(representation=representation).build(home_values, away_values)

    def for_fit(self) -> pd.DataFrame:
        return self._frame.set_index(["target_game_id", "team"])[list(self.feature_columns)].sort_index()

    def for_shap(self) -> pd.DataFrame:
        return self.for_fit().copy()
