"""Audit generated ballots for ordering, duplication, and point-cap errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def audit_ballot(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("ballot_model", "")), []).append(row)

    findings = []
    for model, ballot in sorted(groups.items()):
        ranked = sorted(ballot, key=lambda row: int(float(row["ballot_rank"])))
        top = ranked[:25]
        ranks = [int(float(row["ballot_rank"])) for row in top]
        teams = [str(row.get("keys_team", "")) for row in top]
        points = [float(row.get("poll_points", "nan")) for row in top]
        expected_points = list(range(25, 25 - len(top), -1))
        short_slate = len(top) < 25
        finding = {
            "model": model,
            "rows": len(ranked),
            "top25_rows": len(top),
            "top25_ranks_valid": ranks == list(range(1, len(top) + 1)),
            "top25_unique_teams": len(set(teams)) == len(teams),
            "alphabetical_top25": len(top) >= 10 and teams == sorted(teams, key=str.casefold),
            "points_follow_25_to_1": points == expected_points,
            "constant_points": len(set(points)) <= 1,
            "post_top25_rows_are_zero_point": all(
                float(row.get("poll_points", "nan")) == 0
                for row in ranked[25:]
            ),
            "short_slate": short_slate,
        }
        if finding["alphabetical_top25"]:
            findings.append({"model": model, "red_flag": "alphabetical_top25"})
        if not finding["top25_unique_teams"]:
            findings.append({"model": model, "red_flag": "duplicate_top25_team"})
        if not finding["points_follow_25_to_1"]:
            findings.append({"model": model, "red_flag": "invalid_poll_point_cap"})
        if finding["constant_points"]:
            findings.append({"model": model, "red_flag": "constant_points"})
        finding["team_order_preview"] = teams[:5]
        finding["team_order_tail"] = teams[-5:]
        findings.append({"model": model, "checks": finding})

    return {
        "path": str(path),
        "rows": len(rows),
        "ballot_count": len(groups),
        "red_flags": [item for item in findings if "red_flag" in item],
        "ballots": [item["checks"] for item in findings if "checks" in item],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballot", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "audits": [audit_ballot(path) for path in args.ballot],
        "ranking_cap_policy": (
            "TDEval.poll ranks with uncapped fitted/raw scores; poll_points are "
            "then capped by the declared Top-25 25-to-1 rule."
        ),
        "raw_score_persistence": "not persisted in ballot CSV; source audit required",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
