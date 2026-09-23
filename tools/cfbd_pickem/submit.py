from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tools.cfbd_pickem.common import (
    API_BASE_URL,
    ValidationError,
    api_request,
    load_api_key,
    sha256_file,
    utc_now,
    write_json,
)


CONFIRMATION = "SUBMIT_APPROVED_CFBD_PAYLOAD"


def _payload_picks(payload_bytes: bytes) -> dict[int, Decimal]:
    try:
        payload = json.loads(payload_bytes.decode("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Submission payload is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("picks"), list):
        raise ValidationError("Submission payload must contain a picks list.")

    picks: dict[int, Decimal] = {}
    for item in payload["picks"]:
        if not isinstance(item, dict) or set(item) != {"gameId", "pick"}:
            raise ValidationError("Each pick must contain only gameId and pick.")
        game_id = item["gameId"]
        if not isinstance(game_id, int) or isinstance(game_id, bool):
            raise ValidationError("Each gameId must be an integer.")
        if game_id in picks:
            raise ValidationError(f"Duplicate gameId in payload: {game_id}")
        try:
            pick = Decimal(str(item["pick"]))
        except InvalidOperation as exc:
            raise ValidationError(f"Game {game_id}: pick is not numeric.") from exc
        if not pick.is_finite():
            raise ValidationError(f"Game {game_id}: pick must be finite.")
        picks[game_id] = pick
    if not picks:
        raise ValidationError("Submission payload has no picks.")
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a previously audited CFBD payload with explicit confirmation."
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--confirm-submit", required=True)
    args = parser.parse_args()

    if args.confirm_submit != CONFIRMATION:
        raise ValidationError(
            f"Refusing to submit: --confirm-submit must equal {CONFIRMATION!r}."
        )

    payload_path = args.payload.resolve()
    audit_path = args.audit.resolve()
    payload_bytes = payload_path.read_bytes()
    payload_sha256 = sha256_file(payload_path)
    if payload_sha256 != args.expected_payload_sha256:
        raise ValidationError("Payload SHA-256 does not match the approved checksum.")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "prepared_not_submitted" or audit.get("submission") is not None:
        raise ValidationError("Audit record is not in prepared_not_submitted state.")
    recorded_sha256 = audit["artifacts"]["api_payload"]["sha256"]
    if recorded_sha256 != payload_sha256:
        raise ValidationError("Payload SHA-256 does not match the audit record.")

    payload_picks = _payload_picks(payload_bytes)
    approved_games = {
        int(game["game_id"]): game for game in audit.get("games", [])
    }
    if set(approved_games) != set(payload_picks):
        raise ValidationError("Audit game IDs do not exactly match the payload.")
    now = datetime.now(timezone.utc)
    at_or_past_kickoff = []
    for game_id, game in approved_games.items():
        kickoff = datetime.fromisoformat(
            str(game["kickoff_utc"]).replace("Z", "+00:00")
        )
        if kickoff <= now:
            at_or_past_kickoff.append(game_id)
    if at_or_past_kickoff:
        raise ValidationError(
            "Approved games are at or past kickoff: "
            + ", ".join(map(str, sorted(at_or_past_kickoff)))
        )

    api_key = load_api_key()
    get_status, current_games = api_request("GET", "/api/picks", api_key)
    if get_status != 200 or not isinstance(current_games, list):
        raise ValidationError("Pre-submit GET /api/picks did not return a game list.")
    current_by_id = {int(game["id"]): game for game in current_games}
    unavailable = sorted(set(payload_picks) - set(current_by_id))
    if unavailable:
        raise ValidationError(
            "Payload games are no longer available for submission: "
            + ", ".join(str(game_id) for game_id in unavailable)
        )

    identity_changes = []
    pick_state_changes = []
    for game_id, approved in approved_games.items():
        current = current_by_id[game_id]
        current_identity = (
            int(current.get("week", -1)),
            str(current.get("awayTeam", "")),
            str(current.get("homeTeam", "")),
        )
        approved_identity = (
            int(approved["provider_week"]),
            str(approved["away_team"]),
            str(approved["home_team"]),
        )
        if current_identity != approved_identity:
            identity_changes.append(
                {
                    "game_id": game_id,
                    "approved": approved_identity,
                    "current": current_identity,
                }
            )
        if current.get("pick") != approved.get("existing_cfbd_pick"):
            pick_state_changes.append(game_id)
    if identity_changes:
        raise ValidationError(
            "CFBD game identities/orientations changed after approval: "
            + repr(identity_changes)
        )
    if pick_state_changes:
        raise ValidationError(
            "CFBD pick state changed after approval for games: "
            + ", ".join(map(str, sorted(pick_state_changes)))
        )

    if audit.get("selection", {}).get("full_authenticated_slate_required"):
        unapproved_live_games = sorted(set(current_by_id) - set(payload_picks))
        if unapproved_live_games:
            raise ValidationError(
                "The authenticated CFBD slate changed after review; refusing to "
                "submit unreviewed/omitted games: "
                + ", ".join(map(str, unapproved_live_games))
            )

    outside_existing = {
        game_id: game.get("pick")
        for game_id, game in current_by_id.items()
        if game_id not in payload_picks and game.get("pick") is not None
    }
    if outside_existing:
        raise ValidationError(
            "Bulk submission would remove existing picks omitted from this payload; "
            f"refusing to continue. Existing outside game IDs: {sorted(outside_existing)}"
        )

    submitted_at = utc_now()
    post_status, response_body = api_request(
        "POST", "/api/picks", api_key, body=payload_bytes
    )
    readback_status, readback_games = api_request("GET", "/api/picks", api_key)
    if readback_status != 200 or not isinstance(readback_games, list):
        raise ValidationError("Submission returned, but readback GET /api/picks failed.")
    readback_by_id = {int(game["id"]): game for game in readback_games}
    mismatches: dict[int, dict[str, str]] = {}
    for game_id, expected in payload_picks.items():
        actual_raw = readback_by_id.get(game_id, {}).get("pick")
        actual = Decimal(str(actual_raw)) if actual_raw is not None else None
        if actual != expected:
            mismatches[game_id] = {"expected": str(expected), "actual": str(actual)}

    response_path = payload_path.with_name(
        payload_path.name.replace("_submission.json", "_submission_response.json")
    )
    response_record: dict[str, Any] = {
        "submitted_at_utc": submitted_at,
        "endpoint": f"{API_BASE_URL}/api/picks",
        "payload_sha256": payload_sha256,
        "post_http_status": post_status,
        "post_response": response_body,
        "readback_http_status": readback_status,
        "readback_verified": not mismatches,
        "readback_mismatches": mismatches,
        "submission_id": (
            response_body.get("id") if isinstance(response_body, dict) else None
        ),
        "api_key_fingerprint": hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16],
        "entry_name": audit.get("entry_name"),
    }
    write_json(response_path, response_record)

    audit["status"] = "submitted" if not mismatches else "submitted_readback_mismatch"
    audit["submission"] = {
        **response_record,
        "response_record_path": str(response_path),
        "response_record_sha256": sha256_file(response_path),
    }
    write_json(audit_path, audit)
    if mismatches:
        raise ValidationError(
            "CFBD accepted the POST, but readback did not match; inspect the response record."
        )
    print(
        f"Submitted and verified {len(payload_picks)} picks; "
        f"response record: {response_path}"
    )


if __name__ == "__main__":
    main()
