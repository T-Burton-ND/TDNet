#!/usr/bin/env python3
"""Prepare an approval-gated X post package; never sends externally."""

from argparse import ArgumentParser
from pathlib import Path
import json


def main():
    parser = ArgumentParser()
    parser.add_argument("--weekly-output", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    output = args.weekly_output.resolve()
    package = output / "x_post_package"
    package.mkdir(parents=True, exist_ok=True)
    # Weekly publication outputs are roster-scoped.  X drafts use the
    # operational wide-margin roster; the scientific roster has its own
    # publication bundle and is not silently mixed into this draft.
    wide = output / "wide_margin"
    candidates = [
        ("all_model_predictions", wide / "predictions_all_games.png"),
        ("top25_games", wide / "predictions_top25_games.png"),
        ("tdnet_poll", wide / "poll.png"),
    ]
    media = [{"role": role, "path": str(path), "exists": path.exists()} for role, path in candidates]
    text = (
        f"TDNet {args.season} Week {args.week} predictions are frozen.\n\n"
        "Wide-margin consensus and the Top 25 matchup slate are now available. "
        "Every prediction traces to the preseason model freeze.\n\n"
        "Model output—not betting advice. #CollegeFootball #CFB"
    )
    (package / "post.txt").write_text(text + "\n")
    (package / "alt_text.md").write_text(
        "# X media alt text\n\n"
        "1. Wide-margin consensus table listing weekly college-football matchups, TDNet picks, predicted margins, and win probabilities.\n\n"
        "2. Top 25 matchup cards with team logos and TDNet predicted winners and margins.\n\n"
        "3. TDNet Top 25 poll table with team logos and model-consensus ranks.\n"
    )
    manifest = {"send_status": "draft_only_requires_explicit_approval", "post_text": "post.txt", "alt_text": "alt_text.md", "media": media}
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(package)

if __name__ == "__main__":
    main()
