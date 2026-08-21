#!/usr/bin/env python3
"""Generate public, source-only introductory assets for the 2026 TDNet blog.

The script intentionally uses only compact canonical metadata.  It never reads
CFBD rows, predictions, checkpoints, or generated evaluation tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "publication/2026/figures/intro"
SCRIPT = "scripts/publication/generate_intro_blog_assets.py"
SOURCE_PATHS = [
    "FREEZE_MANIFEST.json",
    "docs/PUBLIC_ARTIFACT_POLICY.md",
    "docs/publication_2026/FINGERPRINT_REGISTRY.yaml",
    "docs/publication_2026/FINGERPRINT_EQUATIONS.md",
    "docs/publication_2026/MODEL_ARTIFACT_RELEASE.json",
    "docs/publication_2026/MODEL_ARTIFACT_SHA256SUMS",
    "docs/publication_2026/MODEL_ARTIFACT_STATUS.json",
    "docs/publication_2026/REGENERATION_2025_STATUS.json",
    "docs/publication_2026/ROSTER_REGISTRY.json",
    "configs/features/feature_ladders.yaml",
    "configs/publication/canonical_2026_protocol.yaml",
    "src/gridiron_ml/cli/publication/run_2026_preseason_release.py",
]

NAVY, VIOLET, PINK, BLUE, MINT, PALE, SLATE, WHITE = (
    "#11214F", "#6A37C8", "#FF5FA2", "#1EA7FF", "#4ED8BD", "#E6E9ED", "#3A4450", "#FFFFFF"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def box(ax, x, y, w, h, title, body, color=WHITE, edge=VIOLET, title_color=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035", facecolor=color, edgecolor=edge, linewidth=1.8))
    ax.text(x + w / 2, y + h * .64, title, ha="center", va="center", color=title_color, fontsize=13, weight="bold")
    ax.text(x + w / 2, y + h * .32, body, ha="center", va="center", color=SLATE, fontsize=9.5, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=SLATE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.7, color=color))


def save(fig, stem: str, timestamp: str, full_sha: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png, svg = OUT / f"{stem}.png", OUT / f"{stem}.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(svg, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    for path in (png, svg):
        (OUT / f"{path.name}.json").write_text(json.dumps({
            "source_paths": SOURCE_PATHS,
            "generated_at_utc": timestamp,
            "git_sha": full_sha,
            "rendered_file": path.name,
            "rendered_file_sha256": sha256(path),
            "script": SCRIPT,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def team_to_fingerprint(timestamp, full_sha):
    fig, ax = plt.subplots(figsize=(13, 4.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xs = [.03, .225, .42, .615, .81]
    entries = [
        ("Team + week", "A team at a point\nin its season", BLUE),
        ("Completed information", "Only what was known\nby that week", MINT),
        ("Time-dependent\nfingerprint", "A week-specific snapshot\n—not the team itself", VIOLET),
        ("Matchup\nrepresentation", "Home and away\nfingerprints together", PINK),
        ("Model output", "Expected margin +\nwin probability", NAVY),
    ]
    for i, (title, body, edge) in enumerate(entries):
        box(ax, xs[i], .29, .155, .42, title, body, edge=edge)
        if i: arrow(ax, xs[i - 1] + .157, .5, xs[i] - .006, .5)
    ax.text(.5, .91, "HOW A TEAM BECOMES A FINGERPRINT", ha="center", color=NAVY, fontsize=20, weight="bold")
    ax.text(.5, .085, "The fingerprint updates with completed history; it is a snapshot used for the next matchup.", ha="center", color=SLATE, fontsize=11)
    save(fig, "team_to_fingerprint", timestamp, full_sha)


def ladder(registry, timestamp, full_sha):
    labels = [(f["id"], f["display_name"], ", ".join(f["feature_families"])) for f in registry["fingerprints"]]
    pd.DataFrame(labels, columns=["fingerprint", "label", "cumulative_information_families"]).to_csv(OUT / "fingerprint_ladder.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 7.4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .95, "THE TDNET FINGERPRINT LADDER", ha="center", color=NAVY, fontsize=20, weight="bold")
    ax.text(.5, .905, "Primary market-free ladder", ha="center", color=SLATE, fontsize=11)
    primary = [x for x in labels if x[0] <= "F6"]
    for i, (fid, title, families) in enumerate(primary):
        y = .80 - i * .095
        box(ax, .12, y, .76, .07, f"{fid}  ·  {title}", families.replace(", ", "  •  "), edge=VIOLET if fid != "F6" else PINK)
        if i < 6: arrow(ax, .5, y - .005, .5, y - .022, VIOLET)
    ax.add_patch(FancyBboxPatch((.08, .065), .84, .13, boxstyle="round,pad=0.015,rounding_size=.025", facecolor="#F3F5F8", edgecolor=SLATE, linewidth=1.2))
    ax.text(.5, .164, "MARKET COMPARISONS — separate from the confirmatory ladder", ha="center", va="center", color=NAVY, fontsize=11, weight="bold")
    ax.text(.285, .105, "F7 · Market-only benchmark\nmarket", ha="center", va="center", color=SLATE, fontsize=9.5)
    ax.text(.715, .105, "F8 · Complete fingerprint + market\nF6 + market", ha="center", va="center", color=SLATE, fontsize=9.5)
    save(fig, "fingerprint_ladder", timestamp, full_sha)


def science_vs_saturday(timestamp, full_sha):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5)); fig.subplots_adjust(wspace=.08)
    for ax, title, color in zip(axes, ["SCIENTIFIC PANEL", "OPERATIONAL WIDE F6 ROSTER"], [VIOLET, PINK]):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((.03, .04), .94, .90, boxstyle="round,pad=.02,rounding_size=.03", facecolor="#F8F9FB", edgecolor=color, linewidth=2.2))
        ax.text(.5, .87, title, ha="center", color=NAVY, fontsize=16, weight="bold")
    axes[0].text(.5, .72, "6 architectures × F0–F8", ha="center", color=VIOLET, fontsize=13, weight="bold")
    axes[0].text(.5, .59, "54 margin cells\n42 F0–F6 market-free, prospective/poll-eligible\n12 F7/F8 market-bearing, comparison-only", ha="center", va="center", color=SLATE, fontsize=11, linespacing=1.55)
    axes[0].text(.5, .27, "Question\nHow does information complexity\ninteract with model complexity?", ha="center", va="center", color=NAVY, fontsize=12, weight="bold", linespacing=1.4)
    axes[1].text(.5, .72, "Corrected F6 fingerprint", ha="center", color=PINK, fontsize=13, weight="bold")
    axes[1].text(.5, .58, "34 learned estimators + 2 equal-weight ensembles\n36 operational margin cells\n33 automated Top-25 voting members\n+ one separately reported owner/manual ballot", ha="center", va="center", color=SLATE, fontsize=11, linespacing=1.55)
    axes[1].text(.5, .27, "Purpose\nWeekly predictions, consensus,\nand Top-25 voting", ha="center", va="center", color=NAVY, fontsize=12, weight="bold", linespacing=1.4)
    fig.text(.5, .025, "The manual ballot is reported separately and does not alter model consensus or performance metrics.", ha="center", color=SLATE, fontsize=9.5)
    save(fig, "science_vs_saturday", timestamp, full_sha)


def freeze_card(freeze, status, release, timestamp, full_sha):
    archives = release["archives"]
    card = {
        "training_through": 2025, "prospective_season": 2026, "scope": "FBS vs FBS regular season",
        "official_weekly_prediction_deadline": "Thursday 23:59 America/New_York", "scientific_checkpoints": 54,
        "market_free_prospective_scientific_cells": 42, "operational_f6_cells": 36,
        "market_tiers_permitted_in_official_predictions_or_polls": False,
        "calibration_status": status["scientific"]["calibration_statuses"][0], "git_sha": full_sha,
        "archive_release_status": release["status"], "public_artifact": release.get("public_release"), "archives": archives,
        "freeze_manifest_status": freeze["status"],
        "note": "The concept DOI identifies the latest Zenodo version; the version DOI identifies this exact v1.1 archive record.",
    }
    (OUT / "freeze_card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 8.4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((.03, .04), .94, .91, boxstyle="round,pad=.02,rounding_size=.03", facecolor="#F8F9FB", edgecolor=NAVY, linewidth=2.2))
    ax.text(.07, .88, "FROZEN BEFORE KICKOFF", color=NAVY, fontsize=21, weight="bold")
    ax.text(.07, .83, "2026 prospective contract · candidate-not-final manifest", color=SLATE, fontsize=10.5)
    left = "Training through: 2025\nProspective season: 2026\nScope: FBS vs FBS regular season\nDeadline: Thursday 23:59 America/New_York\nCalibration: cross-fitted OOF through 2025"
    right = "Scientific checkpoints: 54\nMarket-free prospective cells: 42\nOperational F6 cells: 36\nMarket tiers in predictions/polls: NO\nGit: " + full_sha[:12]
    ax.text(.08, .65, left, color=NAVY, fontsize=11, va="top", linespacing=1.65)
    ax.text(.55, .65, right, color=NAVY, fontsize=11, va="top", linespacing=1.65)
    ax.text(.07, .35, "MODEL ARCHIVES", color=PINK, fontsize=12, weight="bold")
    y = .30
    for archive in archives:
        ax.text(.08, y, f"{archive['filename']}\n{archive['size_bytes']:,} bytes · SHA-256 {archive['sha256']}", color=SLATE, fontsize=7.3, va="top", linespacing=1.45); y -= .135
    public = release.get("public_release", {})
    ax.text(.07, .055, f"Open Zenodo artifact: concept DOI {public.get('concept_doi', 'not recorded')} · v{public.get('version', '?')} DOI {public.get('version_doi', 'not recorded')}", color=SLATE, fontsize=8.7)
    save(fig, "frozen_before_kickoff", timestamp, full_sha)


def complexity(timestamp, full_sha):
    values = [("F0", 2), ("F1", 34), ("F2", 74), ("F3", 148), ("F4", 159), ("F5", 219), ("F6", 227)]
    fig, ax = plt.subplots(figsize=(11, 5.3)); ax.bar([x[0] for x in values], [x[1] for x in values], color=[BLUE, BLUE, MINT, VIOLET, VIOLET, PINK, NAVY])
    for i, (_, n) in enumerate(values): ax.text(i, n + 5, str(n), ha="center", color=NAVY, weight="bold")
    ax.set_title("MARKET-FREE FINGERPRINT COMPLEXITY", color=NAVY, weight="bold", fontsize=18, pad=14)
    ax.set_ylabel("Designed team features", color=NAVY); ax.set_xlabel("Nested market-free ladder", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=PALE); ax.set_axisbelow(True)
    ax.text(.01, -.22, "Counts are canonical team-feature definitions; they do not count home/away or matchup expansion. F7/F8 are separate market controls.", transform=ax.transAxes, color=SLATE, fontsize=9.5)
    save(fig, "fingerprint_complexity", timestamp, full_sha)


def docs():
    (OUT / "post2_expected_assets.md").write_text("""# Post 2 expected assets\n\n`run_2026_preseason_release` is fail-closed and must wait for an official 25-row AP Top 25. Coaches Poll data is never substituted. Once that gate passes, it creates the following reviewable groups under `publication/2026/preseason/` for both the market-free scientific roster and the wide F6 roster:\n\n- `tdnet_preseason_top25/`: TDNet preseason Top 25, ballot outputs, and AP comparison inputs.\n- `week_01_predictions/`: Week 1 prediction table/graphics, all-model predictions, closest projected games, model-failure reports, and the blog package.\n- AP comparison and model-ballot disagreement material produced by the frozen roster-poll and weekly-blog builders.\n- TDNet-versus-market disagreement outputs only where emitted by the canonical weekly-blog workflow; they must remain labeled as comparisons, not a claim that TDNet beats Vegas.\n\nNo preseason predictions, AP comparison, or substitute Coaches-Poll figure has been fabricated here.\n""", encoding="utf-8")
    (OUT / "historical_context_for_blog.md").write_text("""# Historical context for the blog\n\n## Recommended context fact\n\n- **Claim:** TDNet maintains a genuine retrospective 2025 holdout package trained through 2024, separate from a through-2025 pipeline rehearsal.\n- **Source artifact:** `docs/publication_2026/REGENERATION_2025_STATUS.json`.\n- **Training boundary:** 2024.\n- **Evaluation season:** 2025.\n- **Sample size:** 17 poll weeks and 16 prediction weeks; the status artifact does not provide a game count.\n- **Metric:** None proposed. This is an evaluation-design fact, not a performance result.\n- **Apples-to-apples:** Not a TDNet-versus-market comparison.\n- **Caveat:** Do not turn this into a performance claim without a complete, reviewed metric artifact and a matched comparison definition.\n\n## Recommendation\n\nLeave generic historical-performance or “TDNet vs Vegas” graphics out of Post 1. The canonical status materials establish a clean holdout boundary but do not supply a single preselected, apples-to-apples market-comparison metric suitable for a public claim.\n""", encoding="utf-8")
    (OUT / "BLOG_FIGURE_CAPTIONS.md").write_text("""# Blog figure captions\n\n- **How a Team Becomes a Fingerprint.** TDNet treats a team as a moving snapshot: only completed information enters the next matchup.\n- **The TDNet Fingerprint Ladder.** F0–F6 add football information in a market-free sequence; F7 and F8 are separate market comparisons.\n- **Two TDNets: Science and Saturday.** One roster tests the design. The other powers the weekly football product.\n- **Frozen Before Kickoff.** The 2026 contract fixes the models, scope, deadline, and exclusions before the season supplies outcomes.\n- **Fingerprint Complexity.** The market-free ladder adds designed team features without treating a larger representation as automatically better.\n""", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    timestamp, full_sha = datetime.now(timezone.utc).isoformat(), git_sha()
    registry = yaml.safe_load((ROOT / "docs/publication_2026/FINGERPRINT_REGISTRY.yaml").read_text())
    ladder_cfg = yaml.safe_load((ROOT / "configs/features/feature_ladders.yaml").read_text())
    if ladder_cfg["cumulative_paths"]["primary"] != [f"F{i}" for i in range(7)] or registry["fingerprints"][7]["id"] != "F7":
        raise ValueError("Canonical fingerprint sources disagree; refusing to render the ladder.")
    freeze = json.loads((ROOT / "FREEZE_MANIFEST.json").read_text())
    status = json.loads((ROOT / "docs/publication_2026/MODEL_ARTIFACT_STATUS.json").read_text())
    release = json.loads((ROOT / "docs/publication_2026/MODEL_ARTIFACT_RELEASE.json").read_text())
    team_to_fingerprint(timestamp, full_sha); ladder(registry, timestamp, full_sha)
    science_vs_saturday(timestamp, full_sha); freeze_card(freeze, status, release, timestamp, full_sha)
    complexity(timestamp, full_sha); docs()
    (OUT / "roster_summary.json").write_text(json.dumps({"scientific_margin_cells": 54, "market_free_prospective_poll_eligible_cells": 42, "market_bearing_comparison_only_cells": 12, "wide_f6_learned_estimators": 34, "wide_f6_equal_weight_ensembles": 2, "wide_f6_operational_margin_cells": 36, "automated_top25_voting_members": 33, "owner_manual_ballot": "separately reported; excluded from model consensus and performance metrics"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
