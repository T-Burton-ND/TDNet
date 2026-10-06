---
type: reference
up: "[[TDNet-Overview]]"
tags: [publication, operations, 2026]
---

# Weekly Publication Workflow

TDNet’s documented 2026 operation refreshes and inspects one weekly data snapshot, prepares a deadline-frozen prediction bundle, then scores that bundle after games complete.

## Monday: refresh, inspect, approve

The Monday runner refreshes the local CFBD cache, team-game table, v0 fingerprint, and opponent-adjusted fingerprints. It polls only the configured in-season endpoints and records a hash/schema/count record per endpoint. Missing endpoints, empty required responses, or missing required fields make the snapshot uncertified and block approval. The owner reviews the inspection report and approves the snapshot explicitly.

## Tuesday: freeze and prepare

The Tuesday workflow requires Monday approval and a certified snapshot. It prepares one immutable prediction bundle for the week, including locked-roster predictions, Top-25 poll artifacts, comparison material, and draft-only social assets. The workflow does not send an X post. Thursday 23:59 America/New_York remains the default deadline, but an owner-selected earlier weekday cutoff is supported when the slate has an early kickoff. The recorded cutoff must strictly precede the earliest scheduled FBS-involving kickoff; the bundle records both Eastern and UTC time. Prediction bytes remain fixed after the recorded deadline. The October 2026 Week 6 run used Tuesday 19:00 Eastern because the first game started at 20:00. See [2026 Week 5 Postgame and Week 6 Pregame](Weekly-Publication-2026-W05-W06), [Confirmatory Protocol 2026](Confirmatory-Protocol-2026), [README Source](Source-README), and [Weekly Operations Source](Source-Weekly-Operations).

## Sunday: score without refetching

After outcome/statistic completeness is confirmed, the Sunday workflow scores the immutable bundle from the cached results. It is network-free and does not make another CFBD request. It creates retrospective metrics, comparison tables, figures, and draft-only blog/social assets without changing pregame bytes or the Top-25 snapshot.

## Standing figure scope

The owner expects complete pregame and postgame figure packages for both the operational roster and the scientific roster. Keep the market-free F0–F6 scientific cohort and the full F0–F8 cohort as separately labeled outputs; do not collapse them into one scientific roster. For pregame, cover every game in the frozen weekly slate for each applicable roster and generate its prediction, ballot/ranking, and publication figures. For postgame, score every completed game from the frozen predictions, then refresh the per-game results and season-to-date figures through the latest fully scored week. Include the MAE, winner accuracy, upset recall, and Brier views where the underlying outputs support them, and show the matching Vegas baseline on comparative figures. Use the repository palette with warm parchment figure backgrounds.

Commit and push the generated figures together with their manifests and documentation. Keep pregame predictions frozen after the recorded cutoff. A week’s postgame package is complete only after the games have finished and final outcomes have been checked; pending games must not be represented with invented results. External social publishing remains draft-only unless separately approved. See [2026 Week 5 Postgame and Week 6 Pregame](Weekly-Publication-2026-W05-W06) for the October 2026 package example.

## Notebooks and storage

The recurring notebooks named in the weekly notebook guide are:

- `tdnet_weekly_predictions.ipynb` — inspect the upcoming schedule and create the game-level prediction table.
- `tdnet_manual_top25_poll.ipynb` — edit the owner ballot and merge it with learned-model ballots.
- `tdnet_sunday_results.ipynb` — score the immutable bundle and render the weekly review.
- `tdnet_data_and_model_reproduction.ipynb` — reader-run reproduction using the reader’s own CFBD key.

Raw tables, predictions, ballots, and manifests stay in ignored local data paths unless a specific release rule allows a curated artifact. Raw CFBD data are not redistributed. The scientific roster and owner ballot boundaries are described in [Model and Poll Surfaces](Model-and-Poll-Surfaces).

## Scheduling and external publication

Do not install cron until the owner chooses exact run times and the environment has `CFBD_API_KEY`. Social output is draft-only unless separately configured and explicitly approved. The pipeline does not publish externally on its own.

## Sources

- `source-archive/docs/publication_2026/WEEKLY_OPERATIONS.md.txt`
- `source-archive/docs/publication_2026/THREE_NOTEBOOK_WORKFLOW.md.txt`
- `source-archive/docs/publication_2026/CONFIRMATORY_PROTOCOL.md.txt`
- [Weekly Operations Source](Source-Weekly-Operations)
- [Confirmatory Protocol Source](Source-Confirmatory-Protocol)

## See also

[TDNet Overview](TDNet-Overview) · [Package Architecture](Package-Architecture) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [2026 Week 4 and 5 Publication](Weekly-Publication-2026-W04-W05) · [Source-README](Source-README) · [Experiment Program Recovery](Experiment-Program-Recovery) · [Release and Operations History Source](Source-Release-and-Operations-History)
