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

The Tuesday workflow requires Monday approval and a certified snapshot. It prepares one immutable prediction bundle for the week, including locked-roster predictions, Top-25 poll artifacts, comparison material, and draft-only social assets. The workflow does not send an X post. Prediction bytes remain fixed after the protocol-version-2 deadline; the canonical deadline is Thursday 23:59 America/New_York with UTC also recorded. See [Confirmatory Protocol 2026](Confirmatory-Protocol-2026), [README Source](Source-README), and [Weekly Operations Source](Source-Weekly-Operations).

## Sunday: score without refetching

After outcome/statistic completeness is confirmed, the Sunday workflow scores the immutable bundle from the cached results. It is network-free and does not make another CFBD request. It creates retrospective metrics, comparison tables, figures, and draft-only blog/social assets without changing pregame bytes or the Top-25 snapshot.

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

- `../docs/publication_2026/WEEKLY_OPERATIONS.md`
- `../docs/publication_2026/THREE_NOTEBOOK_WORKFLOW.md`
- `../docs/publication_2026/CONFIRMATORY_PROTOCOL.md`
- [Weekly Operations Source](Source-Weekly-Operations)
- [Confirmatory Protocol Source](Source-Confirmatory-Protocol)

## See also

[TDNet Overview](TDNet-Overview) · [Package Architecture](Package-Architecture) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Source-README](Source-README)
