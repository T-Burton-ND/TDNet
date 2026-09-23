# CFBD Model Pick'em export

This isolated utility prepares the corrected-F6 wide-margin TDNet entry for
CFBD Model Pick'em. It never trains, refits, recalibrates, refreshes data, or
submits as part of generation/export.

The already-published eight-game Week 0 consensus remains authoritative. To
cover the complete contest slate, `generate_slate` runs inference only with the
same 36 frozen checkpoints and actual schedule-driven preseason fingerprint
construction. It refuses to write unless the model-level opening slate
reproduces within `1e-12`, then restores the exact published decimal strings for
those eight games.

## CFBD contract verified on 2026-08-28

- Web form, CSV import, and API submission are supported.
- CSV columns are `id,home,away,predicted`.
- The bulk API body is `{"picks":[{"gameId": INTEGER,"pick": NUMBER}]}` and
  is sent to `POST /api/picks` at
  `https://predictionsapi.collegefootballdata.com`.
- API authentication is `Authorization: Bearer YOUR_API_KEY`. An API key is
  available after signing into the site with Twitter/X or Reddit.
- `pick` is projected away score minus projected home score: positive selects
  the away team and negative selects the home team.
- Team IDs and neutral-site flags are not submission fields. The exporter still
  requires exact game ID, team-name, and orientation matches against an
  authenticated `GET /api/picks` snapshot.
- The site says picks must be submitted before kickoff. It does not publish a
  more detailed locking policy in its public page or client contract.
- Bulk CSV/API import is replacement-oriented: omitted existing picks are
  removed. The final export therefore contains the complete authenticated
  53-game slate.

## Reproducible preparation

Run in the `gridiron` conda environment from the repository root. The key file
is ignored by Git and should remain mode `0600`.

Fetch the current slate with a read-only request:

```bash
CFBD_PICKEM_API_KEY="$(tr -d '\r\n' < .Contest_API.key)" \
PYTHONPATH=src conda run -n gridiron python \
  -m tools.cfbd_pickem.fetch_slate \
  --output tools/cfbd_pickem/audit/2026_current_slate/cfbd_contest_slate.json
```

Generate the full corrected-F6 source after the frozen Week 0 reproduction
gate:

```bash
MPLCONFIGDIR=/tmp/tdnet_mplconfig PYTHONPATH=src \
conda run -n gridiron python -m tools.cfbd_pickem.generate_slate \
  --cfbd-slate tools/cfbd_pickem/audit/2026_current_slate/cfbd_contest_slate.json \
  --output-dir tools/cfbd_pickem/audit/2026_current_slate
```

Create the exact CSV and API payload without submitting:

```bash
PYTHONPATH=src conda run -n gridiron python -m tools.cfbd_pickem.export \
  --source tools/cfbd_pickem/audit/2026_current_slate/2026_current_slate_margin_consensus_source.csv \
  --cfbd-slate tools/cfbd_pickem/audit/2026_current_slate/cfbd_contest_slate.json \
  --output-dir tools/cfbd_pickem/audit/2026_current_slate \
  --season 2026 --reader-week 0 \
  --provider-week 1 --provider-week 2 \
  --season-type regular \
  --entry-name "TDNet corrected-F6 margin consensus" \
  --roster "corrected-F6 wide-margin operational roster" \
  --rounding none --require-full-slate
```

CFBD reverses the neutral-site Notre Dame/Wisconsin home-away orientation from
the local CFBD schedule. Generation predicts the local schedule orientation,
then explicitly negates that game's home margin when expressing it in the
contest orientation. Reversed non-neutral games are rejected.

## Approval-gated submission

The only command that performs a POST is below. Do not run it without explicit
owner approval of the payload and its SHA-256:

```bash
CFBD_PICKEM_API_KEY="$(tr -d '\r\n' < .Contest_API.key)" \
PYTHONPATH=src conda run -n gridiron python \
  -m tools.cfbd_pickem.submit \
  --payload tools/cfbd_pickem/audit/2026_current_slate/2026_current_slate_tdnet_corrected_f6_margin_consensus_cfbd_submission.json \
  --audit tools/cfbd_pickem/audit/2026_current_slate/2026_current_slate_tdnet_corrected_f6_margin_consensus_cfbd_audit.json \
  --expected-payload-sha256 c84f9a41208dc060f658ad3ac8a615ae5e672ea31b2ea04ce7f78da706308779 \
  --confirm-submit SUBMIT_APPROVED_CFBD_PAYLOAD
```

Immediately before POST, the submitter fetches the live slate again, refuses
new/unapproved games or outside existing picks, sends the exact approved bytes,
reads every pick back, and writes non-secret response/audit metadata. Never
commit an API key, cookie, or token.
