# F20 CFBD Week 7 source audit (2026)

Checked October 23, 2026. The F20 feature contract is a market margin plus
the projected home margin from TeamRankings, Massey Ratings, ESPN FPI, FEI,
and Sagarin Pred. The historical 2026 tracker CSV still ends at CFBD Week 6:
its own Week 7 label is offset by one. The tracker [Last Week results page](https://www.thepredictiontracker.com/ncaaresults.php?orderby=wpct+desc&type=3&year=26)
shows aggregate results through October 18 for 59 Week 7 games, but gives no
game-level expert margins. The [season CSV](https://www.thepredictiontracker.com/ncaaarchive.html)
is unchanged from the archived copy used in the current replay (SHA-256
`8dbbeed2551c5f53858af8c19a95b666a089fb1ca35b6c8a396caf8fc8df961f`,
329 rows). Its five-expert coverage is 321 of 329 rows; the other eight have
four experts.

| Frozen expert input | Individual Week 7 source found | Import status |
| --- | --- | --- |
| FEI | [BCF Toys 2026 game projections](https://bcftoys.com/2026-gp), Week 7 section, 59 game rows. The page says projections use ratings from the previous week. | Candidate source with a full weekly table; not yet matched to CFBD IDs or the tracker FEI column. |
| Massey Ratings | [October 15 FBS game page](https://masseyratings.com/cf/fbs/games?dt=20261015) contains pregame predicted scores for a subset of the Week 7 slate. | Candidate source. Its dated page shows 38 games, including games already final by October 15, so coverage and the per-game pregame date need checking before import. |
| TeamRankings | [Rutgers at Maryland Week 7 recap](https://www.teamrankings.com/college-football/matchup/scarlet-knights-terrapins-week-7-2026) retains a final score projection. | Individual game found; the complete slate and equivalence to tracker `lineteamrank` are unverified. |
| ESPN FPI | [Tracker weekly results](https://www.thepredictiontracker.com/ncaaresults.php?orderby=wpct+desc&type=3&year=26) reports 59 graded FPI games. | Aggregate only. No complete set of original per-game projected margins found. |
| Sagarin Pred. | [Tracker weekly results](https://www.thepredictiontracker.com/ncaaresults.php?orderby=wpct+desc&type=3&year=26) reports Sagarin system totals. | Aggregate only. The current Sagarin ratings are after Week 7 and cannot reconstruct the pregame tracker margin. |

The current F20 replay and charts continue to mark CFBD Week 7 unavailable.
The weekly source timing assumption the owner accepted does not supply missing
numeric inputs. Do not fill frozen feature columns with another site's model,
current postgame ratings, a reconstructed point spread, or final scores. A
later game-level tracker archive or complete, reconciled individual source
collection can produce a separate immutable Week 7 replay and chart version.
The chart builder now detects Week 7 F20 replay rows and requires all 59
games in the existing Week 7 common cohort before reporting a Week 7 F20
score. Its current 1,050-row scorecard retains the same numeric values after
this change; the complete-Week-7 branch was checked with disposable synthetic
data in `/tmp`, not published as a result.
Until then, the existing 304-game paired Weeks 1–6 comparison is the last
valid F20 historical score. F20's six M1–M5/M10 fits remain frozen; the
external expert forecasts are features for each fit, not TDNet F9–F19 model
predictions.
