# Three-notebook 2026 production workflow

These are the only notebooks intended for the recurring publication run:

1. `publication/notebooks/weekly/tdnet_sunday_results.ipynb` — score the latest immutable
   prediction bundle against completed games and display prediction-versus-
   results tables and scorecards.
2. `publication/notebooks/weekly/tdnet_manual_top25_poll.ipynb` — drag the manual ballot;
   the shared poll combines it with every learned-model ballot in the weekly
   inventory, including KNN, and writes the Top-25 and full ballot-grid
   figures.
3. `publication/notebooks/weekly/tdnet_weekly_predictions.ipynb` — inspect the upcoming schedule
   and generate the new game-level prediction table and AP comparison figure.

Raw tables, predictions, ballots, and manifests belong under the ignored
`data/publication/<season>/` tree. Curated PNG/SVG figures belong under
`publication/<season>/figures/`. The end-to-end reader workflow lives in
`publication/notebooks/reproduction/`; research and legacy notebooks are not
part of the publication procedure.
