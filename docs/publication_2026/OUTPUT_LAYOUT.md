# Publication output layout

Current generated publication packages live beneath the external durable
artifact root. Prediction and poll outputs are not split between a
`data/` tree, a figures tree, manual-poll tree, or winner/margin aliases.

Each regenerated season-week uses exactly this layout:

```text
publication/<season>/week_<NN>/
├── wide_margin/
│   ├── predictions_all_games.csv
│   ├── predictions_top25_games.csv
│   ├── poll.csv
│   ├── predictions_all_games.png
│   ├── predictions_top25_games.png
│   ├── poll.png
│   └── full_ballots.png
└── scientific/
    ├── predictions_all_games.csv
    ├── predictions_top25_games.csv
    ├── poll.csv
    ├── predictions_all_games.png
    ├── predictions_top25_games.png
    ├── poll.png
    └── full_ballots.png
```

Each week-season directory also gets one manifest recording the generation
timestamp, roster label, model-bundle hash, fit cutoff, data snapshot hashes,
and model count. CSVs and PNGs are co-located; no duplicate private/public or
objective-specific copies are produced.

`wide_margin` is the corrected-F6 operational margin roster. `scientific` is
the six-model panel evaluated at every F0–F8 fingerprint. The weekly writer
requires the full 54-cell inventory but emits predictions and polls from only
the 42 market-free F0–F6 cells. F7/F8 never enter official outputs.

`full_ballots.png` shows every model ballot. Scientific retrospective ballot
grids constructed from held-out game predictions are visualization-only implied
rankings, not an additional preregistered poll objective.

The frozen model bundles remain separately under the external durable artifact
root; they are not weekly prediction outputs.

The weekly writer must reject any output path outside this contract. Existing
cleared output trees are not compatibility inputs and should not be recreated.
