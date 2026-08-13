# F6-C compression study

`F6-C` is a derived compression of F6, not a new information tier and not
“F6.5.” Exact deployed models are labeled `F6-CN`, where `N` is the actual
source-feature count after enforcing reviewed matchup-pair closure.

## Leakage-safe ranking and selection

For outer fold (k), source features are ranked using only corrected-F6 SHAP
results from folds earlier than (k). No attribution from fold (k)'s test
season may rank its predictors. Raw offense/defense fields that form a reviewed
matchup are an atomic group: selecting either member adds its counterpart.
Therefore target budgets and actual counts are both recorded.

The candidate target budgets are 10, 15, 20, 25, 35, 50, 75, 100, 150, and
227. This yields at most

\[
9\ \text{eligible folds}\times 2\ \text{objectives}\times
6\ \text{models}\times 10\ \text{budgets}=1080\ \text{fits}.
\]

Fold 1 uses the prespecified target count 25 because no prior compression
performance exists. At each later fold, the selected count is the smallest
candidate within one standard error of the best candidate on earlier eligible
folds. Its performance on the current fold is the sequentially nested estimate.
After all folds, each objective/model deployment count is selected by the same
smallest-within-one-SE rule across all eligible historical folds.

The six requested metrics are retained for every candidate and for the
sequentially selected F6-C path: MAE, upset recall, chalk recall, winner
accuracy, Brier score, and ATS accuracy. F6-C enters the all-fingerprint
heatmaps only through the sequentially selected path; a globally SHAP-ranked
top-N score is exploratory and cannot substitute for it.

Machine-readable settings are in
`configs/publication/f6_compression_study.yaml`.
