"""Human-readable row and training semantics used in docs and metadata."""

V0_ROW_SEMANTICS = (
    "v0 keys_week=N represents team state after week N has completed. "
    "Week 0 is a preseason/bootstrap state."
)
Y_NEXT_MARGIN_SEMANTICS = (
    "y_next_margin is the default training target: the team's margin in its next scheduled game."
)
Y_MARGIN_THIS_WEEK_WARNING = (
    "y_margin_this_week is same-row completed-game information and is unsafe with postgame/current-week features."
)
MARKET_EVALUATION_ONLY_WARNING = (
    "market_* columns are evaluation context by default and must stay out of model training features unless explicitly opted in."
)
