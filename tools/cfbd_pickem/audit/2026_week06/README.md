# Week 6 CFBD contest submission

The authenticated CFBD slate contained 57 Week 6 regular-season games, all with no existing pick. The frozen TDNet package has 58 scheduled FBS-involving games; CFBD's active contest slate did not include Marshall at Coastal Carolina (game ID `401869943`). The contest export therefore uses all 57 authenticated contest games and requires a full-slate match.

The export was prepared from the frozen 33-model consensus with no rounding. For game `401858254` (Florida State at Louisville), the published near-zero positive signed home margin (`0.209127545808444`) disagreed with its `pred_winner` label. The contest source corrects only that label to Louisville, as required by the signed-margin convention; the numerical pick is unchanged. The export audit records the source and payload hashes, game mappings, and pre-kickoff check.
