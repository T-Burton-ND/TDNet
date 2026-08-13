# TDNet fingerprint ladder, F0--F8

This is the equation-level definition of the 2026 publication ladder as it is
implemented. A row is team (t)'s state after completed week (w), used to
predict its next game. Define
(mathcal G_{t,w}=\{g:g\text{ is completed for }t\text{ by }w\}),
(n_{t,w}=|\mathcal G_{t,w}|), and let ([a;b]) denote concatenation.
Identifiers, outcomes, and next-game lookup keys are not fingerprint features.

| Tier | Implemented mathematical definition | Information added | Team features |
|---|---|---|---:|
| **F0 — minimal strength baseline** | (phi^{(0)}_{t,w}=[n_{t,w};p_{t,s}]) | Preseason roster-talent strength prior (p) and completed-game count; no observed performance statistics. | 2 |
| **F1 — raw box scores** | (phi^{(1)}=[phi^{(0)};\bar\beta]), (\bar\beta_{t,w}=n^{-1}\sum_{g\in\mathcal G_{t,w}}\beta_{t,g}) | Season-to-date offense, defense, general, and special-teams box scores. | 34 (+32) |
| **F2 — efficiency and rates** | (phi^{(2)}=[phi^{(1)};\bar\eta]), (\bar\eta=n^{-1}\sum_g\eta_g) | PPA, success, explosiveness, havoc, and related source-defined rates. | 74 (+40) |
| **F3 — opponent adjusted** | (phi^{(3)}=[phi^{(2)};A]), (A_j=[\operatorname{mean}(a_j);\operatorname{mean}_3(a_j);\operatorname{EWM}_{.45}(a_j)]) | Pregame-Elo contextual residuals and schedule-strength summaries. | 148 (+74) |
| **F4 — contextual football profile** | (phi^{(4)}=[phi^{(3)};q]) | Returning production, prior coaching history, travel distance, and time-zone displacement. This is the representation used by the current locked margin roster. | 159 (+11) |
| **F5 — temporal dynamics** | (phi^{(5)}=[phi^{(4)};T]) | Last-game, last-3, half-life-3 EWM, recent-minus-season trend, and five-game volatility for 12 core performance measures. | 219 (+60) |
| **F6 — complete market-free** | (phi^{(6)}=[phi^{(5)};G]) | Eight causal schedule-network coordinates: Colley, PageRank, schedule strength, win/loss quality, experience, opponent diversity, and next-opponent edge. | 227 (+8) |
| **F7 — market-only benchmark** | (phi^{(7)}=m=[\text{open spread};\text{close spread};\text{total};\text{win probability}]) | Betting-market information only. This is not on the nested football ladder. | 4 |
| **F8 — complete plus market** | (phi^{(8)}=[phi^{(6)};phi^{(7)}]) | F6 plus the four market variables, isolating incremental market value. | 231 (+4 vs. F6) |

The scientific nesting claim is
(phi^{(0)}\subset\phi^{(1)}\subset\cdots\subset\phi^{(6)}).
F7 is an external benchmark, while F8 supports the paired comparison F6 versus
F8. The exact row-level grid is
[FINGERPRINT_FEATURE_MATRIX.csv](FINGERPRINT_FEATURE_MATRIX.csv).

## F3 opponent adjustment

For home and away pregame Elo ratings (R_h,R_a), home margin (m), and home
indicator (H), the sequential update is

\[
E_h=\left(1+10^{-((R_h-R_a)+55)/400}\right)^{-1},\qquad
\Delta R=20c(m)(S_h-E_h),
\]

where (S_h\in\{0,.5,1\}) and
(c(m)=\min(2.5,\max(.75,\log(|m|+1)))). For non-margin statistic (j),

\[
a_{t,g,j}=x_{t,g,j}-\mu_j+0.10(R_{o(g)}/s_R)s_j.
\]

Margin uses
(a_{t,g,m}=m_{t,g}-[7(R_t-R_{o(g)})/s_R+2.5H_{t,g}]).
All ratings and rolling summaries use completed games available at the row
cutoff.

## F5 temporal dynamics

The source columns are season-to-date means. When the completed-game count
advances from (n_{w-1}) to (n_w), the latest game contribution is recovered
without future data:

\[
x_{t,w}=\frac{n_w\bar x_{t,w}-n_{w-1}\bar x_{t,w-1}}
{n_w-n_{w-1}}.
\]

Bye rows add no observation and carry the prior temporal state. For each of 12
core measures, F5 records

\[
T_j=[x_w;\operatorname{mean}(x_{w-2:w});e_w;
\operatorname{mean}(x_{w-2:w})-\bar x_w;
\operatorname{sd}(x_{w-4:w})],
\]

with (e_w=\alpha x_w+(1-\alpha)e_{w-1}) and
(alpha=1-2^{-1/3}), corresponding to a three-game half-life.

## F6 schedule graph

For the season graph containing games completed through (w), the Colley
system is (Cr=b):

\[
C_{ii}=2+n_i,\quad C_{ij}=-n_{ij},\quad
b_i=1+\tfrac12(w_i-l_i).
\]

The directed PageRank graph sends a loser-to-winner edge with weight
(1+\min(|m|,35)/14); games within seven points add a reverse edge of weight
0.25. PageRank uses damping 0.85 and is z-scored within season-week. Schedule
strength and win/loss quality are means of current Colley ratings over faced,
beaten, and lost-to opponents. The next-opponent edge is (r_t-r_o), where
both ratings come from the same through-(w) graph.

## Authoritative implementation

- Ladder membership: `configs/features/feature_ladders.yaml`
- Feature registry: `configs/features/feature_registry.yaml`
- F5/F6 builder: `src/gridiron_ml/fingerprints/ladder.py`
- F0/F1/F2 aggregation: `src/gridiron_ml/fingerprints/builders/v0.py`
- F3 Elo context: `src/gridiron_ml/experiments/opponent_adjusted.py`
- Exact schemas: `docs/publication_2026/feature_manifests/F0.json` through `F8.json`
