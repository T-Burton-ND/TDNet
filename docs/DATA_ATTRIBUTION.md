# Data attribution

TDNet uses data supplied by [CollegeFootballData (CFBD)](https://collegefootballdata.com/), including schedules, game outcomes, team statistics, advanced statistics, roster/returning-production context, coaching context, and—where explicitly labeled—market-line context.

Please credit CFBD whenever you publish, present, or redistribute results generated with TDNet. Suggested wording:

> Data provided by CollegeFootballData (CFBD), https://collegefootballdata.com/.

TDNet does not redistribute raw CFBD data. Users must obtain and use CFBD data under the provider's current terms, attribution requirements, and API access policy. The endpoint-level scope, cutoff rules, and completeness checks are documented in [`configs/publication/data_source_manifest.yaml`](configs/publication/data_source_manifest.yaml).

CFBD's current terms are published at
<https://collegefootballdata.com/terms>. They prohibit reselling or
redistributing API data without explicit permission and require each user to
keep their own API key private. TDNet therefore publishes fetch and processing
code, compact non-row-level provenance, and selected figures—not CFBD source
rows, derived game-level tables, or credentials.

Any software license selected for TDNet applies only to TDNet-authored source
code and does not grant rights to CFBD data, third-party marks, team logos, or
other separately identified material.
