# Weekly social media packs

After the weekly pregame publication is complete, build a pack with:

```bash
python3 scripts/publication/build_social_media.py --season 2026 --week 5
```

Replace the season and week for each Monday. The command reads published tables and figures from `publication/<season>/week_<WW>/pre_game/` and writes five to nine graphics into `social_media/<season>/week_<WW>/`. Each PNG has a same-named Markdown file containing a short caption, hashtags, image alt text, and hashed source files. The week's `README.md` lists every option.

The ratings comparison and rank-disagreement graphics are included when their published table and both figures are present. The generator stops if only part of that ratings set exists. It requires a completed pregame package and never fetches live scores or changes prediction data.

Matchup cards show each game's Vegas spread captured at publication in parentheses beneath the market favorite. That line can differ from the TDNet predicted winner or margin. A missing published line stops generation rather than leaving a game unlabeled.

On Monday, review the pack, choose five or six posts, and schedule game previews before their kickoffs. Rankings and ratings can stay relevant longer; do not schedule a prediction after the game. The captions deliberately omit URLs, so you can paste them directly into X and add a link only when useful.
