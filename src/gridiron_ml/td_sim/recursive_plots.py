"""src.gridiron_ml.td_sim.recursive_plots.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def save_recursive_top25_plot(average_poll, output_path, top_n=25, logo_dir=None, title=None, dpi=220):
    """Run the save_recursive_top25_plot step and return its normalized result."""
    if average_poll is None or average_poll.empty:
        return None

    import matplotlib.pyplot as plt
    from gridiron_ml.td_run.poll_viz import draw_team_logo

    table = average_poll.head(int(top_n)).copy().reset_index(drop=True)
    colors = _palette()
    logo_dir = Path(logo_dir) if logo_dir is not None else PROJECT_ROOT / "data" / "meta" / "logos" / "by_team"
    manifest = _logo_manifest()

    fig_height = max(8.0, 0.42 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(table) + 1)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_facecolor(colors["panel"])
    fig.patch.set_facecolor("white")

    title = title or f"TD Sim Projected Top {top_n}"
    ax.text(0.0, 0.25, title, fontsize=18, fontweight="bold", color=colors["text"], va="bottom")
    headers = [
        ("#", 0.03),
        ("Team", 0.14),
        ("Projected Record", 0.53),
        ("Expected Wins", 0.72),
        ("Top 25", 0.86),
        ("No. 1", 0.97),
    ]
    for label, x in headers:
        ax.text(x, 0.95, label, fontsize=9.5, fontweight="bold", color=colors["spine"], va="center")
    ax.hlines(1.18, 0.0, 1.0, color=colors["grid"], linewidth=1.0)

    for idx, row in table.iterrows():
        y = idx + 1.65
        if idx % 2 == 0:
            ax.axhspan(y - 0.28, y + 0.28, color=colors["panel"], alpha=0.85)
        ax.text(0.03, y, str(int(row.get("rank", idx + 1))), fontsize=10.5, fontweight="bold", color=colors["text"], va="center")

        team = str(row.get("team", ""))
        logo_path = _resolve_logo(team, logo_dir, manifest)
        if logo_path is not None:
            # Use the shared renderer so transparent/opaque canvas padding is
            # cropped before sizing and every mark stays inside the Team cell.
            draw_team_logo(ax, logo_path, 0.105, y, target_px=18)
        else:
            ax.text(0.105, y, _abbr(team), fontsize=7.5, color=colors["text"], ha="center", va="center")

        ax.text(0.14, y, team, fontsize=10.5, color=colors["text"], va="center")
        ax.text(0.53, y, str(row.get("projected_record", "")), fontsize=10.5, color=colors["text"], va="center")
        ax.text(0.76, y, f"{float(row.get('expected_wins', 0.0)):.2f}", fontsize=10.5, color=colors["text"], va="center", ha="right")
        ax.text(0.89, y, f"{100.0 * float(row.get('top25_probability', 0.0)):.1f}%", fontsize=10.5, color=colors["text"], va="center", ha="right")
        ax.text(0.99, y, f"{100.0 * float(row.get('first_place_probability', 0.0)):.1f}%", fontsize=10.5, color=colors["text"], va="center", ha="right")
        ax.hlines(y + 0.31, 0.0, 1.0, color=colors["grid"], linewidth=0.55)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_sim_accuracy_plot(sim_accuracy, output_path, title=None, dpi=220):
    """Save a retrospective TD Sim winner/chalk/upset accuracy chart."""
    if sim_accuracy is None or sim_accuracy.empty:
        return None

    import matplotlib.pyplot as plt

    metrics = [
        ("winner_accuracy", "Winner"),
        ("chalk_accuracy", "Chalk Calls"),
        ("upset_accuracy", "Upset Calls"),
    ]
    available = [(col, label) for col, label in metrics if col in sim_accuracy.columns]
    if not available:
        return None

    try:
        from gridiron_ml.td_run.season_vs_vegas import source_style
    except Exception:
        source_style = None

    table = sim_accuracy.copy()
    table["winner_accuracy"] = pd.to_numeric(table.get("winner_accuracy"), errors="coerce")
    table = table.sort_values("winner_accuracy", ascending=False).reset_index(drop=True)
    x = range(len(table))
    width = min(0.75 / max(len(available), 1), 0.22)
    colors = _palette()

    fig, ax = plt.subplots(figsize=(max(9.0, 0.75 * len(table)), 4.9))
    for metric_idx, (col, label) in enumerate(available):
        offsets = [value + (metric_idx - (len(available) - 1) / 2) * width for value in x]
        bars = []
        for idx, (_, row) in enumerate(table.iterrows()):
            model = str(row.get("model", "model"))
            style = source_style(model, {}, idx) if source_style is not None else {"color": "#1EA7FF"}
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            bars.append(
                ax.bar(
                    offsets[idx],
                    value,
                    width=width,
                    color=style["color"],
                    alpha=0.90 if metric_idx == 0 else 0.62,
                    hatch="" if metric_idx == 0 else "/" if metric_idx == 1 else "\\",
                    label=label if idx == 0 else None,
                )[0]
            )
        for bar in bars:
            value = bar.get_height()
            if pd.notna(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_title(title or "TD Sim Retrospective Winner Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x))
    ax.set_xticklabels(table["model"].astype(str), rotation=30, ha="right")
    ax.grid(axis="y", color=colors["grid"], alpha=0.55)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _palette():
    """Internal helper for the palette step."""
    palette_path = PROJECT_ROOT / "style" / "color_palettes" / "tdnet_palette.csv"
    colors = {
        "panel": "#F3F5F8",
        "grid": "#DDE2E7",
        "spine": "#3A4450",
        "text": "#0D0D0D",
    }
    if not palette_path.exists():
        return colors
    palette = pd.read_csv(palette_path)
    by_name = {str(row["name"]).strip(): str(row["hex"]).strip() for _, row in palette.iterrows()}
    colors["panel"] = by_name.get("Mist Panel", colors["panel"])
    colors["grid"] = by_name.get("Fog Silver", colors["grid"])
    colors["spine"] = by_name.get("Slate Line", colors["spine"])
    colors["text"] = by_name.get("Flat Black", colors["text"])
    return colors


def _logo_manifest():
    """Internal helper for the logo_manifest step."""
    path = PROJECT_ROOT / "data" / "meta" / "logos" / "logo_name_manifest.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out = {}
    for _, row in frame.iterrows():
        for key in [row.get("school"), row.get("slug")]:
            if pd.isna(key):
                continue
            team_file = row.get("team_file")
            if pd.isna(team_file):
                continue
            out[_slug(key)] = PROJECT_ROOT / str(team_file)
    return out


def _resolve_logo(team, logo_dir, manifest):
    """Internal helper for the resolve_logo step."""
    slug = _slug(team)
    manifest_path = manifest.get(slug)
    if manifest_path is not None and manifest_path.exists():
        return manifest_path
    for candidate in [slug, slug.replace("_and_", "_"), slug.replace("_", "")]:
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            path = logo_dir / f"{candidate}{ext}"
            if path.exists():
                return path
    return None


def _slug(value):
    """Internal helper for the slug step."""
    text = str(value).strip().lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _abbr(team, max_len=10):
    """Internal helper for the abbr step."""
    words = str(team).split()
    if not words:
        return ""
    if len(words) == 1:
        return words[0][:max_len]
    return "".join(word[0] for word in words[:4]).upper()
