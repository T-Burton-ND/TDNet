"""Represent one football season as teams (nodes) and games (edges)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import networkx as nx
import numpy as np
import pandas as pd
from gridiron_ml.td_run.poll_viz import logo_slug_candidates, load_team_logo_image


CONFERENCE_COLORS = {
    "ACC": "#1F77B4",
    "American Athletic": "#17BECF",
    "Big 12": "#FF7F0E",
    "Big Ten": "#9467BD",
    "Conference USA": "#2CA02C",
    "FBS Independents": "#7F7F7F",
    "Independent/Other": "#7F7F7F",
    "Mid-American": "#D62728",
    "Mountain West": "#E377C2",
    "Pac-12": "#8C564B",
    "SEC": "#BCBD22",
    "Sun Belt": "#393B79",
}
NON_CONFERENCE_EDGE_COLOR = "#1F2937"


def build_season_graph(games: pd.DataFrame, *, season: int, completed_only=False) -> nx.MultiDiGraph:
    """Build a directed multigraph; each edge points from away team to home team."""
    frame = games[pd.to_numeric(games["season"], errors="coerce").eq(int(season))].copy()
    if completed_only:
        frame = frame[frame["completed"].fillna(False).astype(bool)]
    graph = nx.MultiDiGraph(season=int(season), representation="TDGraph-v1")
    for side in ("home", "away"):
        for row in frame.itertuples(index=False):
            team = str(getattr(row, f"{side}_team"))
            graph.add_node(
                team, team_id=_json_scalar(getattr(row, f"{side}_id", None)),
                conference=str(getattr(row, f"{side}_conference", None) or "Independent/Other"),
                classification=str(getattr(row, f"{side}_classification", None) or "unknown"),
            )
    for row in frame.itertuples(index=False):
        away, home = str(row.away_team), str(row.home_team)
        hp, ap = _number(row.home_points), _number(row.away_points)
        winner = home if hp is not None and ap is not None and hp > ap else away if hp is not None and ap is not None and ap > hp else ""
        graph.add_edge(away, home, key=str(row.id), game_id=_json_scalar(row.id), week=int(row.week),
                       start_date=str(row.start_date), completed=bool(row.completed), neutral_site=bool(row.neutral_site),
                       conference_game=bool(row.conference_game), home_points=hp, away_points=ap,
                       margin=None if hp is None or ap is None else hp-ap, winner=winner, venue=str(row.venue or ""))
    for team in graph.nodes:
        games_for_team = [data for u, v, data in graph.edges(data=True) if team in {u, v}]
        completed = [g for g in games_for_team if g.get("home_points") is not None and g.get("away_points") is not None]
        wins = sum(g.get("winner") == team for g in completed)
        graph.nodes[team].update(games=len(games_for_team), completed_games=len(completed), wins=int(wins), losses=int(len(completed)-wins),
                                 degree=int(graph.degree(team)))
    return graph


def export_season_graph(graph: nx.MultiDiGraph, output_dir: str | Path) -> dict:
    """Write portable GraphML plus query-friendly node/edge parquet tables."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame([{"team": team, **data} for team, data in graph.nodes(data=True)])
    edges = pd.DataFrame([{"away_team": u, "home_team": v, "edge_key": key, **data}
                          for u, v, key, data in graph.edges(keys=True, data=True)])
    nodes.to_parquet(output / "nodes.parquet", index=False)
    edges.to_parquet(output / "games.parquet", index=False)
    safe = nx.MultiDiGraph()
    safe.graph.update({k: str(v) for k, v in graph.graph.items()})
    for node, data in graph.nodes(data=True): safe.add_node(node, **{k: _graphml(v) for k, v in data.items()})
    for u, v, key, data in graph.edges(keys=True, data=True): safe.add_edge(u, v, key=key, **{k: _graphml(vv) for k, vv in data.items()})
    nx.write_graphml(safe, output / "season.graphml")
    metadata = {"format": "TDGraph-v1", "season": int(graph.graph["season"]),
                "nodes": graph.number_of_nodes(), "games": graph.number_of_edges(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": ["season.graphml", "nodes.parquet", "games.parquet"]}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return metadata


def plot_season_graph(graph: nx.MultiDiGraph, path: str | Path, *, logo_dir=None, seed=42, dpi=180, fbs_only=True) -> Path:
    """Plot conference-colored/shaped team nodes with optional logo overlays."""
    selected = [node for node, data in graph.nodes(data=True)
                if not fbs_only or str(data.get("classification", "")).lower() == "fbs"]
    subgraph = graph.subgraph(selected)
    simple = nx.Graph()
    simple.add_nodes_from(subgraph.nodes(data=True))
    simple.add_edges_from((u, v) for u, v in subgraph.edges())
    positions = nx.spring_layout(simple, seed=int(seed), k=max(0.3, 2.4/np.sqrt(max(1, len(simple)))), iterations=250, weight=None)
    conferences = sorted({str(graph.nodes[node].get("conference", "Other")) for node in simple})
    colors = conference_color_map(conferences)
    shapes = ["o", "s", "^", "D", "P", "v", "h", "X", "<", ">"]
    fig, axis = plt.subplots(figsize=(20, 16)); fig.patch.set_facecolor("#F7F4ED"); axis.set_facecolor("#F7F4ED")
    conference_edges, non_conference_edges = _season_edge_groups(subgraph, graph)
    if non_conference_edges:
        nx.draw_networkx_edges(
            simple,
            positions,
            edgelist=non_conference_edges,
            ax=axis,
            alpha=0.34,
            width=1.35,
            edge_color=NON_CONFERENCE_EDGE_COLOR,
            style="dashed",
        )
    for conference, edges in conference_edges.items():
        if edges:
            nx.draw_networkx_edges(
                simple,
                positions,
                edgelist=edges,
                ax=axis,
                alpha=0.52,
                width=1.75,
                edge_color=colors.get(conference, "#59636E"),
            )
    for index, conference in enumerate(conferences):
        nodes = [node for node in simple if graph.nodes[node].get("conference") == conference]
        nx.draw_networkx_nodes(simple, positions, nodelist=nodes, node_color=[colors[conference]], node_shape=shapes[index % len(shapes)],
                               node_size=650, linewidths=1.8, edgecolors="white", ax=axis, label=conference)
    if logo_dir:
        logo_root = Path(logo_dir)
        for team, (x, y) in positions.items():
            logo = next((logo_root / f"{slug}.png" for slug in logo_slug_candidates(team)
                         if (logo_root / f"{slug}.png").exists()), None)
            if logo:
                try:
                    image = load_team_logo_image(logo)
                    zoom = 30.0 / max(image.shape[:2])
                    conference = str(graph.nodes[team].get("conference", "Other"))
                    axis.add_artist(
                        AnnotationBbox(
                            OffsetImage(image, zoom=zoom),
                            (x, y),
                            frameon=True,
                            bboxprops={
                                "boxstyle": "circle,pad=0.10",
                                "facecolor": (1.0, 1.0, 1.0, 0.82),
                                "edgecolor": colors.get(conference, "#59636E"),
                                "linewidth": 1.7,
                            },
                        )
                    )
                except Exception:
                    pass
    nx.draw_networkx_labels(simple, positions, ax=axis, font_size=4.2, font_color="#17263C",
                            verticalalignment="top", labels={n: "\n\n"+n for n in simple})
    axis.set_title(f"{graph.graph['season']} TDGraph — teams as nodes, games as connections", fontsize=22, weight="bold", pad=8)
    handles, labels = axis.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.extend([
        Line2D([0], [0], color="#3A4450", lw=1.75, label="Conference game edge uses conference color"),
        Line2D([0], [0], color=NON_CONFERENCE_EDGE_COLOR, lw=1.35, linestyle="--", label="Non-conference game"),
    ])
    labels.extend(["Conference game edge uses conference color", "Non-conference game"])
    axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=min(6, max(1, len(labels))), frameon=False, fontsize=8)
    axis.axis("off"); fig.tight_layout(rect=[0, 0.045, 1, 0.99])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); return path


def conference_color_map(conferences):
    """Return stable, high-contrast colors for known conferences."""
    out = {}
    fallback = plt.get_cmap("tab20")
    fallback_index = 0
    for conference in conferences:
        if conference in CONFERENCE_COLORS:
            out[conference] = CONFERENCE_COLORS[conference]
        else:
            out[conference] = fallback(fallback_index % 20)
            fallback_index += 1
    return out


def _season_edge_groups(subgraph, graph):
    conference_edges = {}
    non_conference_edges = []
    for away, home, data in subgraph.edges(data=True):
        away_conference = str(graph.nodes[away].get("conference", "Other"))
        home_conference = str(graph.nodes[home].get("conference", "Other"))
        is_conference_game = bool(data.get("conference_game")) and away_conference == home_conference
        if is_conference_game:
            conference_edges.setdefault(home_conference, []).append((away, home))
        else:
            non_conference_edges.append((away, home))
    return conference_edges, non_conference_edges


def _number(value):
    return None if pd.isna(value) else float(value)


def _json_scalar(value):
    return value.item() if hasattr(value, "item") else value


def _graphml(value):
    return "" if value is None else value if isinstance(value, (str, int, float, bool)) else str(value)
