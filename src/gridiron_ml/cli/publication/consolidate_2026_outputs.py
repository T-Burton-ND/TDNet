#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Consolidate active 2026 outputs into raw-data and publication-figure roots."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

IMAGE_SUFFIXES = {".png", ".svg", ".jpg", ".jpeg", ".webp"}
SOURCE_DIRS = (
    Path("publication/2026/manual_polls"),
    Path("data/publication/2026/weekly_operations"),
    Path("publication/2026/weekly_predictions"),
)


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--apply", action="store_true", help="Perform the moves; otherwise print the plan.")
    args = parser.parse_args()
    project = args.project_root.resolve()
    raw_root = project / "data/publication/2026"
    figure_root = project / "publication/2026/figures"
    plan = []
    for source_rel in SOURCE_DIRS:
        source = project / source_rel
        if not source.exists():
            continue
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            relative = path.relative_to(source)
            destination_root = figure_root if path.suffix.lower() in IMAGE_SUFFIXES else raw_root
            destination = destination_root / source_rel.name / relative
            plan.append({"source": str(path), "destination": str(destination), "kind": "figure" if destination_root == figure_root else "raw"})
    if not args.apply:
        print(json.dumps({"planned_files": len(plan), "moves": plan}, indent=2))
        return
    for item in plan:
        source = Path(item["source"])
        destination = Path(item["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if source.read_bytes() != destination.read_bytes():
                raise FileExistsError(f"Destination exists with different bytes: {destination}")
            source.unlink()
        else:
            shutil.move(str(source), str(destination))
    for source_rel in SOURCE_DIRS:
        source = project / source_rel
        if source.exists():
            for directory in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            try:
                source.rmdir()
            except OSError:
                pass
    index = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root.relative_to(project)),
        "figure_root": str(figure_root.relative_to(project)),
        "moved_files": plan,
        "excluded": ["docs/publication_2026/preseason", "publication/2026/*.ipynb"],
    }
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "OUTPUT_INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"moved_files": len(plan), "raw_root": str(raw_root), "figure_root": str(figure_root)}, indent=2))


if __name__ == "__main__":
    main()
