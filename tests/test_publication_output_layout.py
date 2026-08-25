from pathlib import Path
import json
import os
import subprocess
import sys


def test_x_package_reads_canonical_pre_game_layout(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    pre_game = tmp_path / "publication/2026/week_01/pre_game"
    expected = [
        pre_game / "figures/all_games_predictions.png",
        pre_game / "figures/top25_week_matchups.png",
        pre_game / "figures/tdnet_top25.png",
    ]
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    subprocess.run(
        [
            sys.executable,
            "src/gridiron_ml/cli/publication/build_x_post_package.py",
            "--weekly-output",
            str(pre_game),
            "--season",
            "2026",
            "--week",
            "1",
        ],
        cwd=project,
        env={**os.environ, "PYTHONPATH": str(project / "src")},
        check=True,
    )

    manifest = json.loads((pre_game / "x_post_package/manifest.json").read_text())
    assert all(item["exists"] for item in manifest["media"])
    assert [Path(item["path"]) for item in manifest["media"]] == expected
