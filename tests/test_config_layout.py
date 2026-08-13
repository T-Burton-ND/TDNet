from pathlib import Path
import subprocess

from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import DEFAULT_CONFIG_PATH


def test_fetch_default_config_path_exists():
    assert DEFAULT_CONFIG_PATH.exists()


def test_yaml_configs_live_under_configs_directory():
    repo_root = Path(__file__).resolve().parents[1]
    non_runtime_config_roots = {
        ".github",
        "docs",
        "models",
        "publication",
        "tdnet_model_freeze_release_candidate",
    }
    tracked = subprocess.run(
        ["git", "ls-files", "*.yaml", "*.yml"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    yaml_paths = sorted(
        Path(value)
        for value in tracked
        if Path(value).parts[0] not in non_runtime_config_roots
        and ".cleanup_quarantine" not in Path(value).parts
        and Path(value).name != "environment.yaml"
    )

    assert yaml_paths
    misplaced = [path for path in yaml_paths if path.parts[0] != "configs"]
    assert not misplaced, f"runtime YAML outside configs/: {misplaced}"
