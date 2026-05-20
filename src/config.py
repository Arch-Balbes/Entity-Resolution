from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["random_state"] = int(os.getenv("RANDOM_STATE", config.get("random_state", 42)))
    config["paths"] = config.get("paths", {})
    config["paths"]["data_dir"] = Path(
        os.getenv("DATA_DIR", config["paths"].get("data_dir", "data"))
    )
    if not config["paths"]["data_dir"].is_absolute():
        config["paths"]["data_dir"] = ROOT_DIR / config["paths"]["data_dir"]

    artifacts = os.getenv("ARTIFACTS_DIR", config["paths"].get("artifacts_dir", "artifacts"))
    config["paths"]["artifacts_dir"] = Path(artifacts)
    if not config["paths"]["artifacts_dir"].is_absolute():
        config["paths"]["artifacts_dir"] = ROOT_DIR / config["paths"]["artifacts_dir"]

    return config
