from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier

from src.config import load_config


@dataclass
class ModelArtifacts:
    model: CatBoostClassifier
    feature_columns: list[str]
    thresholds: dict[str, Any]
    artifacts_dir: Path


def artifacts_available(artifacts_dir: Path | None = None) -> bool:
    config = load_config()
    directory = artifacts_dir or config["paths"]["artifacts_dir"]
    return (directory / "model.cbm").exists()


def load_artifacts(artifacts_dir: Path | None = None) -> ModelArtifacts:
    config = load_config()
    directory = artifacts_dir or config["paths"]["artifacts_dir"]

    model_path = directory / "model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first: "
            "docker compose run --rm train"
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with (directory / "feature_columns.json").open(encoding="utf-8") as f:
        feature_columns = json.load(f)

    thresholds = {
        "auto_merge_threshold": 0.9999,
        "manual_review_threshold": 0.99,
    }
    thresholds_path = directory / "thresholds.json"
    if thresholds_path.exists():
        with thresholds_path.open(encoding="utf-8") as f:
            thresholds.update(json.load(f))

    return ModelArtifacts(
        model=model,
        feature_columns=feature_columns,
        thresholds=thresholds,
        artifacts_dir=directory,
    )


def classify_score(score: float, thresholds: dict[str, Any]) -> str:
    auto_threshold = thresholds.get("auto_merge_threshold", 0.9999)
    manual_threshold = thresholds.get("manual_review_threshold", 0.99)
    if score >= auto_threshold:
        return "auto_merge"
    if score >= manual_threshold:
        return "manual_review"
    return "reject"
