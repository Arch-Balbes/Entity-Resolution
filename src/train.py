from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.features import PairFeatureEngine, build_profile_sets
from src.io import load_events, resolve_data_path
from src.pairs import make_pair_dataset
from src.preprocessing import build_entities, build_profiles, prepare_events


def _compute_thresholds(
    y_test: pd.Series,
    test_pred_proba: np.ndarray,
    target_precisions: list[float],
) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_test, test_pred_proba)
    threshold_table = pd.DataFrame({
        "threshold": np.r_[thresholds, 1.0],
        "precision": precision,
        "recall": recall,
    })

    operating_points = []
    for target_precision in target_precisions:
        candidates = threshold_table[threshold_table["precision"] >= target_precision]
        if candidates.empty:
            operating_points.append({
                "target_precision": target_precision,
                "threshold": None,
                "precision": None,
                "recall": None,
            })
            continue
        best = candidates.sort_values("recall", ascending=False).iloc[0]
        operating_points.append({
            "target_precision": float(target_precision),
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
        })

    op_df = pd.DataFrame(operating_points)
    auto_row = op_df[op_df["target_precision"] == 0.99]
    manual_row = op_df[op_df["target_precision"] == 0.90]

    return {
        "operating_points": operating_points,
        "auto_merge_threshold": float(auto_row.iloc[0]["threshold"])
        if not auto_row.empty and pd.notna(auto_row.iloc[0]["threshold"])
        else 0.99,
        "manual_review_threshold": float(manual_row.iloc[0]["threshold"])
        if not manual_row.empty and pd.notna(manual_row.iloc[0]["threshold"])
        else 0.5,
        "reject_threshold": 0.0,
    }


def _build_demo_pairs(test_meta: pd.DataFrame, n_each: int = 3) -> list[dict]:
    demos = []
    for target, label in [(1, "duplicate"), (0, "not_duplicate")]:
        subset = test_meta[test_meta["target"] == target].head(n_each)
        for row in subset.itertuples(index=False):
            demos.append({
                "left_profile_id": row.left_profile_id,
                "right_profile_id": row.right_profile_id,
                "expected_target": int(row.target),
                "label": label,
            })
    return demos


def run_training(
    data_path: Path | None = None,
    artifacts_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    random_state = config["random_state"]
    negative_multiplier = config["negative_multiplier"]
    test_size = config["test_size"]
    catboost_params = config["catboost"]
    target_precisions = config["thresholds"]["target_precisions"]

    resolved_data = resolve_data_path(data_path, config["paths"]["data_dir"])
    out_dir = artifacts_dir or config["paths"]["artifacts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {resolved_data}")
    df = load_events(resolved_data)
    eda = prepare_events(df)
    profiles = build_profiles(eda)
    entities = build_entities(profiles)
    profile_sets = build_profile_sets(eda, profiles["profile_id"])
    feature_engine = PairFeatureEngine(profiles, profile_sets)

    entity_split_frame = entities[["entity_id", "is_multi"]].copy()
    train_entity_ids, test_entity_ids = train_test_split(
        entity_split_frame["entity_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=entity_split_frame["is_multi"],
    )
    train_entity_ids = set(train_entity_ids)
    test_entity_ids = set(test_entity_ids)

    X_train, y_train, train_meta = make_pair_dataset(
        profiles, feature_engine, train_entity_ids, negative_multiplier, random_state
    )
    X_test, y_test, test_meta = make_pair_dataset(
        profiles, feature_engine, test_entity_ids, negative_multiplier, random_state + 1
    )

    model = CatBoostClassifier(
        iterations=catboost_params["iterations"],
        depth=catboost_params["depth"],
        learning_rate=catboost_params["learning_rate"],
        loss_function=catboost_params["loss_function"],
        eval_metric=catboost_params["eval_metric"],
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(X_train, y_train),
        eval_set=Pool(X_test, y_test),
        use_best_model=True,
    )

    test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_pred_label = (test_pred_proba >= 0.5).astype(int)

    threshold_info = _compute_thresholds(y_test, test_pred_proba, target_precisions)
    report = classification_report(
        y_test, test_pred_label,
        target_names=["not_duplicate", "duplicate"],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, test_pred_label)

    feature_importance = model.get_feature_importance(prettified=True)
    top_features = feature_importance.head(10).to_dict(orient="records")

    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "trained_at": trained_at,
        "data_path": str(resolved_data),
        "events": len(df),
        "profiles": len(profiles),
        "entities": len(entities),
        "train_pairs": len(X_train),
        "test_pairs": len(X_test),
        "train_positive_share": float(y_train.mean()),
        "test_positive_share": float(y_test.mean()),
        "features": int(X_train.shape[1]),
        "roc_auc": float(roc_auc_score(y_test, test_pred_proba)),
        "pr_auc": float(average_precision_score(y_test, test_pred_proba)),
        "best_iteration": int(model.get_best_iteration()),
        "classification_report": report,
        "confusion_matrix": {
            "labels": ["not_duplicate", "duplicate"],
            "matrix": cm.tolist(),
        },
        "top_features": top_features,
        "pair_dataset_summary": {
            "train": {"pairs": len(X_train), "positive": int(y_train.sum())},
            "test": {"pairs": len(X_test), "positive": int(y_test.sum())},
        },
    }

    model.save_model(str(out_dir / "model.cbm"))
    profiles.to_parquet(out_dir / "profiles.parquet", index=False)

    with (out_dir / "profile_sets.pkl").open("wb") as f:
        pickle.dump(profile_sets, f)

    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f, indent=2)

    with (out_dir / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "demo_pairs.json").open("w", encoding="utf-8") as f:
        json.dump(_build_demo_pairs(test_meta), f, indent=2)

    with (out_dir / "train_test_entities.json").open("w", encoding="utf-8") as f:
        json.dump({
            "train_entity_count": len(train_entity_ids),
            "test_entity_count": len(test_entity_ids),
        }, f, indent=2)

    print(f"Model saved to {out_dir / 'model.cbm'}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
    return metrics
