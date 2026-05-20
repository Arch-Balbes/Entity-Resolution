from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.blocking import make_candidate_pairs
from src.features import PairFeatureEngine, ProfileSets, build_profile_sets
from src.graph import build_components
from src.inference import ModelArtifacts
from src.preprocessing import build_profiles, prepare_events

REQUIRED_COLUMNS = {
    "profile_id",
    "created_at",
    "first_name",
    "last_name",
    "email",
    "phone",
    "birthday",
    "sex",
    "non_processing_features",
    "realtime_features",
    "fs_features",
}
OPTIONAL_COLUMNS = {"entity_id", "entity_type"}

HARD_CONFLICT_COLUMNS = ["conflict_birthday", "conflict_sex", "conflict_phone_prefix"]
MAX_AUTO_COMPONENT_SIZE = 20


@dataclass
class ResolutionResult:
    profiles: pd.DataFrame
    clusters: pd.DataFrame
    pairs_scored: pd.DataFrame
    summary: dict[str, Any]


def validate_events_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"В файле не хватает колонок: {sorted(missing)}. "
            f"Ожидаются поля событий профиля (см. dataset_description.md)."
        )


def _score_candidates(
    candidate_pairs: set[tuple[str, str]],
    feature_engine: PairFeatureEngine,
    artifacts: ModelArtifacts,
    auto_merge_threshold: float,
    manual_review_threshold: float,
) -> pd.DataFrame:
    pair_rows = sorted(candidate_pairs)
    if not pair_rows:
        return pd.DataFrame(columns=[
            "left_profile_id", "right_profile_id", "score", "action",
            "has_hard_conflict", "strong_auto_evidence",
        ])

    X = feature_engine.compute_batch(pair_rows)
    scores = artifacts.model.predict_proba(X[artifacts.feature_columns])[:, 1]

    meta = pd.DataFrame(pair_rows, columns=["left_profile_id", "right_profile_id"])
    meta["score"] = scores

    meta["has_hard_conflict"] = X[HARD_CONFLICT_COLUMNS].sum(axis=1).gt(0).to_numpy()
    meta["strong_auto_evidence"] = (
        X["same_phone"].eq(1)
        | X["same_email"].eq(1)
        | (X["fs_jaccard"].ge(0.90) & X["non_processing_jaccard"].ge(0.80))
    ).to_numpy()

    meta["action"] = "reject"
    meta.loc[meta["score"].ge(manual_review_threshold), "action"] = "manual_review"
    meta.loc[
        meta["score"].ge(auto_merge_threshold)
        & ~meta["has_hard_conflict"]
        & meta["strong_auto_evidence"],
        "action",
    ] = "auto_merge"

    return meta


def _clusters_from_edges(
    profiles: pd.DataFrame,
    edges: pd.DataFrame,
    merge_actions: set[str] | None,
    max_component_size: int,
    min_score: float | None = None,
) -> pd.DataFrame:
    profile_ids = profiles["profile_id"].astype(str).tolist()
    if min_score is not None:
        merge_edges = edges[edges["score"] >= min_score].copy()
    else:
        merge_edges = edges[edges["action"].isin(merge_actions or set())].copy()

    if merge_edges.empty:
        components = {pid: [pid] for pid in profile_ids}
    else:
        components, _, _ = build_components(
            profile_ids,
            merge_edges,
            max_component_size=max_component_size,
        )

    cluster_rows = []
    cluster_idx = 0
    for _root, members in components.items():
        if len(members) < 2:
            continue
        cluster_idx += 1
        for profile_id in sorted(members):
            cluster_rows.append({
                "cluster_id": f"person_{cluster_idx:05d}",
                "profile_id": profile_id,
                "cluster_size": len(members),
            })

    if not cluster_rows:
        return pd.DataFrame(columns=["cluster_id", "profile_id", "cluster_size"])

    clusters = pd.DataFrame(cluster_rows)
    profile_cols = [
        "profile_id", "entity_id", "first_name", "last_name", "email",
        "phone", "email_domain", "phone_digits", "events",
    ]
    available = [c for c in profile_cols if c in profiles.columns]
    return clusters.merge(profiles[available], on="profile_id", how="left")


def resolve_entities(
    events: pd.DataFrame,
    artifacts: ModelArtifacts,
    *,
    merge_actions: set[str] | None = None,
    min_score: float | None = None,
    auto_merge_threshold: float | None = None,
    manual_review_threshold: float | None = None,
    max_component_size: int = MAX_AUTO_COMPONENT_SIZE,
) -> ResolutionResult:
    validate_events_schema(events)

    auto_thr = auto_merge_threshold or artifacts.thresholds.get("auto_merge_threshold", 0.9999)
    manual_thr = manual_review_threshold or artifacts.thresholds.get("manual_review_threshold", 0.99)
    if merge_actions is None and min_score is None:
        merge_actions = {"auto_merge", "manual_review"}

    eda = prepare_events(events)
    profiles = build_profiles(eda)
    profile_sets = build_profile_sets(eda, profiles["profile_id"])
    feature_engine = PairFeatureEngine(profiles, profile_sets)

    candidate_pairs = make_candidate_pairs(profiles, profile_sets)
    pairs_scored = _score_candidates(
        candidate_pairs,
        feature_engine,
        artifacts,
        auto_merge_threshold=auto_thr,
        manual_review_threshold=manual_thr,
    )

    clusters = _clusters_from_edges(
        profiles,
        pairs_scored,
        merge_actions=merge_actions,
        max_component_size=max_component_size,
        min_score=min_score,
    )

    n_multi = 0
    if not clusters.empty:
        n_multi = clusters["cluster_id"].nunique()

    summary = {
        "events": len(events),
        "profiles": len(profiles),
        "candidate_pairs": len(candidate_pairs),
        "pairs_scored": len(pairs_scored),
        "pairs_auto_merge": int((pairs_scored["action"] == "auto_merge").sum()) if len(pairs_scored) else 0,
        "pairs_manual_review": int((pairs_scored["action"] == "manual_review").sum()) if len(pairs_scored) else 0,
        "clusters_found": n_multi,
        "profiles_in_clusters": len(clusters),
        "merge_actions": sorted(merge_actions) if merge_actions else [],
        "min_score": min_score,
        "auto_merge_threshold": auto_thr,
        "manual_review_threshold": manual_thr,
    }

    return ResolutionResult(
        profiles=profiles,
        clusters=clusters,
        pairs_scored=pairs_scored,
        summary=summary,
    )
