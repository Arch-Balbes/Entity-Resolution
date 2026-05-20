from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.features import PairFeatureEngine


def make_positive_pairs(
    profiles: pd.DataFrame,
    entity_ids: set[str],
) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    subset = profiles[profiles["entity_id"].isin(entity_ids)]
    for _, group in subset.groupby("entity_id", sort=False):
        if len(group) < 2:
            continue
        profile_ids = group["profile_id"].to_list()
        rows.extend((left, right, 1) for left, right in itertools.combinations(profile_ids, 2))
    return rows


def add_negative_pair(
    pair_set: set[tuple[str, str, int]],
    left: str,
    right: str,
    profile_by_id: pd.DataFrame,
) -> bool:
    if left == right:
        return False
    if profile_by_id.at[left, "entity_id"] == profile_by_id.at[right, "entity_id"]:
        return False
    pair = tuple(sorted((left, right))) + (0,)
    if pair in pair_set:
        return False
    pair_set.add(pair)
    return True


def make_negative_pairs(
    profiles: pd.DataFrame,
    profile_by_id: pd.DataFrame,
    entity_ids: set[str],
    target_count: int,
    random_state: int,
) -> list[tuple[str, str, int]]:
    local_rng = np.random.default_rng(random_state)
    subset = profiles[profiles["entity_id"].isin(entity_ids)].copy()
    pair_set: set[tuple[str, str, int]] = set()

    for block_col in ["email_domain", "first_name", "phone_prefix"]:
        valid = subset.dropna(subset=[block_col])
        for _, group in valid.groupby(block_col, sort=False):
            if len(group) < 2:
                continue
            profile_ids = group["profile_id"].to_numpy().copy()
            local_rng.shuffle(profile_ids)
            for left, right in zip(profile_ids[:-1], profile_ids[1:]):
                add_negative_pair(pair_set, left, right, profile_by_id)
                if len(pair_set) >= target_count:
                    break
            if len(pair_set) >= target_count:
                break
        if len(pair_set) >= target_count:
            break

    profile_ids = subset["profile_id"].to_numpy()
    attempts = 0
    max_attempts = max(target_count * 30, 1_000)
    while len(pair_set) < target_count and attempts < max_attempts and len(profile_ids) >= 2:
        left, right = local_rng.choice(profile_ids, size=2, replace=False)
        add_negative_pair(pair_set, left, right, profile_by_id)
        attempts += 1

    return list(pair_set)


def make_pair_dataset(
    profiles: pd.DataFrame,
    feature_engine: PairFeatureEngine,
    entity_ids: set[str],
    negative_multiplier: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    positive_pairs = make_positive_pairs(profiles, entity_ids)
    profile_by_id = feature_engine.profile_by_id
    negative_pairs = make_negative_pairs(
        profiles=profiles,
        profile_by_id=profile_by_id,
        entity_ids=entity_ids,
        target_count=max(len(positive_pairs) * negative_multiplier, 1_000),
        random_state=random_state,
    )
    pair_rows = positive_pairs + negative_pairs
    local_rng = np.random.default_rng(random_state)
    local_rng.shuffle(pair_rows)

    meta = pd.DataFrame(pair_rows, columns=["left_profile_id", "right_profile_id", "target"])
    pair_ids = [(left, right) for left, right, _ in pair_rows]
    X = feature_engine.compute_batch(pair_ids)
    y = meta["target"].astype(int)
    return X, y, meta
