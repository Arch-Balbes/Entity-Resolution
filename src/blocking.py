from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.features import ProfileSets

MAX_BLOCK_SIZE = 200
MAX_RARE_FEATURE_BLOCK_SIZE = 50
MAX_CANDIDATE_PAIRS = 250_000


def normalized_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def add_block_pairs(candidate_pairs: set[tuple[str, str]], group: pd.DataFrame) -> None:
    if len(group) < 2 or len(group) > MAX_BLOCK_SIZE:
        return
    profile_ids = group["profile_id"].to_list()
    for left, right in itertools.combinations(profile_ids, 2):
        if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
            return
        candidate_pairs.add(normalized_pair(left, right))


def add_column_blocks(
    candidate_pairs: set[tuple[str, str]],
    subset: pd.DataFrame,
    columns: list[list[str]],
) -> None:
    for columns_for_block in columns:
        valid = subset.dropna(subset=columns_for_block)
        for _, group in valid.groupby(columns_for_block, sort=False):
            add_block_pairs(candidate_pairs, group)
            if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                return


def add_rare_feature_blocks(
    candidate_pairs: set[tuple[str, str]],
    subset: pd.DataFrame,
    profile_fs_sets: dict[str, set[str]],
) -> None:
    feature_to_profile_ids: dict[str, list[str]] = {}
    subset_profile_ids = set(subset["profile_id"])

    for profile_id in subset_profile_ids:
        for feature in profile_fs_sets.get(profile_id, ()):
            feature_to_profile_ids.setdefault(feature, []).append(profile_id)

    rare_feature_blocks = [
        profile_ids
        for profile_ids in feature_to_profile_ids.values()
        if 2 <= len(profile_ids) <= MAX_RARE_FEATURE_BLOCK_SIZE
    ]
    rare_feature_blocks = sorted(rare_feature_blocks, key=len)

    for profile_ids in rare_feature_blocks:
        for left, right in itertools.combinations(profile_ids, 2):
            if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                return
            candidate_pairs.add(normalized_pair(left, right))


def make_candidate_pairs(
    profiles: pd.DataFrame,
    profile_sets: ProfileSets,
    max_pairs: int = MAX_CANDIDATE_PAIRS,
) -> set[tuple[str, str]]:
    subset = profiles.copy()
    subset["primary_geoid"] = subset["profile_id"].map(
        lambda pid: next(iter(sorted(profile_sets.geoid.get(pid, ()))), pd.NA)
    )
    subset["primary_country"] = subset["profile_id"].map(
        lambda pid: next(iter(sorted(profile_sets.country.get(pid, ()))), pd.NA)
    )

    candidate_pairs: set[tuple[str, str]] = set()
    blocking_columns = [
        ["phone_digits"],
        ["email_domain", "first_name"],
        ["email_domain", "phone_prefix"],
        ["phone_prefix", "first_name"],
        ["phone_prefix", "sex_clean"],
        ["email_domain", "primary_geoid"],
        ["first_name", "primary_geoid"],
        ["primary_geoid", "sex_clean"],
    ]

    add_column_blocks(candidate_pairs, subset, blocking_columns)
    if len(candidate_pairs) < max_pairs:
        add_rare_feature_blocks(candidate_pairs, subset, profile_sets.fs)

    return candidate_pairs
