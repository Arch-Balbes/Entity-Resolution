from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing import NA_TOKENS


def as_feature_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value if item is not None]
    value_str = str(value)
    if value_str in NA_TOKENS or value_str == "[]":
        return []
    return re.findall(r"'([^']+)'", value_str)


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap_count(left: set, right: set) -> int:
    if not left or not right:
        return 0
    return len(left & right)


@dataclass
class ProfileSets:
    fs: dict[str, set[str]]
    non_processing: dict[str, set[str]]
    geoid: dict[str, set[str]]
    country: dict[str, set[str]]


def build_profile_sets(eda: pd.DataFrame, profile_ids: pd.Series) -> ProfileSets:
    fs_sets = {pid: set() for pid in profile_ids}
    non_processing_sets = {pid: set() for pid in profile_ids}
    geoid_sets = {pid: set() for pid in profile_ids}
    country_sets = {pid: set() for pid in profile_ids}

    cols = ["profile_id", "fs_features", "non_processing_features"]
    if "rt_geoid" in eda.columns:
        cols.append("rt_geoid")
    if "rt_country" in eda.columns:
        cols.append("rt_country")

    for row in eda[cols].itertuples(index=False):
        pid = row.profile_id
        fs_sets[pid].update(as_feature_list(row.fs_features))
        non_processing_sets[pid].update(as_feature_list(row.non_processing_features))
        if hasattr(row, "rt_geoid") and pd.notna(row.rt_geoid):
            geoid_sets[pid].add(str(row.rt_geoid))
        if hasattr(row, "rt_country") and pd.notna(row.rt_country):
            country_sets[pid].add(str(row.rt_country))

    return ProfileSets(
        fs=fs_sets,
        non_processing=non_processing_sets,
        geoid=geoid_sets,
        country=country_sets,
    )


class PairFeatureEngine:
    def __init__(self, profiles: pd.DataFrame, profile_sets: ProfileSets):
        self.profiles = profiles
        self.profile_sets = profile_sets
        self.profile_by_id = profiles.set_index("profile_id", drop=False)

    def compute(self, left_id: str, right_id: str) -> dict:
        left = self.profile_by_id.loc[left_id]
        right = self.profile_by_id.loc[right_id]
        ps = self.profile_sets

        def both_have(column: str) -> bool:
            return pd.notna(left[column]) and pd.notna(right[column])

        def same(column: str) -> int:
            return int(both_have(column) and left[column] == right[column])

        def conflict(column: str) -> int:
            return int(both_have(column) and left[column] != right[column])

        left_fs = ps.fs[left_id]
        right_fs = ps.fs[right_id]
        left_np = ps.non_processing[left_id]
        right_np = ps.non_processing[right_id]
        left_geoid = ps.geoid[left_id]
        right_geoid = ps.geoid[right_id]
        left_country = ps.country[left_id]
        right_country = ps.country[right_id]

        return {
            "same_first_name": same("first_name"),
            "same_last_name": same("last_name"),
            "same_email": same("email"),
            "same_email_domain": same("email_domain"),
            "same_phone": same("phone_digits"),
            "same_phone_prefix": same("phone_prefix"),
            "same_birthday": same("birthday"),
            "same_sex": same("sex_clean"),
            "conflict_first_name": conflict("first_name"),
            "conflict_last_name": conflict("last_name"),
            "conflict_email_domain": conflict("email_domain"),
            "conflict_phone_prefix": conflict("phone_prefix"),
            "conflict_birthday": conflict("birthday"),
            "conflict_sex": conflict("sex_clean"),
            "both_have_first_name": int(both_have("first_name")),
            "both_have_last_name": int(both_have("last_name")),
            "both_have_phone": int(both_have("phone_digits")),
            "both_have_birthday": int(both_have("birthday")),
            "both_have_sex": int(both_have("sex_clean")),
            "fs_jaccard": jaccard(left_fs, right_fs),
            "fs_overlap": overlap_count(left_fs, right_fs),
            "fs_min_size": min(len(left_fs), len(right_fs)),
            "fs_max_size": max(len(left_fs), len(right_fs)),
            "non_processing_jaccard": jaccard(left_np, right_np),
            "non_processing_overlap": overlap_count(left_np, right_np),
            "geoid_jaccard": jaccard(left_geoid, right_geoid),
            "country_jaccard": jaccard(left_country, right_country),
            "events_min": min(left["events"], right["events"]),
            "events_max": max(left["events"], right["events"]),
            "events_sum": left["events"] + right["events"],
            "created_at_delta_days": abs(
                (left["created_at_min"] - right["created_at_min"]).total_seconds()
            )
            / 86_400,
        }

    def compute_batch(self, pairs: list[tuple[str, str]]) -> pd.DataFrame:
        return pd.DataFrame([self.compute(left, right) for left, right in pairs])
