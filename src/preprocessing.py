from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

NA_TOKENS = {"", "\\N", "nan", "None", "<NA>"}
TEXT_COLS = ["first_name", "last_name", "email", "phone", "sex"]


def normalize_text_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.isin(NA_TOKENS))


def parse_json_dict(value: Any) -> dict:
    if pd.isna(value):
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def email_domain(value: Any) -> Any:
    if pd.isna(value) or "@" not in str(value):
        return pd.NA
    return str(value).split("@")[-1].lower()


def only_digits(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    digits = re.sub(r"\D+", "", str(value))
    return digits or pd.NA


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    eda = df.copy()
    for col in TEXT_COLS:
        eda[col] = normalize_text_series(eda[col])
    eda["sex_clean"] = eda["sex"].mask(eda["sex"].eq("unknown"))

    if "created_at" in eda.columns:
        eda["created_at"] = pd.to_datetime(eda["created_at"], errors="coerce")

    if "realtime_features" in eda.columns:
        rt = eda["realtime_features"].map(parse_json_dict)
        for key in ["country", "geoname", "geoid"]:
            eda[f"rt_{key}"] = rt.map(lambda x: x.get(key, pd.NA))

    return eda


def build_profiles(eda: pd.DataFrame) -> pd.DataFrame:
    profile_group = eda.groupby("profile_id", sort=False, dropna=False)
    profiles = profile_group.agg(
        entity_id=("entity_id", "first"),
        events=("profile_id", "size"),
        created_at_min=("created_at", "min"),
        created_at_max=("created_at", "max"),
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        email=("email", "first"),
        phone=("phone", "first"),
        birthday=("birthday", "first"),
        sex=("sex", "first"),
        sex_clean=("sex_clean", "first"),
        first_name_nunique=("first_name", "nunique"),
        last_name_nunique=("last_name", "nunique"),
        email_nunique=("email", "nunique"),
        phone_nunique=("phone", "nunique"),
        birthday_nunique=("birthday", "nunique"),
        sex_nunique=("sex_clean", "nunique"),
    ).reset_index()

    entity_profile_counts = profiles.groupby("entity_id")["profile_id"].size()
    profiles["entity_type"] = np.where(
        profiles["entity_id"].map(entity_profile_counts).ge(2),
        "multi_profile",
        "single_profile",
    )
    profiles["email_domain"] = profiles["email"].map(email_domain)
    profiles["phone_digits"] = profiles["phone"].map(only_digits)
    profiles["phone_prefix"] = profiles["phone_digits"].map(
        lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 4 else pd.NA
    )
    return profiles


def build_entities(profiles: pd.DataFrame) -> pd.DataFrame:
    entities = profiles.groupby("entity_id", sort=False).agg(
        profiles=("profile_id", "size"),
        events=("events", "sum"),
        first_names=("first_name", "nunique"),
        last_names=("last_name", "nunique"),
        emails=("email", "nunique"),
        phones=("phone", "nunique"),
        birthdays=("birthday", "nunique"),
        sex_values=("sex_clean", "nunique"),
    ).reset_index()
    entities["is_multi"] = entities["profiles"].ge(2)
    entities["entity_type"] = np.where(entities["is_multi"], "multi_profile", "single_profile")
    return entities
