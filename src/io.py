from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".parquet", ".csv"}


def resolve_data_path(
    data_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported dataset format: {path.suffix}")
        return path

    env_path = os.getenv("DATA_PATH")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"DATA_PATH not found: {path}")
        return path

    search_dir = Path(data_dir) if data_dir is not None else Path("data")
    if not search_dir.is_absolute():
        search_dir = Path(__file__).resolve().parents[1] / search_dir

    if not search_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {search_dir}. "
            "Place a .parquet or .csv file in ml_pipeline/data/ or set DATA_PATH."
        )

    parquet_files = sorted(search_dir.glob("*.parquet"))
    if parquet_files:
        return parquet_files[0]

    csv_files = sorted(search_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        f"No dataset found in {search_dir}. "
        "Add split_label_train_V3.snappy.parquet or split_label_train_V3.csv."
    )


def load_events(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(data_path)
    if suffix == ".csv":
        return pd.read_csv(data_path, low_memory=False)
    raise ValueError(f"Unsupported format: {suffix}")


def load_events_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    raise ValueError("Поддерживаются только файлы .csv и .parquet")
