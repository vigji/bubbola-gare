from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import RAW_EXCEL_PATH, RAW_PARQUET_PATH, ensure_data_dirs


def load_raw_excel(source_path: Optional[Path] = None) -> pd.DataFrame:
    path = source_path or RAW_EXCEL_PATH
    df = pd.read_excel(path)
    return df


def save_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)


def ingest_excel_to_parquet(
    source_path: Optional[Path] = None, dest_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load the source Excel file and persist the full table to Parquet.
    Returns the loaded dataframe.
    """
    ensure_data_dirs()
    src = source_path or RAW_EXCEL_PATH
    dest = dest_path or RAW_PARQUET_PATH
    df = load_raw_excel(src)
    df = df.reset_index(drop=True)
    save_parquet(df, dest)
    return df


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
