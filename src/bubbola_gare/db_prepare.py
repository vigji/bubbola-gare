from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import (
    DB_READY_PATH,
    ID_COLUMN,
    RAW_DATE_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    RAW_PARQUET_PATH,
    RAW_SOURCE_COLUMNS,
    SUMMARY_EMBEDDINGS_PATH,
    SUMMARY_PATH,
    TEXT_COLUMNS,
    VENDOR_COLUMN,
)
from .preprocessing import (
    ARTIFACT_REGEXES,
    INTERNAL_CODE_REGEXES,
    LICENSE_PLATE_REGEX,
    combine_text_fields,
    normalize_text,
    normalize_whitespace,
)
from .summarization import SUMMARY_COLUMN

logger = logging.getLogger(__name__)


REGION_NORMALIZATION = {
    "friuli venezia giulia": "Friuli Venezia Giulia",
    "friuli - venezia giulia": "Friuli Venezia Giulia",
    "emilia romagna": "Emilia-Romagna",
    "tarragona": "Tarragona",
}

TEXT_STRIP_REGEXES: Iterable[str] = (
    list(ARTIFACT_REGEXES)
    + list(INTERNAL_CODE_REGEXES)
    + [LICENSE_PLATE_REGEX]
)


def _clean_text_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value
    for rx in TEXT_STRIP_REGEXES:
        text = re.sub(rx, " ", text, flags=re.IGNORECASE)
    text = normalize_whitespace(text)
    return text


def _normalize_region(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    base = normalize_whitespace(value)
    if not base:
        return None
    key = normalize_whitespace(re.sub(r"[^a-zA-Z ]+", " ", base)).lower()
    if key in REGION_NORMALIZATION:
        return REGION_NORMALIZATION[key]
    return base


def _to_vector(val: object) -> Optional[list[float]]:
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, np.ndarray):
        return val.astype(np.float32).tolist()
    return None


def build_db_ready_dataframe(
    raw_path: str | Path = RAW_PARQUET_PATH,
    summary_path: str | Path = SUMMARY_PATH,
    summary_embeddings_path: str | Path = SUMMARY_EMBEDDINGS_PATH,
    drop_missing_vectors: bool = True,
) -> pd.DataFrame:
    raw_df = pd.read_parquet(raw_path)
    for col in RAW_SOURCE_COLUMNS:
        if col not in raw_df.columns:
            raw_df[col] = None
    for col in RAW_DATE_COLUMNS:
        raw_df[col] = pd.to_datetime(raw_df[col], errors="coerce").dt.date
    for col in RAW_NUMERIC_COLUMNS:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    df = raw_df.reset_index(drop=False).rename(columns={"index": "source_row"})
    df["record_id"] = df.index.astype("int64")

    if ID_COLUMN in df.columns:
        df["order_code"] = df[ID_COLUMN]
    else:
        df["order_code"] = df["record_id"].astype(str)

    df["vendor"] = df[VENDOR_COLUMN] if VENDOR_COLUMN in df.columns else ""
    df["order_date"] = pd.to_datetime(df.get("data_ord"), errors="coerce").dt.date
    df["delivery_date"] = pd.to_datetime(df.get("data_consegna"), errors="coerce").dt.date
    df["category"] = df.get("tipologia")
    df["contract_type"] = df.get("tipo_contratto_txt")
    df["order_type"] = df.get("tipo_ordine")
    df["requester"] = df.get("richiedente")
    df["executor"] = df.get("esecutore")
    df["region"] = df.get("regione").apply(_normalize_region) if "regione" in df.columns else None
    df["city"] = df.get("comune").apply(_clean_text_value) if "comune" in df.columns else ""
    df["province"] = df.get("provincia").apply(_clean_text_value) if "provincia" in df.columns else ""
    df["country"] = df.get("nazione").apply(_clean_text_value) if "nazione" in df.columns else ""
    df["cap"] = df.get("cap") if "cap" in df.columns else ""
    df["amount"] = pd.to_numeric(df.get("importo_ordine"), errors="coerce")

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text_value)

    df["text_raw"] = combine_text_fields(df, TEXT_COLUMNS)
    df["text_normalized"] = df["text_raw"].apply(normalize_text)

    summaries = pd.read_parquet(summary_path)
    summaries = summaries[["record_id", SUMMARY_COLUMN, "summary_normalized"]]

    summaries_emb = pd.read_parquet(summary_embeddings_path)
    summaries_emb = summaries_emb[["record_id", "embedding"]].rename(
        columns={"embedding": "summary_embedding"}
    )

    merged = df.merge(summaries, on="record_id", how="left")
    merged = merged.merge(summaries_emb, on="record_id", how="left")
    merged["summary_embedding"] = merged["summary_embedding"].apply(_to_vector)

    before = len(merged)
    if drop_missing_vectors:
        merged = merged[merged["summary_embedding"].notnull()].copy()
    dropped = before - len(merged)
    if dropped:
        logger.info("Dropped %d rows without summary embeddings", dropped)

    base_cols = [
        "record_id",
        "source_row",
        "order_code",
        "order_date",
        "delivery_date",
        "category",
        "contract_type",
        "order_type",
        "requester",
        "executor",
        "vendor",
        "region",
        "city",
        "province",
        "country",
        "cap",
        "amount",
        "text_raw",
        "text_normalized",
        SUMMARY_COLUMN,
        "summary_normalized",
        "summary_embedding",
    ]
    raw_cols_unique = [c for c in RAW_SOURCE_COLUMNS if c not in base_cols]
    cleaned = merged[base_cols + raw_cols_unique].copy()
    return cleaned


def write_db_ready_parquet(
    dest_path: str | Path = DB_READY_PATH,
    raw_path: str | Path = RAW_PARQUET_PATH,
    summary_path: str | Path = SUMMARY_PATH,
    summary_embeddings_path: str | Path = SUMMARY_EMBEDDINGS_PATH,
) -> pd.DataFrame:
    df = build_db_ready_dataframe(
        raw_path=raw_path,
        summary_path=summary_path,
        summary_embeddings_path=summary_embeddings_path,
    )
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("Saved cleaned dataset with %d rows to %s", len(df), dest)
    return df


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Clean and assemble DB-ready orders table")
    parser.add_argument("--raw", type=str, default=str(RAW_PARQUET_PATH), help="Path to raw parquet table")
    parser.add_argument("--summary", type=str, default=str(SUMMARY_PATH), help="Path to summaries parquet")
    parser.add_argument(
        "--summary-emb", type=str, default=str(SUMMARY_EMBEDDINGS_PATH), help="Path to summary embeddings parquet"
    )
    parser.add_argument("--dest", type=str, default=str(DB_READY_PATH), help="Destination parquet path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    write_db_ready_parquet(
        dest_path=args.dest,
        raw_path=args.raw,
        summary_path=args.summary,
        summary_embeddings_path=args.summary_emb,
    )


if __name__ == "__main__":
    main()
