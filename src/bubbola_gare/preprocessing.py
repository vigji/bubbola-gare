from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, List

import pandas as pd

from .config import (
    COMMESSE_CLIENT_COLUMN,
    COMMESSE_DATE_COLUMNS,
    COMMESSE_ID_COLUMN,
    COMMESSE_NUMERIC_COLUMNS,
    COMMESSE_RENAME_MAP,
    COMMESSE_SOURCE_COLUMNS,
    COMMESSE_TEXT_COLUMNS,
    ID_COLUMN,
    TEXT_COLUMNS,
    VENDOR_COLUMN,
)

logger = logging.getLogger(__name__)


BOILERPLATE_REGEXES = [
    r"con la presente vi confermiamo l[’'` ]ordine[^:]*:",
    r"ai seguenti prezzi e condizioni:?",
    r"a termine lavori verr[aà] redatto un consuntivo[^.]*\.",
]

ARTIFACT_REGEXES = [
    r"_x000D_",
]

INTERNAL_CODE_REGEXES = [
    r"\(cod\. int\.[^)]+\)",
    r"s/n\s*\S+",
    r"\[[^\]]+\]",
]

LICENSE_PLATE_REGEX = r"\btargat[oa]\s+[A-Z]{1,2}\s*\d{3}\s*[A-Z]{1,2}\b"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    s = text
    for rx in ARTIFACT_REGEXES:
        s = re.sub(rx, " ", s, flags=re.IGNORECASE)
    for rx in INTERNAL_CODE_REGEXES:
        s = re.sub(rx, " ", s, flags=re.IGNORECASE)
    s = re.sub(LICENSE_PLATE_REGEX, " ", s, flags=re.IGNORECASE)
    for rx in BOILERPLATE_REGEXES:
        s = re.sub(rx, " ", s, flags=re.IGNORECASE)
    s = s.lower()
    s = strip_accents(s)
    s = normalize_whitespace(s)
    return s


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def combine_text_fields(df: pd.DataFrame, text_columns: Iterable[str]) -> pd.Series:
    """
    Combine the configured text columns into a single free-text field.
    Missing columns are ignored.
    """
    available = [c for c in text_columns if c in df.columns]
    if not available:
        raise ValueError("No configured text columns were found in the dataframe.")
    return (
        df[available]
        .fillna("")
        .agg(lambda row: " | ".join([part for part in row if str(part).strip()]), axis=1)
        .str.strip()
    )


def preprocess_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the text corpus and attach identifiers + vendor metadata.
    Rows without usable text are dropped but the source row reference is preserved.
    """
    df_work = df.reset_index(drop=False).rename(columns={"index": "source_row"})
    df_work["record_id"] = df_work.index.astype("int64")

    if ID_COLUMN in df_work.columns:
        df_work["order_id"] = df_work[ID_COLUMN]
    else:
        df_work["order_id"] = df_work["record_id"]

    df_work["text_raw"] = combine_text_fields(df_work, TEXT_COLUMNS)
    df_work["text_normalized"] = df_work["text_raw"].apply(normalize_text)

    processed = df_work[df_work["text_normalized"].str.len() > 0].copy()
    dropped = len(df_work) - len(processed)
    if dropped:
        logger.info("Dropped %d rows without usable text", dropped)

    if VENDOR_COLUMN in processed.columns:
        processed["vendor"] = processed[VENDOR_COLUMN]
    else:
        processed["vendor"] = ""

    return processed[
        ["record_id", "source_row", "order_id", "vendor", "text_raw", "text_normalized"]
    ]


def _clean_commessa_value(val: object) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    if isinstance(val, str):
        return normalize_whitespace(val)
    return normalize_whitespace(str(val))


def _build_commessa_oggetto(row: pd.Series) -> str:
    """
    Prefer the explicit oggetto; if empty, fall back to a concise concatenation of
    nome_commessa + settore + committente.
    """
    raw = _clean_commessa_value(row.get("oggetto"))
    if raw:
        return raw
    fallback_parts = [
        _clean_commessa_value(row.get("nome_commessa")),
        _clean_commessa_value(row.get("settore")),
        _clean_commessa_value(row.get("committente")),
    ]
    parts = [p for p in fallback_parts if p]
    return normalize_whitespace(" - ".join(parts))


def preprocess_commesse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and prepare commesse rows for embedding: clean strings, fill oggetto,
    attach identifiers, and build normalized text fields.
    """
    work = df.rename(columns=COMMESSE_RENAME_MAP).copy()
    for col in COMMESSE_SOURCE_COLUMNS:
        if col not in work.columns:
            work[col] = None

    work["record_id"] = work.index.astype("int64")
    if COMMESSE_ID_COLUMN in work.columns:
        work["commessa_code"] = work[COMMESSE_ID_COLUMN].apply(_clean_commessa_value)
    else:
        work["commessa_code"] = work["record_id"].astype(str)

    for col in COMMESSE_NUMERIC_COLUMNS:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    for col in COMMESSE_DATE_COLUMNS:
        work[col] = pd.to_datetime(work.get(col), errors="coerce").dt.date
    for col in COMMESSE_SOURCE_COLUMNS:
        if col in COMMESSE_NUMERIC_COLUMNS or col in COMMESSE_DATE_COLUMNS:
            continue
        work[col] = work[col].apply(_clean_commessa_value)

    work["oggetto_filled"] = work.apply(_build_commessa_oggetto, axis=1)
    text_cols = list(dict.fromkeys(COMMESSE_TEXT_COLUMNS + ["oggetto_filled"]))
    work["text_raw"] = combine_text_fields(work, text_cols)
    work["text_normalized"] = work["text_raw"].apply(normalize_text)
    work["cliente"] = work.get(COMMESSE_CLIENT_COLUMN, "")

    processed = work[work["text_normalized"].str.len() > 0].copy()
    dropped = len(work) - len(processed)
    if dropped:
        logger.info("Dropped %d commessa rows without usable text", dropped)

    cols = [
        "record_id",
        "commessa_code",
        "text_raw",
        "text_normalized",
        "oggetto_filled",
        "cliente",
        *COMMESSE_SOURCE_COLUMNS,
    ]
    return processed[cols]
