from __future__ import annotations

import argparse
import logging
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

from .config import (
    COMMESSE_DB_READY_PATH,
    COMMESSE_DATE_COLUMNS,
    COMMESSE_NUMERIC_COLUMNS,
    COMMESSE_SOURCE_COLUMNS,
    DB_READY_PATH,
    PGDATABASE,
    PGHOST,
    PGPORT,
    PGPASSWORD,
    PGUSER,
    RAW_DATE_COLUMNS,
    RAW_NUMERIC_COLUMNS,
RAW_SOURCE_COLUMNS,
)

logger = logging.getLogger(__name__)


def _conn_str(database: str | None = None) -> str:
    db = database or PGDATABASE
    return f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{db}"


BASE_COLUMN_TYPES = {
    "record_id": "BIGINT PRIMARY KEY",
    "order_code": "TEXT",
    "order_date": "DATE",
    "delivery_date": "DATE",
    "category": "TEXT",
    "contract_type": "TEXT",
    "order_type": "TEXT",
    "requester": "TEXT",
    "executor": "TEXT",
    "vendor": "TEXT",
    "region": "TEXT",
    "city": "TEXT",
    "province": "TEXT",
    "country": "TEXT",
    "cap": "TEXT",
    "amount": "NUMERIC",
    "text_raw": "TEXT",
    "text_normalized": "TEXT",
    "order_summary": "TEXT",
    "summary_normalized": "TEXT",
}
RAW_SOURCE_COLUMNS_UNIQUE = [c for c in RAW_SOURCE_COLUMNS if c not in BASE_COLUMN_TYPES]

COMMESSE_BASE_COLUMN_TYPES = {
    "record_id": "BIGINT PRIMARY KEY",
    "commessa_code": "TEXT",
    "ditta": "TEXT",
    "nome_commessa": "TEXT",
    "settore": "TEXT",
    "mgo": "TEXT",
    "committente": "TEXT",
    "cliente": "TEXT",
    "ente_appaltante": "TEXT",
    "oggetto": "TEXT",
    "oggetto_filled": "TEXT",
    "importo": "NUMERIC",
    "consegna": "DATE",
    "ultimazione": "DATE",
    "codice_gara": "TEXT",
    "responsabile_commessa": "TEXT",
    "responsabile_cantiere": "TEXT",
    "preposti": "TEXT",
    "text_raw": "TEXT",
    "text_normalized": "TEXT",
}
COMMESSE_RAW_COLUMNS_UNIQUE = [
    c for c in COMMESSE_SOURCE_COLUMNS if c not in COMMESSE_BASE_COLUMN_TYPES
]
COMMESSE_VECTOR_COLUMN = "oggetto_embedding"


def _raw_column_type(col: str) -> str:
    if col in RAW_NUMERIC_COLUMNS:
        return "NUMERIC"
    if col in RAW_DATE_COLUMNS:
        return "DATE"
    return "TEXT"


def _commesse_column_type(col: str) -> str:
    if col in COMMESSE_NUMERIC_COLUMNS:
        return "NUMERIC"
    if col in COMMESSE_DATE_COLUMNS:
        return "DATE"
    return "TEXT"


def ensure_schema(conn: psycopg.Connection, embed_dim: int) -> None:
    with conn.cursor() as cur:
        # Recreate table to ensure column type matches current setting (vector, not legacy halfvec/vector mix).
        cur.execute("DROP TABLE IF EXISTS orders CASCADE;")
        column_defs = [
            *(f"\"{col}\" {ctype}" for col, ctype in BASE_COLUMN_TYPES.items()),
            *(f"\"{col}\" {_raw_column_type(col)}" for col in RAW_SOURCE_COLUMNS_UNIQUE),
            f"\"summary_embedding\" vector({embed_dim}) NOT NULL",
        ]
        cur.execute(f"CREATE TABLE IF NOT EXISTS orders ({', '.join(column_defs)});")
        cur.execute('CREATE INDEX IF NOT EXISTS orders_region_idx ON orders ("region");')
        cur.execute('CREATE INDEX IF NOT EXISTS orders_order_date_idx ON orders ("order_date");')
        cur.execute('CREATE INDEX IF NOT EXISTS orders_vendor_idx ON orders ("vendor");')
        if embed_dim <= 2000:
            cur.execute(
                'CREATE INDEX IF NOT EXISTS orders_summary_embedding_idx ON orders '
                'USING ivfflat ("summary_embedding" vector_cosine_ops) WITH (lists = 200);'
            )
        else:
            logger.warning(
                "Skipping ivfflat index: vector dim %d exceeds 2000; queries will scan",
                embed_dim,
            )
    conn.commit()


def ensure_commesse_schema(conn: psycopg.Connection, embed_dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS commesse CASCADE;")
        column_defs = [
            *(f"\"{col}\" {ctype}" for col, ctype in COMMESSE_BASE_COLUMN_TYPES.items()),
            *(f"\"{col}\" {_commesse_column_type(col)}" for col in COMMESSE_RAW_COLUMNS_UNIQUE),
            f"\"{COMMESSE_VECTOR_COLUMN}\" vector({embed_dim}) NOT NULL",
        ]
        cur.execute(f"CREATE TABLE IF NOT EXISTS commesse ({', '.join(column_defs)});")
        cur.execute('CREATE INDEX IF NOT EXISTS commesse_code_idx ON commesse ("commessa_code");')
        cur.execute('CREATE INDEX IF NOT EXISTS commesse_committente_idx ON commesse ("committente");')
        cur.execute('CREATE INDEX IF NOT EXISTS commesse_consegna_idx ON commesse ("consegna");')
        if embed_dim <= 2000:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS commesse_{COMMESSE_VECTOR_COLUMN}_idx ON commesse '
                f'USING ivfflat ("{COMMESSE_VECTOR_COLUMN}" vector_cosine_ops) WITH (lists = 100);'
            )
        else:
            logger.warning(
                "Skipping ivfflat index for commesse: vector dim %d exceeds 2000; queries will scan",
                embed_dim,
            )
    conn.commit()


def ensure_vector_extension(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()


def _null_if_na(value: object):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _prepare_rows(df: pd.DataFrame) -> Iterable[dict]:
    for row in df.itertuples(index=False):
        emb = getattr(row, "summary_embedding", None)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        base = {
            "record_id": int(row.record_id),
            "order_code": _null_if_na(getattr(row, "order_code", None)),
            "order_date": _null_if_na(getattr(row, "order_date", None)),
            "delivery_date": _null_if_na(getattr(row, "delivery_date", None)),
            "category": _null_if_na(getattr(row, "category", None)),
            "contract_type": _null_if_na(getattr(row, "contract_type", None)),
            "order_type": _null_if_na(getattr(row, "order_type", None)),
            "requester": _null_if_na(getattr(row, "requester", None)),
            "executor": _null_if_na(getattr(row, "executor", None)),
            "vendor": _null_if_na(getattr(row, "vendor", None)),
            "region": _null_if_na(getattr(row, "region", None)),
            "city": _null_if_na(getattr(row, "city", None)),
            "province": _null_if_na(getattr(row, "province", None)),
            "country": _null_if_na(getattr(row, "country", None)),
            "cap": _null_if_na(getattr(row, "cap", None)),
            "amount": _null_if_na(getattr(row, "amount", None)),
            "text_raw": _null_if_na(getattr(row, "text_raw", None)),
            "text_normalized": _null_if_na(getattr(row, "text_normalized", None)),
            "order_summary": _null_if_na(getattr(row, "order_summary", None)),
            "summary_normalized": _null_if_na(getattr(row, "summary_normalized", None)),
        }
        for col in RAW_SOURCE_COLUMNS_UNIQUE:
            base[col] = _null_if_na(getattr(row, col, None))
        base["summary_embedding"] = emb if emb is not None else None
        yield base


def _prepare_commesse_rows(df: pd.DataFrame) -> Iterable[dict]:
    for row in df.itertuples(index=False):
        emb = getattr(row, COMMESSE_VECTOR_COLUMN, None)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        base = {
            "record_id": int(row.record_id),
            "commessa_code": _null_if_na(getattr(row, "commessa_code", None)),
            "ditta": _null_if_na(getattr(row, "ditta", None)),
            "nome_commessa": _null_if_na(getattr(row, "nome_commessa", None)),
            "settore": _null_if_na(getattr(row, "settore", None)),
            "mgo": _null_if_na(getattr(row, "mgo", None)),
            "committente": _null_if_na(getattr(row, "committente", None)),
            "cliente": _null_if_na(getattr(row, "cliente", None)),
            "ente_appaltante": _null_if_na(getattr(row, "ente_appaltante", None)),
            "oggetto": _null_if_na(getattr(row, "oggetto", None)),
            "oggetto_filled": _null_if_na(getattr(row, "oggetto_filled", None)),
            "importo": _null_if_na(getattr(row, "importo", None)),
            "consegna": _null_if_na(getattr(row, "consegna", None)),
            "ultimazione": _null_if_na(getattr(row, "ultimazione", None)),
            "codice_gara": _null_if_na(getattr(row, "codice_gara", None)),
            "responsabile_commessa": _null_if_na(getattr(row, "responsabile_commessa", None)),
            "responsabile_cantiere": _null_if_na(getattr(row, "responsabile_cantiere", None)),
            "preposti": _null_if_na(getattr(row, "preposti", None)),
            "text_raw": _null_if_na(getattr(row, "text_raw", None)),
            "text_normalized": _null_if_na(getattr(row, "text_normalized", None)),
        }
        for col in COMMESSE_RAW_COLUMNS_UNIQUE:
            base[col] = _null_if_na(getattr(row, col, None))
        base[COMMESSE_VECTOR_COLUMN] = emb if emb is not None else None
        yield base


def load_orders(
    df: pd.DataFrame,
    conn: psycopg.Connection,
    truncate: bool = False,
    batch_size: int = 500,
) -> None:
    ensure_schema(conn, embed_dim=len(df.iloc[0].summary_embedding))
    with conn.cursor() as cur:
        if truncate:
            cur.execute("TRUNCATE TABLE orders;")
            logger.info("Truncated existing rows")
    conn.commit()

    insert_columns = list(BASE_COLUMN_TYPES.keys()) + list(RAW_SOURCE_COLUMNS_UNIQUE) + ["summary_embedding"]
    insert_columns_sql = ", ".join(f"\"{c}\"" for c in insert_columns)
    values_clause = ", ".join(
        [f"%({col})s" if col != "summary_embedding" else "%(summary_embedding)s::vector" for col in insert_columns]
    )
    update_clause = ", ".join(
        [f"\"{col}\" = EXCLUDED.\"{col}\"" for col in insert_columns if col != "record_id"]
    )
    query = (
        f"INSERT INTO orders ({insert_columns_sql}) VALUES ({values_clause}) "
        f"ON CONFLICT (record_id) DO UPDATE SET {update_clause}"
    )
    with conn.cursor() as cur:
        rows_iter = _prepare_rows(df)
        total = 0
        while True:
            batch = list(islice(rows_iter, batch_size))
            if not batch:
                break
            cur.executemany(query, batch)
            conn.commit()  # commit per batch to keep memory/txn small
            total += len(batch)
        logger.info("Loaded %d rows into orders table", total)


def load_commesse(
    df: pd.DataFrame,
    conn: psycopg.Connection,
    truncate: bool = False,
    batch_size: int = 500,
) -> None:
    if COMMESSE_VECTOR_COLUMN not in df.columns:
        raise ValueError(f"{COMMESSE_VECTOR_COLUMN} column missing from commesse dataframe")
    ensure_commesse_schema(conn, embed_dim=len(df.iloc[0][COMMESSE_VECTOR_COLUMN]))
    with conn.cursor() as cur:
        if truncate:
            cur.execute("TRUNCATE TABLE commesse;")
            logger.info("Truncated existing commesse rows")
    conn.commit()

    insert_columns = (
        list(COMMESSE_BASE_COLUMN_TYPES.keys()) + list(COMMESSE_RAW_COLUMNS_UNIQUE) + [COMMESSE_VECTOR_COLUMN]
    )
    insert_columns_sql = ", ".join(f"\"{c}\"" for c in insert_columns)
    values_clause = ", ".join(
        [f"%({col})s" if col != COMMESSE_VECTOR_COLUMN else f"%({COMMESSE_VECTOR_COLUMN})s::vector" for col in insert_columns]
    )
    update_clause = ", ".join(
        [f"\"{col}\" = EXCLUDED.\"{col}\"" for col in insert_columns if col != "record_id"]
    )
    query = (
        f"INSERT INTO commesse ({insert_columns_sql}) VALUES ({values_clause}) "
        f"ON CONFLICT (record_id) DO UPDATE SET {update_clause}"
    )
    with conn.cursor() as cur:
        rows_iter = _prepare_commesse_rows(df)
        total = 0
        while True:
            batch = list(islice(rows_iter, batch_size))
            if not batch:
                break
            cur.executemany(query, batch)
            conn.commit()
            total += len(batch)
        logger.info("Loaded %d rows into commesse table", total)


def load_parquet_to_db(
    path: Path = DB_READY_PATH,
    truncate: bool = False,
    batch_size: int = 500,
) -> None:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("DB-ready dataset is empty; nothing to load")
    conn = psycopg.connect(_conn_str(), autocommit=False)
    ensure_vector_extension(conn)
    register_vector(conn)
    try:
        load_orders(df, conn, truncate=truncate, batch_size=batch_size)
    finally:
        conn.close()


def load_commesse_parquet_to_db(
    path: Path = COMMESSE_DB_READY_PATH,
    truncate: bool = False,
    batch_size: int = 500,
) -> None:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("DB-ready commesse dataset is empty; nothing to load")
    conn = psycopg.connect(_conn_str(), autocommit=False)
    ensure_vector_extension(conn)
    register_vector(conn)
    try:
        load_commesse(df, conn, truncate=truncate, batch_size=batch_size)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load cleaned orders into Postgres with pgvector")
    parser.add_argument("--path", type=Path, default=DB_READY_PATH, help="Path to DB-ready parquet")
    parser.add_argument("--truncate", action="store_true", help="Truncate table before loading")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for inserts")
    parser.add_argument("--commesse-path", type=Path, default=None, help="Path to DB-ready commesse parquet (optional)")
    parser.add_argument("--truncate-commesse", action="store_true", help="Truncate commesse table before loading")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    load_parquet_to_db(path=args.path, truncate=args.truncate, batch_size=args.batch_size)
    if args.commesse_path:
        if Path(args.commesse_path).exists():
            load_commesse_parquet_to_db(
                path=args.commesse_path,
                truncate=args.truncate_commesse,
                batch_size=args.batch_size,
            )
        else:
            logger.warning("Skipping commesse load: file not found at %s", args.commesse_path)


if __name__ == "__main__":
    main()
