from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

from .config import (
    DB_READY_PATH,
    PGDATABASE,
    PGHOST,
    PGPORT,
    PGPASSWORD,
    PGUSER,
)

logger = logging.getLogger(__name__)


def _conn_str(database: str | None = None) -> str:
    db = database or PGDATABASE
    return f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{db}"


def ensure_schema(conn: psycopg.Connection, embed_dim: int) -> None:
    with conn.cursor() as cur:
        # Recreate table to ensure column type matches current setting (vector, not legacy halfvec/vector mix).
        cur.execute("DROP TABLE IF EXISTS orders CASCADE;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS orders (
                record_id BIGINT PRIMARY KEY,
                order_code TEXT,
                order_date DATE,
                delivery_date DATE,
                category TEXT,
                contract_type TEXT,
                order_type TEXT,
                requester TEXT,
                executor TEXT,
                vendor TEXT,
                region TEXT,
                city TEXT,
                province TEXT,
                country TEXT,
                cap TEXT,
                amount NUMERIC,
                text_raw TEXT,
                text_normalized TEXT,
                order_summary TEXT,
                summary_normalized TEXT,
                summary_embedding vector({embed_dim}) NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS orders_region_idx ON orders (region);")
        cur.execute("CREATE INDEX IF NOT EXISTS orders_order_date_idx ON orders (order_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS orders_vendor_idx ON orders (vendor);")
        if embed_dim <= 2000:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS orders_summary_embedding_idx ON orders "
                "USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 200);"
            )
        else:
            logger.warning(
                "Skipping ivfflat index: vector dim %d exceeds 2000; queries will scan",
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
        yield {
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
            "summary_embedding": emb if emb is not None else None,
        }


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

    rows = list(_prepare_rows(df))
    query = (
        "INSERT INTO orders (record_id, order_code, order_date, delivery_date, category, contract_type, "
        "order_type, requester, executor, vendor, region, city, province, country, cap, amount, text_raw, "
        "text_normalized, order_summary, summary_normalized, summary_embedding) "
        "VALUES (%(record_id)s, %(order_code)s, %(order_date)s, %(delivery_date)s, %(category)s, %(contract_type)s, "
        "%(order_type)s, %(requester)s, %(executor)s, %(vendor)s, %(region)s, %(city)s, %(province)s, %(country)s, "
        "%(cap)s, %(amount)s, %(text_raw)s, %(text_normalized)s, %(order_summary)s, %(summary_normalized)s, %(summary_embedding)s::vector) "
        "ON CONFLICT (record_id) DO UPDATE SET "
        "order_code = EXCLUDED.order_code, order_date = EXCLUDED.order_date, delivery_date = EXCLUDED.delivery_date, "
        "category = EXCLUDED.category, contract_type = EXCLUDED.contract_type, order_type = EXCLUDED.order_type, "
        "requester = EXCLUDED.requester, executor = EXCLUDED.executor, vendor = EXCLUDED.vendor, "
        "region = EXCLUDED.region, city = EXCLUDED.city, province = EXCLUDED.province, country = EXCLUDED.country, "
        "cap = EXCLUDED.cap, amount = EXCLUDED.amount, text_raw = EXCLUDED.text_raw, "
        "text_normalized = EXCLUDED.text_normalized, order_summary = EXCLUDED.order_summary, "
        "summary_normalized = EXCLUDED.summary_normalized, summary_embedding = EXCLUDED.summary_embedding"
    )
    with conn.cursor() as cur:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            if not batch:
                continue
            cur.executemany(query, batch)
        conn.commit()
    logger.info("Loaded %d rows into orders table", len(rows))


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load cleaned orders into Postgres with pgvector")
    parser.add_argument("--path", type=Path, default=DB_READY_PATH, help="Path to DB-ready parquet")
    parser.add_argument("--truncate", action="store_true", help="Truncate table before loading")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for inserts")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    load_parquet_to_db(path=args.path, truncate=args.truncate, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
