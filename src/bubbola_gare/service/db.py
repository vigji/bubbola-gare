from __future__ import annotations

import logging
from typing import Iterator

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ..config import (
    PGDATABASE,
    PGHOST,
    PGPORT,
    PGPASSWORD,
    PGUSER,
    PG_MAX_CONN,
    PG_MIN_CONN,
)

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _conninfo() -> str:
    return f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_conninfo(),
            min_size=PG_MIN_CONN,
            max_size=PG_MAX_CONN,
            configure=register_vector,
        )
        logger.info("Created connection pool (host=%s db=%s)", PGHOST, PGDATABASE)
    return _pool


def connection_scope() -> Iterator:
    """
    FastAPI dependency to yield a pooled connection.
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn
