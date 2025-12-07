from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
import psycopg

from ..config import (
    PGDATABASE,
    PGHOST,
    PGPORT,
    PGPASSWORD,
    PGUSER,
    PG_MAX_CONN,
    PG_MIN_CONN,
)
from ..embedding import embed_texts, l2_normalize
from ..preprocessing import normalize_text
from ..summarization import SUMMARY_COLUMN

logger = logging.getLogger(__name__)

conninfo = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
pool = ConnectionPool(
    conninfo=conninfo,
    min_size=PG_MIN_CONN,
    max_size=PG_MAX_CONN,
    configure=register_vector,
)

app = FastAPI(title="Orders similarity service (Postgres + pgvector)", version="1.0.0")


class SearchFilters(BaseModel):
    date_from: Optional[date] = Field(None, description="Include orders on/after this date")
    date_to: Optional[date] = Field(None, description="Include orders on/before this date")
    region: Optional[str] = Field(None, description="Region name to filter (ILIKE)")
    vendor: Optional[str] = Field(None, description="Vendor name to filter (ILIKE)")
    contract_type: Optional[str] = Field(None, description="Tipo contratto filter (ILIKE)")
    category: Optional[str] = Field(None, description="Tipologia filter (ILIKE)")
    order_type: Optional[str] = Field(None, description="Tipo ordine filter (ILIKE)")
    min_amount: Optional[float] = Field(None, description="Minimum order amount")
    max_amount: Optional[float] = Field(None, description="Maximum order amount")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Material or order description to match")
    top_k: int = Field(10, ge=1, le=200, description="Number of similar orders to return")
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SearchHit(BaseModel):
    record_id: int
    order_code: Optional[str]
    order_date: Optional[date]
    delivery_date: Optional[date]
    vendor: Optional[str]
    region: Optional[str]
    category: Optional[str]
    contract_type: Optional[str]
    order_type: Optional[str]
    amount: Optional[float]
    summary: Optional[str]
    similarity: float


class SearchResponse(BaseModel):
    query: str
    filters: SearchFilters
    results: List[SearchHit]


def _query_vector(text: str) -> list[float]:
    normalized = normalize_text(text)
    vec = embed_texts([normalized], log_prefix="[embed-query]")
    vec = l2_normalize(vec)[0].astype(float).tolist()
    return vec


def _build_where_clause(filters: SearchFilters, params: dict) -> str:
    clauses = ["summary_embedding IS NOT NULL"]
    if filters.region:
        clauses.append("region ILIKE %(region)s")
        params["region"] = f"%{filters.region}%"
    if filters.vendor:
        clauses.append("vendor ILIKE %(vendor)s")
        params["vendor"] = f"%{filters.vendor}%"
    if filters.contract_type:
        clauses.append("contract_type ILIKE %(contract_type)s")
        params["contract_type"] = f"%{filters.contract_type}%"
    if filters.category:
        clauses.append("category ILIKE %(category)s")
        params["category"] = f"%{filters.category}%"
    if filters.order_type:
        clauses.append("order_type ILIKE %(order_type)s")
        params["order_type"] = f"%{filters.order_type}%"
    if filters.date_from:
        clauses.append("order_date >= %(date_from)s")
        params["date_from"] = filters.date_from
    if filters.date_to:
        clauses.append("order_date <= %(date_to)s")
        params["date_to"] = filters.date_to
    if filters.min_amount is not None:
        clauses.append("amount >= %(min_amount)s")
        params["min_amount"] = filters.min_amount
    if filters.max_amount is not None:
        clauses.append("amount <= %(max_amount)s")
        params["max_amount"] = filters.max_amount
    return " AND ".join(clauses)


@app.on_event("startup")
def _startup() -> None:
    pool.open()
    logger.info("Connection pool opened (host=%s db=%s)", PGHOST, PGDATABASE)


@app.on_event("shutdown")
def _shutdown() -> None:
    pool.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _get_conn():
    with pool.connection() as conn:
        yield conn


@app.post("/search", response_model=SearchResponse)
def search_orders(payload: SearchRequest, conn: psycopg.Connection = Depends(_get_conn)) -> SearchResponse:
    query_vec = _query_vector(payload.query)
    params = {"vec": query_vec, "limit": payload.top_k}
    where_clause = _build_where_clause(payload.filters, params)

    sql = (
        "SELECT record_id, order_code, order_date, delivery_date, vendor, region, category, contract_type, "
        "order_type, amount, order_summary, (1 - (summary_embedding <=> %(vec)s::vector)) AS similarity "
        "FROM orders WHERE "
        + where_clause
        + " ORDER BY summary_embedding <=> %(vec)s::vector LIMIT %(limit)s"
    )

    rows = conn.execute(sql, params).fetchall()
    results = [
        SearchHit(
            record_id=r[0],
            order_code=r[1],
            order_date=r[2],
            delivery_date=r[3],
            vendor=r[4],
            region=r[5],
            category=r[6],
            contract_type=r[7],
            order_type=r[8],
            amount=float(r[9]) if r[9] is not None else None,
            summary=r[10],
            similarity=max(0.0, float(r[11])) if r[11] is not None else 0.0,
        )
        for r in rows
    ]

    return SearchResponse(query=payload.query, filters=payload.filters, results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bubbola_gare.service.api:app", host="0.0.0.0", port=8000, reload=False)
