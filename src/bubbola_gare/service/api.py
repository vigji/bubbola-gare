from __future__ import annotations

import asyncio
import logging
import time
from datetime import date
from typing import List, Optional

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import PGDATABASE, PGHOST, SQL_DEFAULT_LIMIT, SQL_MAX_LIMIT
from ..embedding import embed_query_vectors
from ..preprocessing import normalize_text
from ..summarization import SUMMARY_COLUMN
from .chat_gateway import ChatResult, chat_gateway
from .analytics import AnalyticsEngine, ORDERS_SCHEMA_TEXT
from .db import connection_scope, get_pool

logger = logging.getLogger(__name__)

app = FastAPI(title="Orders similarity service (Postgres + pgvector)", version="1.0.0")
analytics_engine = AnalyticsEngine(pool=get_pool())


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


class SqlQueryRequest(BaseModel):
    sql: str = Field(..., description="Read-only SQL to run against the orders table")
    params: dict = Field(default_factory=dict, description="Optional parameters for the SQL query")
    max_rows: int = Field(
        SQL_DEFAULT_LIMIT, ge=1, le=SQL_MAX_LIMIT, description="Maximum rows to return (LIMIT enforced)"
    )


class SqlQueryResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[dict]
    warnings: List[str] = Field(default_factory=list)


class NLQueryRequest(BaseModel):
    question: str = Field(..., description="Natural language analytics request")
    max_rows: int = Field(
        SQL_DEFAULT_LIMIT, ge=1, le=SQL_MAX_LIMIT, description="Maximum rows to return (LIMIT enforced)"
    )


class NLQueryResponse(BaseModel):
    question: str
    sql: str
    rationale: Optional[str]
    columns: List[str]
    rows: List[dict]
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class SchemaResponse(BaseModel):
    schema: str
    columns: List[str]


class SemanticQueryRequest(BaseModel):
    query: Optional[str] = Field(None, description="Free-text description to find similar orders")
    order_code: Optional[str] = Field(
        None, description="Existing order code to use as the anchor for similarity search"
    )
    top_k: int = Field(5, ge=1, le=200, description="Number of similar orders to return")
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SemanticQueryResponse(BaseModel):
    request: SemanticQueryRequest
    results: List[SearchHit]


class ChatRequest(BaseModel):
    question: str = Field(..., description="Plaintext question for the chat gateway (MCP-backed)")


class ChatResponse(BaseModel):
    answer: str
    tool_used: str
    tool_arguments: dict
    tool_result: dict
    token_usage: dict
    model: str


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.2


def _query_vector(text: str) -> list[float]:
    normalized = normalize_text(text)
    vec = embed_query_vectors([normalized])[0]
    return vec.astype(float).tolist()


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
    get_pool().open()
    logger.info("Connection pool opened (host=%s db=%s)", PGHOST, PGDATABASE)


@app.on_event("shutdown")
def _shutdown() -> None:
    get_pool().close()
    try:
        # Close MCP client if it was opened
        conn = chat_gateway.mcp
        if conn:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(conn.close())
            else:
                loop.run_until_complete(conn.close())
    except Exception:
        logger.warning("Failed to close MCP client cleanly", exc_info=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

def _get_conn():
    yield from connection_scope()


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


@app.get("/analytics/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    from .analytics import ORDERS_COLUMNS

    return SchemaResponse(schema=ORDERS_SCHEMA_TEXT, columns=list(ORDERS_COLUMNS.keys()))


@app.post("/analytics/sql", response_model=SqlQueryResponse)
def run_sql_query(payload: SqlQueryRequest, conn: psycopg.Connection = Depends(_get_conn)) -> SqlQueryResponse:
    result = analytics_engine.run_sql(payload.sql, params=payload.params, limit=payload.max_rows, conn=conn)
    return SqlQueryResponse(sql=result.sql, columns=result.columns, rows=result.rows, warnings=result.warnings)


@app.post("/analytics/nlq", response_model=NLQueryResponse)
def run_nl_query(payload: NLQueryRequest, conn: psycopg.Connection = Depends(_get_conn)) -> NLQueryResponse:
    result = analytics_engine.run_nl(question=payload.question, limit=payload.max_rows, conn=conn)
    return NLQueryResponse(
        question=result.question,
        sql=result.sql,
        rationale=result.rationale,
        columns=result.columns,
        rows=result.rows,
        warnings=result.warnings,
        error=result.error,
    )


def _resolve_query_vector(payload: SemanticQueryRequest, conn: psycopg.Connection) -> list[float]:
    if payload.query:
        return _query_vector(payload.query)
    if payload.order_code:
        return analytics_engine.vector_for_order_code(payload.order_code, conn=conn)
    raise HTTPException(status_code=400, detail="Provide either 'query' or 'order_code'.")


@app.post("/analytics/semantic", response_model=SemanticQueryResponse)
def semantic_search(payload: SemanticQueryRequest, conn: psycopg.Connection = Depends(_get_conn)) -> SemanticQueryResponse:
    if not payload.query and not payload.order_code:
        raise HTTPException(status_code=400, detail="Provide either 'query' or 'order_code'.")

    try:
        query_vec = _resolve_query_vector(payload, conn)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    results = analytics_engine.semantic_search(
        query_vector=query_vec,
        top_k=payload.top_k,
        filters=payload.filters.model_dump(),
        conn=conn,
    )
    hits = [
        SearchHit(
            record_id=r.get("record_id"),
            order_code=r.get("order_code"),
            order_date=r.get("order_date"),
            delivery_date=r.get("delivery_date"),
            vendor=r.get("vendor"),
            region=r.get("region"),
            category=r.get("category"),
            contract_type=r.get("contract_type"),
            order_type=r.get("order_type"),
            amount=r.get("amount"),
            summary=r.get(SUMMARY_COLUMN),
            similarity=r.get("similarity", 0.0),
        )
        for r in results
    ]
    return SemanticQueryResponse(request=payload, results=hits)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    result: ChatResult = await chat_gateway.chat(payload.question)
    return ChatResponse(**result.to_dict())


@app.post("/v1/chat/completions")
async def openai_compatible_chat(payload: OpenAIChatRequest) -> dict:
    # Use the last user message as the question
    user_messages = [m for m in payload.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found.")
    question = user_messages[-1].content
    result: ChatResult = await chat_gateway.chat(question)
    created = int(time.time())
    usage = result.token_usage or {}
    return {
        "id": "chatcmpl-mcp",
        "object": "chat.completion",
        "created": created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bubbola_gare.service.api:app", host="0.0.0.0", port=8000, reload=False)
