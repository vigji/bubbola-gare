from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import psycopg
from openai import OpenAI
from psycopg.rows import dict_row

from ..config import (
    COMMESSE_SOURCE_COLUMNS,
    RAW_SOURCE_COLUMNS,
    SQL_DEFAULT_LIMIT,
    SQL_GENERATION_MODEL,
    SQL_MAX_LIMIT,
)
from ..embedding import get_openai_client
from ..preprocessing import normalize_text
from ..summarization import SUMMARY_COLUMN

logger = logging.getLogger(__name__)

ORDERS_COLUMNS: dict[str, str] = {
    "record_id": "Internal synthetic id assigned when preparing the DB.",
    "order_code": "Original order code (id) from the source system.",
    "order_date": "Order date.",
    "delivery_date": "Delivery date when present.",
    "category": "Tipologia / product category.",
    "contract_type": "Tipo contratto.",
    "order_type": "Tipo ordine.",
    "requester": "Richiedente column from source.",
    "executor": "Esecutore column from source.",
    "vendor": "Fornitore / supplier name.",
    "region": "Normalized region name.",
    "city": "City / comune text.",
    "province": "Province text.",
    "country": "Country text.",
    "cap": "Postal code.",
    "amount": "Order amount as numeric.",
    "text_raw": "Concatenated raw text across description columns.",
    "text_normalized": "Lowercased token-normalized version of text_raw.",
    "order_summary": "LLM-generated concise summary of the order (Italian).",
    "summary_normalized": "Normalized summary used for embeddings.",
    "summary_embedding": "Vector embedding of the normalized summary.",
}
for raw_col in RAW_SOURCE_COLUMNS:
    ORDERS_COLUMNS.setdefault(raw_col, "Raw column from the source file (kept verbatim).")

ORDERS_SCHEMA_TEXT = "\n".join(
    [
        "Dataset origin: ICOP SpA procurement — orders placed by ICOP to external suppliers.",
        "Table: orders (read-only)",
        *(f"- {name}: {desc}" for name, desc in ORDERS_COLUMNS.items()),
    ]
)

COMMESSE_COLUMNS: dict[str, str] = {
    "record_id": "Internal synthetic id for each commessa row.",
    "commessa_code": "Original num_commessa value from the Excel.",
    "ditta": "Ditta field from source.",
    "nome_commessa": "Project/commessa name.",
    "settore": "Settore column from source.",
    "mgo": "MGO column from source.",
    "committente": "Client / contracting entity.",
    "cliente": "Alias of committente (same content).",
    "ente_appaltante": "Ente appaltante column from source.",
    "oggetto": "Oggetto as provided in the source file.",
    "oggetto_filled": "Oggetto populated with nome_commessa + settore + committente when the original oggetto is empty.",
    "importo": "Importo column as numeric.",
    "consegna": "Consegna date.",
    "ultimazione": "Ultimazione/completion date.",
    "codice_gara": "Codice gara column.",
    "responsabile_commessa": "Responsabile commessa column.",
    "responsabile_cantiere": "Responsabile cantiere column.",
    "preposti": "Preposti column.",
    "text_raw": "Text used for embeddings (oggetto_filled).",
    "text_normalized": "Normalized version of text_raw.",
    "oggetto_embedding": "Vector embedding of oggetto_filled text.",
}
for raw_col in COMMESSE_SOURCE_COLUMNS:
    COMMESSE_COLUMNS.setdefault(raw_col, "Raw column from the commesse source file (kept verbatim).")

COMMESSE_SCHEMA_TEXT = "\n".join(
    [
        "Dataset origin: ICOP SpA commesse — works/projects executed by ICOP over the years.",
        "Table: commesse (read-only)",
        *(f"- {name}: {desc}" for name, desc in COMMESSE_COLUMNS.items()),
    ]
)

SQL_FORBIDDEN = {"insert", "update", "delete", "alter", "drop", "create", "grant", "copy", "truncate"}
TABLE_REGEX = re.compile(r"\\bfrom\\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
JOIN_REGEX = re.compile(r"\\bjoin\\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[dict]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "warnings": self.warnings,
        }


@dataclass
class NLQueryResult(QueryResult):
    question: str
    rationale: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({"question": self.question, "rationale": self.rationale, "error": self.error})
        return base


def _serialize_value(val: Any) -> Any:
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _allowed_tables(sql: str) -> set[str]:
    tables = set(TABLE_REGEX.findall(sql))
    tables |= set(JOIN_REGEX.findall(sql))
    return {t.lower() for t in tables}


def enforce_safe_sql(
    sql: str,
    default_limit: int = SQL_DEFAULT_LIMIT,
    max_limit: int = SQL_MAX_LIMIT,
    allowed_tables: Iterable[str] | None = None,
) -> Tuple[str, List[str]]:
    """
    Ensure the SQL is read-only, targets allowed tables, and is limited.
    Returns the sanitized SQL and any warnings applied.
    """
    allowed = {t.lower() for t in (allowed_tables or {"orders"})}
    warnings: List[str] = []

    cleaned = sql.strip().rstrip(";")
    lowered = cleaned.lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    if any(bad in lowered for bad in SQL_FORBIDDEN):
        raise ValueError("Mutating SQL statements are not allowed.")

    tables = _allowed_tables(cleaned)
    if tables and not tables.issubset(allowed):
        raise ValueError(f"Query references tables outside allowlist: {tables - allowed}")

    limit_match = re.search(r"limit\s+(\d+)", lowered)
    if limit_match:
        limit_val = int(limit_match.group(1))
        if limit_val > max_limit:
            warnings.append(f"LIMIT reduced from {limit_val} to {max_limit}")
            cleaned = re.sub(r"limit\s+\d+", f"LIMIT {max_limit}", cleaned, flags=re.IGNORECASE)
    else:
        cleaned = f"{cleaned} LIMIT {default_limit}"
        warnings.append(f"Added default LIMIT {default_limit}")

    return cleaned, warnings


def execute_sql(conn: psycopg.Connection, sql: str, params: Dict[str, Any] | None = None) -> QueryResult:
    # psycopg treats `%` as a placeholder marker; double it to keep literal patterns (e.g., ILIKE '%foo%').
    safe_sql = sql.replace("%", "%%")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(safe_sql, params or {})
        rows = cur.fetchall()
        columns = [d.name for d in (cur.description or [])]
    serialized = [{k: _serialize_value(v) for k, v in row.items()} for row in rows]
    return QueryResult(sql=safe_sql, columns=columns, rows=serialized, warnings=[])


@dataclass
class SQLPlan:
    sql: str
    rationale: str | None


class SQLGenerator:
    """
    Generates read-only SQL for the orders table via an LLM.
    """

    def __init__(self, client: OpenAI | None = None, model: str = SQL_GENERATION_MODEL):
        self.client = client or get_openai_client()
        self.model = model

    def generate(self, question: str, limit: int = SQL_DEFAULT_LIMIT) -> SQLPlan:
        prompt = (
            "You are an expert data analyst that writes safe, read-only PostgreSQL for a single table `orders`.\n"
            "Only use SELECT/CTE queries. Never modify data. Always include a LIMIT.\n"
            "When summing money use COALESCE(amount, 0). For counts use COUNT(*).\n"
            "Prefer human readable column aliases in English or Italian when appropriate.\n"
            "If the question is about suppliers/companies, map that to the `vendor` column.\n"
            "If dates are requested, use order_date unless otherwise specified.\n"
            "Schema:\n"
            f"{ORDERS_SCHEMA_TEXT}\n\n"
            "Example intents:\n"
            "- \"5 companies for order X\" -> select vendor, order_code, amount where order_code matches X, limit 5\n"
            "- \"what is total amount I ordered for company X\" -> sum(amount) grouped by vendor filtered by vendor ilike X\n"
            "- \"top 10 vendors in Lombardia in 2024\" -> group by vendor with region filter and date range\n"
            "- \"monthly spend by category\" -> group by date_trunc('month', order_date) and category with sums\n"
        )
        user_content = (
            f"Question: {question}\\n"
            f"Return JSON with keys: sql (string), rationale (short string). "
            f"Limit results to {limit} rows unless the question is aggregate."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            sql = str(parsed.get("sql", "")).strip()
            rationale = parsed.get("rationale")
            if not sql:
                raise ValueError("Empty SQL returned from model.")
            return SQLPlan(sql=sql, rationale=rationale)
        except Exception as exc:
            logger.error("Failed to generate SQL: %s", exc)
            raise


class AnalyticsEngine:
    def __init__(
        self,
        pool,
        sql_generator: SQLGenerator | None = None,
        default_limit: int = SQL_DEFAULT_LIMIT,
        max_limit: int = SQL_MAX_LIMIT,
    ):
        self.pool = pool
        self.sql_generator = sql_generator or SQLGenerator()
        self.default_limit = default_limit
        self.max_limit = max_limit
        self._search_fields = {
            "region",
            "vendor",
            "contract_type",
            "category",
            "order_type",
            "date_from",
            "date_to",
            "min_amount",
            "max_amount",
        }

    def run_sql(
        self,
        sql: str,
        params: Dict[str, Any] | None = None,
        *,
        limit: int | None = None,
        conn: psycopg.Connection | None = None,
    ) -> QueryResult:
        enforced_limit = min(limit or self.default_limit, self.max_limit)
        safe_sql, warnings = enforce_safe_sql(
            sql,
            default_limit=enforced_limit,
            max_limit=self.max_limit,
            allowed_tables={"orders", "commesse"},
        )

        def _execute(active_conn: psycopg.Connection) -> QueryResult:
            result = execute_sql(active_conn, safe_sql, params=params)
            result.warnings.extend(warnings)
            return result

        if conn is not None:
            return _execute(conn)

        with self.pool.connection() as pooled_conn:
            return _execute(pooled_conn)

    def run_nl(
        self,
        question: str,
        *,
        limit: int | None = None,
        conn: psycopg.Connection | None = None,
    ) -> NLQueryResult:
        effective_limit = min(limit or self.default_limit, self.max_limit)
        try:
            plan = self.sql_generator.generate(question=question, limit=effective_limit)
            result = self.run_sql(plan.sql, params=None, limit=effective_limit, conn=conn)
            return NLQueryResult(
                question=question,
                sql=result.sql,
                rows=result.rows,
                columns=result.columns,
                warnings=result.warnings,
                rationale=plan.rationale,
            )
        except Exception as exc:
            logger.error("NL query failed: %s", exc)
            return NLQueryResult(
                question=question,
                sql="",
                rows=[],
                columns=[],
                warnings=[],
                rationale=None,
                error=str(exc),
            )

    def _build_where(self, filters: Dict[str, Any], params: Dict[str, Any]) -> str:
        clauses = ["summary_embedding IS NOT NULL"]
        if filters.get("region"):
            clauses.append("region ILIKE %(region)s")
            params["region"] = f"%{filters['region']}%"
        if filters.get("vendor"):
            clauses.append("vendor ILIKE %(vendor)s")
            params["vendor"] = f"%{filters['vendor']}%"
        if filters.get("contract_type"):
            clauses.append("contract_type ILIKE %(contract_type)s")
            params["contract_type"] = f"%{filters['contract_type']}%"
        if filters.get("category"):
            clauses.append("category ILIKE %(category)s")
            params["category"] = f"%{filters['category']}%"
        if filters.get("order_type"):
            clauses.append("order_type ILIKE %(order_type)s")
            params["order_type"] = f"%{filters['order_type']}%"
        if filters.get("date_from"):
            clauses.append("order_date >= %(date_from)s")
            params["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            clauses.append("order_date <= %(date_to)s")
            params["date_to"] = filters["date_to"]
        if filters.get("min_amount") is not None:
            clauses.append("amount >= %(min_amount)s")
            params["min_amount"] = filters["min_amount"]
        if filters.get("max_amount") is not None:
            clauses.append("amount <= %(max_amount)s")
            params["max_amount"] = filters["max_amount"]
        return " AND ".join(clauses)

    def semantic_search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
        conn: psycopg.Connection | None = None,
    ) -> list[dict]:
        filt = {k: v for k, v in (filters or {}).items() if k in self._search_fields and v is not None}
        params = {"vec": query_vector, "limit": top_k}
        where_clause = self._build_where(filt, params)

        sql = (
            "SELECT record_id, order_code, order_date, delivery_date, vendor, region, category, contract_type, "
            "order_type, amount, order_summary, (1 - (summary_embedding <=> %(vec)s::vector)) AS similarity "
            "FROM orders WHERE "
            + where_clause
            + " ORDER BY summary_embedding <=> %(vec)s::vector LIMIT %(limit)s"
        )

        def _run(active_conn: psycopg.Connection) -> list[dict]:
            with active_conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            for r in rows:
                if "similarity" in r and r["similarity"] is not None:
                    r["similarity"] = max(0.0, float(r["similarity"]))
                if "amount" in r and r["amount"] is not None:
                    r["amount"] = float(r["amount"])
            return rows

        if conn is not None:
            return _run(conn)

        with self.pool.connection() as pooled_conn:
            return _run(pooled_conn)

    def vector_for_order_code(self, order_code: str, conn: psycopg.Connection | None = None) -> list[float]:
        def _run(active_conn: psycopg.Connection) -> list[float]:
            with active_conn.cursor() as cur:
                cur.execute(
                    "SELECT summary_embedding FROM orders WHERE order_code = %s OR CAST(record_id AS TEXT) = %s LIMIT 1",
                    (order_code, order_code),
                )
                row = cur.fetchone()
            if not row or row[0] is None:
                raise ValueError(f"No embedding found for order code {order_code}")
            emb = row[0]
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            return [float(x) for x in emb]

        if conn is not None:
            return _run(conn)
        with self.pool.connection() as pooled_conn:
            return _run(pooled_conn)
