from __future__ import annotations

import argparse
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..config import SQL_DEFAULT_LIMIT, SQL_GENERATION_MODEL
from ..embedding import embed_query_vectors
from ..preprocessing import normalize_text
from .analytics import (
    AnalyticsEngine,
    COMMESSE_COLUMNS,
    COMMESSE_SCHEMA_TEXT,
    ORDERS_COLUMNS,
    ORDERS_SCHEMA_TEXT,
)
from .db import get_pool
from ..embedding import get_openai_client
from openai import OpenAI


def build_server(host: str = "0.0.0.0", port: int = 8100) -> FastMCP:
    """
    Configure the MCP server with analytics tools.
    """
    pool = get_pool()
    pool.open()
    engine = AnalyticsEngine(pool=pool)
    llm_client: OpenAI = get_openai_client()

    instructions = (
        "Interact with the `orders` (purchase orders) and `commesse` (projects) tables. Use tools to inspect schema, "
        "run SQL, ask NL questions, or run semantic similarity searches. All queries are read-only and limited. "
        "Helpful intents: from the orders table: check vendor totals, top suppliers, spend over time, order lookups by code/region/category, "
        "finding orders similar to a given order or text"
        "from the commesse table: browse historical data, importi, looking up commesse responsabili/committente and "
        "similar projects by oggetto."
    )
    server = FastMCP(name="orders-mcp", instructions=instructions, host=host, port=port, log_level="INFO")

    @server.tool(name="list_schema", description="Show available columns and descriptions for orders and commesse tables.")
    def list_schema() -> dict[str, Any]:
        return {
            "schema": ORDERS_SCHEMA_TEXT + "\n\n" + COMMESSE_SCHEMA_TEXT,
            "columns": {"orders": ORDERS_COLUMNS, "commesse": COMMESSE_COLUMNS},
        }

    @server.tool(
        name="run_sql",
        description="Execute a read-only SQL query against the orders or commesse tables. LIMIT is enforced server-side.",
    )
    def run_sql(sql: str, max_rows: int = SQL_DEFAULT_LIMIT) -> dict[str, Any]:
        result = engine.run_sql(sql, limit=max_rows)
        return result.to_dict()

    @server.tool(
        name="ask_orders",
        description="Answer a natural language analytics question by generating safe SQL and returning the result.",
    )
    def ask_orders(question: str, max_rows: int = SQL_DEFAULT_LIMIT) -> dict[str, Any]:
        result = engine.run_nl(question=question, limit=max_rows)
        return result.to_dict()

    def _generate_commesse_sql(question: str, limit: int = SQL_DEFAULT_LIMIT) -> str:
        prompt = (
            "You are an expert data analyst that writes safe, read-only PostgreSQL for a single table `commesse`.\n"
            "Only use SELECT/CTE queries. Never modify data. Always include a LIMIT.\n"
            "Use COALESCE where sums may involve NULLs. Prefer human readable aliases.\n"
            "Schema:\n"
            f"{COMMESSE_SCHEMA_TEXT}\n"
        )
        user = (
            f"Question: {question}\n"
            f"Return JSON with keys: sql (string), rationale (short string). "
            f"Limit results to {limit} rows unless the question is aggregate."
        )
        resp = llm_client.chat.completions.create(
            model=SQL_GENERATION_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user}],
        )
        parsed = resp.choices[0].message.content or "{}"
        try:
            import json as _json

            data = _json.loads(parsed)
            sql = str(data.get("sql", "")).strip()
        except Exception:
            sql = ""
        if not sql:
            raise ValueError("Empty SQL returned for commesse question.")
        return sql

    @server.tool(
        name="ask_commesse",
        description="Answer a natural language analytics question over the commesse table by generating safe SQL.",
    )
    def ask_commesse(question: str, max_rows: int = SQL_DEFAULT_LIMIT) -> dict[str, Any]:
        sql = _generate_commesse_sql(question, limit=max_rows)
        result = engine.run_sql(sql, limit=max_rows)
        out = result.to_dict()
        out["question"] = question
        out["rationale"] = out.get("rationale")
        return out

    @server.tool(
        name="sample_questions",
        description="Get a list of example analytics questions that work well on this dataset.",
    )
    def sample_questions() -> list[str]:
        return [
            "Top 5 vendors by total spend in 2024",
            "Monthly spend by category for the last 12 months",
            "Orders for vendor 'ACME' with amount > 10k",
            "Average order amount by region",
            "Orders in Lombardia in Q1 2024 grouped by vendor",
            "Find orders containing 'guanti nitrile' sorted by amount",
            "5 companies from which we placed orders similar to order 12345",
            "Orders similar to 'guanti in nitrile taglia L' in the last 6 months",
            "Commesse with oggetto about 'tunnel' and their responsabile_commessa",
            "Commesse by committente SNAM with importo > 1M",
            "For commessa code 1002 find similar commesse and list committente/responsabili",
            "For commessa 1002, list responsabile_commessa and orders whose text mentions the commessa name",
        ]

    def _text_vector(text: str) -> list[float]:
        norm = normalize_text(text)
        vec = embed_query_vectors([norm])[0]
        return vec.astype(float).tolist()

    def _commessa_query_vector(text: str) -> list[float]:
        norm = normalize_text(text)
        vec = embed_query_vectors([norm])[0]
        return vec.astype(float).tolist()

    def _commessa_vector_for_code(commessa_code: str) -> list[float]:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT oggetto_embedding FROM commesse WHERE commessa_code = %s OR CAST(record_id AS TEXT) = %s LIMIT 1",
                    (commessa_code, commessa_code),
                )
                row = cur.fetchone()
            if not row or row[0] is None:
                raise ValueError(f"No embedding found for commessa {commessa_code}")
            emb = row[0]
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            return [float(x) for x in emb]

    def _commessa_where(filters: dict[str, Any], params: dict[str, Any]) -> str:
        clauses = ["oggetto_embedding IS NOT NULL"]
        if filters.get("committente"):
            clauses.append("committente ILIKE %(committente)s")
            params["committente"] = f"%{filters['committente']}%"
        if filters.get("settore"):
            clauses.append("settore ILIKE %(settore)s")
            params["settore"] = f"%{filters['settore']}%"
        if filters.get("responsabile_commessa"):
            clauses.append("responsabile_commessa ILIKE %(responsabile_commessa)s")
            params["responsabile_commessa"] = f"%{filters['responsabile_commessa']}%"
        if filters.get("responsabile_cantiere"):
            clauses.append("responsabile_cantiere ILIKE %(responsabile_cantiere)s")
            params["responsabile_cantiere"] = f"%{filters['responsabile_cantiere']}%"
        if filters.get("date_from"):
            clauses.append("consegna >= %(date_from)s")
            params["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            clauses.append("ultimazione <= %(date_to)s")
            params["date_to"] = filters["date_to"]
        if filters.get("min_importo") is not None:
            clauses.append("importo >= %(min_importo)s")
            params["min_importo"] = filters["min_importo"]
        if filters.get("max_importo") is not None:
            clauses.append("importo <= %(max_importo)s")
            params["max_importo"] = filters["max_importo"]
        return " AND ".join(clauses)

    @server.tool(
        name="semantic_search",
        description=(
            "Find orders similar to a text description or an existing order code. "
            "Supports filters (region, vendor, category, contract_type, order_type, date_from, date_to, min_amount, max_amount)."
        ),
    )
    def semantic_search(
        query_text: str | None = None,
        order_code: str | None = None,
        top_k: int = 5,
        region: str | None = None,
        vendor: str | None = None,
        category: str | None = None,
        contract_type: str | None = None,
        order_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
    ) -> dict[str, Any]:
        if not query_text and not order_code:
            raise ValueError("Provide query_text or order_code.")
        filters = {
            "region": region,
            "vendor": vendor,
            "category": category,
            "contract_type": contract_type,
            "order_type": order_type,
            "date_from": date_from,
            "date_to": date_to,
            "min_amount": min_amount,
            "max_amount": max_amount,
        }
        with pool.connection() as conn:
            if order_code:
                query_vec = engine.vector_for_order_code(order_code, conn=conn)
            else:
                query_vec = _text_vector(query_text or "")
            rows = engine.semantic_search(query_vector=query_vec, top_k=top_k, filters=filters, conn=conn)
        return {"query_text": query_text, "order_code": order_code, "results": rows}

    @server.tool(
        name="semantic_search_commesse",
        description=(
            "Find commesse similar to a text description or a commessa_code. "
            "Filters: committente, settore, responsabile_commessa, responsabile_cantiere, date_from (consegna), "
            "date_to (ultimazione), min_importo, max_importo."
        ),
    )
    def semantic_search_commesse(
        query_text: str | None = None,
        commessa_code: str | None = None,
        top_k: int = 5,
        committente: str | None = None,
        settore: str | None = None,
        responsabile_commessa: str | None = None,
        responsabile_cantiere: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_importo: float | None = None,
        max_importo: float | None = None,
    ) -> dict[str, Any]:
        if not query_text and not commessa_code:
            raise ValueError("Provide query_text or commessa_code.")
        filters = {
            "committente": committente,
            "settore": settore,
            "responsabile_commessa": responsabile_commessa,
            "responsabile_cantiere": responsabile_cantiere,
            "date_from": date_from,
            "date_to": date_to,
            "min_importo": min_importo,
            "max_importo": max_importo,
        }
        params = {"limit": top_k}
        where_clause = _commessa_where(filters, params)
        if commessa_code:
            query_vec = _commessa_vector_for_code(commessa_code)
        else:
            query_vec = _commessa_query_vector(query_text or "")
        params["vec"] = query_vec

        sql = (
            "SELECT record_id, commessa_code, nome_commessa, committente, responsabile_commessa, "
            "responsabile_cantiere, settore, importo, consegna, ultimazione, oggetto_filled, "
            "(1 - (oggetto_embedding <=> %(vec)s::vector)) AS similarity "
            "FROM commesse WHERE "
            + where_clause
            + " ORDER BY oggetto_embedding <=> %(vec)s::vector LIMIT %(limit)s"
        )

        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "record_id": r[0],
                    "commessa_code": r[1],
                    "nome_commessa": r[2],
                    "committente": r[3],
                    "responsabile_commessa": r[4],
                    "responsabile_cantiere": r[5],
                    "settore": r[6],
                    "importo": float(r[7]) if r[7] is not None else None,
                    "consegna": r[8].isoformat() if r[8] else None,
                    "ultimazione": r[9].isoformat() if r[9] else None,
                    "oggetto": r[10],
                    "similarity": max(0.0, float(r[11])) if r[11] is not None else 0.0,
                }
            )
        return {"query_text": query_text, "commessa_code": commessa_code, "results": results}

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run MCP server exposing analytics tools over the orders DB")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to expose (stdio for CLI, SSE/streamable-http for network).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind when using SSE or streamable-http.")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind when using SSE or streamable-http.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    server = build_server(host=args.host, port=args.port)
    server.run(transport=args.transport)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()
