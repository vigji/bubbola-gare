from __future__ import annotations

import argparse
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..config import SQL_DEFAULT_LIMIT
from ..embedding import embed_texts, l2_normalize
from ..preprocessing import normalize_text
from .analytics import AnalyticsEngine, ORDERS_COLUMNS, ORDERS_SCHEMA_TEXT
from .db import get_pool


def build_server(host: str = "0.0.0.0", port: int = 8100) -> FastMCP:
    """
    Configure the MCP server with analytics tools.
    """
    pool = get_pool()
    pool.open()
    engine = AnalyticsEngine(pool=pool)

    instructions = (
        "Interact with the `orders` table. Use tools to inspect schema, run SQL, ask NL questions, or run "
        "semantic similarity searches. All queries are read-only and limited. Helpful intents: vendor totals, "
        "top suppliers, spend over time, order lookups by code/region/category, and finding orders similar to "
        "a given order or text."
    )
    server = FastMCP(name="orders-mcp", instructions=instructions, host=host, port=port, log_level="INFO")

    @server.tool(name="list_schema", description="Show available columns and descriptions for the orders table.")
    def list_schema() -> dict[str, Any]:
        return {"schema": ORDERS_SCHEMA_TEXT, "columns": ORDERS_COLUMNS}

    @server.tool(
        name="run_sql",
        description="Execute a read-only SQL query against the orders table. LIMIT is enforced server-side.",
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
        ]

    def _text_vector(text: str) -> list[float]:
        norm = normalize_text(text)
        vec = embed_texts([norm])
        return l2_normalize(vec)[0].astype(float).tolist()

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
