# Bubbola Gare - Search Pipeline

Refactored pipeline to ingest the Excel orders table, produce LLM summaries, embed those summaries, and serve similarity search via Postgres + pgvector. Intermediate artifacts are saved under `data/processed` to keep the full table, summaries, and summary embeddings handy for tinkering or visualisation.

## Setup (local)
- Prereq: `uv` installed (or python 3.12 + pip). From repo root: `uv venv && . .venv/bin/activate && uv sync`.
- Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, adjust columns/paths if needed.
- Run commands from repo root; `uv run` will pick up the project (or set `PYTHONPATH=src`).

## Quickstart (docker-compose: Postgres + pgvector)
- Ensure `data/processed/orders_db_ready.parquet` exists (run steps in "Data pipeline" + `uv run python -m bubbola_gare.db_prepare`).
- Start everything (API + MCP): `docker compose up --build` (compose reads `.env` automatically).
  - Services: `db` (Postgres + pgvector), `loader` (one-shot import of reduced vectors), `app` (FastAPI at :8000), `mcp` (streamable HTTP MCP at :8100). Embeddings are reduced to <=1999 dims so an IVFFlat index is created automatically.
- Health check: `curl http://localhost:8000/health` → `{ "status": "ok" }`.
- MCP endpoint: streamable HTTP at `http://localhost:8100/mcp` (configured via `LLM_PROVIDER`, `OPENAI_MODEL` or `OLLAMA_MODEL` env). Point your MCP client to that URL; docker compose already starts the DB and loader.
- Search example (POST):
  ```bash
  curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{
          "query": "guanti in nitrile taglia L",
          "top_k": 5,
          "filters": {"region": "Lombardia", "date_from": "2024-01-01"}
        }'
  ```
  Filters are optional: `region`, `vendor`, `category`, `contract_type`, `order_type`, `min_amount`, `max_amount`, `date_from`, `date_to`.

## Data pipeline (summary-only)
- `uv run python -m bubbola_gare.pipeline ingest`  
  Reads `data/QQ-Ordini_Luigi.xlsx` (override via `RAW_EXCEL_PATH`) and stores the **full** table at `data/processed/orders_raw.parquet`.
- `uv run python -m bubbola_gare.pipeline summarize`  
  Calls the `SUMMARY_MODEL` on each raw row to produce a concise technical summary (`order_summary`) and saves to `data/processed/orders_summary.parquet` (cached by row signature to avoid recompute).
- `uv run python -m bubbola_gare.pipeline embed-summary`  
  Embeds the normalized summaries and writes `data/processed/orders_summary_embeddings.parquet` (reduced to `EMBED_OUTPUT_DIM`, default 1999).

End-to-end locally (with API):
1) Ingest → summarize → embed-summary: run the three commands above in order.
2) Build DB-ready parquet: `uv run python -m bubbola_gare.db_prepare`
3) Load Postgres (requires a running pgvector Postgres at `PGHOST/PGUSER/PGPASSWORD/PGDATABASE`):
   `uv run python -m bubbola_gare.db_loader --truncate`
4) Start API locally (expects DB + OPENAI_API_KEY):
   `uv run uvicorn bubbola_gare.service.api:app --host 0.0.0.0 --port 8000`

## DB-ready dataset
- `uv run python -m bubbola_gare.db_prepare`  
  Cleans the raw table, standardizes text/location fields, joins summaries + summary embeddings, and writes `data/processed/orders_db_ready.parquet` (keeps row ids and references). Embeddings are reduced to `EMBED_OUTPUT_DIM` (default 1999) before loading so they can be indexed with pgvector IVFFlat.

## Serve via Postgres + API
- Docker (recommended): see "Quickstart" section for the one-liner and sample cURL.
- Local manual run (after DB is loaded):
  `uv run uvicorn bubbola_gare.service.api:app --host 0.0.0.0 --port 8000`
  - Requires env: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, `OPENAI_API_KEY`.
  - Test: `curl http://localhost:8000/health`

## Analytics & NL-to-SQL
- Run ad-hoc SQL safely:  
  `curl -X POST http://localhost:8000/analytics/sql -H "Content-Type: application/json" -d '{"sql": "select vendor, sum(amount) as total from orders group by vendor order by total desc limit 5"}'`
- Natural language analytics (SQL is generated and enforced read-only/limited):  
  `curl -X POST http://localhost:8000/analytics/nlq -H "Content-Type: application/json" -d '{"question": "Top 5 vendors by total spend in 2024"}'`
- Semantic similarity analytics (by text or order code, with filters):  
  `curl -X POST http://localhost:8000/analytics/semantic -H "Content-Type: application/json" -d '{"order_code": "12345", "top_k": 5, "filters": {"region": "Lombardia"}}'`
- Schema endpoint: `curl http://localhost:8000/analytics/schema`
- Limits are clamped to `SQL_MAX_LIMIT` (env, default 500) and only the `orders` table is allowed.

## Searching (summary embeddings only)
- Run a quick local search over summary embeddings:  
  `uv run python -m bubbola_gare.pipeline search "query terms" --top-k 10 --alpha 0.4`  
  Uses `data/processed/orders_summary_embeddings.parquet` by default.
- Programmatic use:  
  ```python
  from bubbola_gare.search import load_index_from_parquet
  index = load_index_from_parquet("data/processed/orders_summary_embeddings.parquet")
  hits = index.search("guanti nitrile", top_k=5, alpha=0.4)
  ```

## Analysis plot
- Build a 2D PCA projection with hoverable text and vendor colours:  
  `uv run python analysis/plot_embeddings.py --limit 5000 --output data/processed/embeddings_plot.html`  
  Open the HTML to explore; `--limit` keeps the plot responsive on large datasets.

## Notes
- Adjust `TEXT_COLUMNS`, `ID_COLUMN`, `VENDOR_COLUMN`, and embedding/search knobs in `.env`.
- Intermediate parquet files preserve row references so you can join back to the original table even after filtering.
- MCP server (analytics tools over the DB):
  - Run locally over stdio: `uv run python -m bubbola_gare.service.mcp_server --transport stdio`
  - Or expose over HTTP (used in docker-compose): `uv run python -m bubbola_gare.service.mcp_server --transport streamable-http --host 0.0.0.0 --port 8100`
  - Tools: `list_schema`, `run_sql` (read-only with enforced LIMIT + allowlist), `ask_orders` (NL->SQL), `semantic_search` (by text or order_code with filters), and `sample_questions`.
- LLM provider config (set in `.env`):  
  - `LLM_PROVIDER=openai|ollama` (default openai)  
  - For Ollama: `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`, `OLLAMA_MODEL=qwen3-coder:30b` (used for SQL generation / chat), set `EMBEDDING_MODEL` to a model that serves embeddings.  
  - For OpenAI: `OPENAI_API_KEY=...`, `OPENAI_MODEL=gpt-4.1-mini` (used for SQL generation), `EMBEDDING_MODEL=text-embedding-3-large`.

## MCP query gallery (sample requests)
- Top spenders: `ask_orders` with question `"Top 10 vendors by total spend in 2024"`
- Regional filter: `ask_orders` with question `"Monthly spend in Lombardia grouped by vendor for 2024"`
- Order lookup: `run_sql` with SQL `select order_code, vendor, amount from orders where order_code ilike 'ABC%' limit 20`
- Similar by text: `semantic_search` with `query_text="guanti in nitrile taglia L", top_k=5`
- Similar by order code: `semantic_search` with `order_code="12345", filters={region: "Lombardia"}`
- Thresholded amounts: `ask_orders` with question `"Orders over 10000 EUR for vendor ACME in 2023"`
- Category trend: `ask_orders` with question `"Total amount per category per month for the last 6 months"`
