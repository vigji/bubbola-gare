# Bubbola Gare - Search Pipeline

Refactored pipeline to ingest the Excel orders table, produce LLM summaries, embed those summaries, and serve similarity search via Postgres + pgvector. Intermediate artifacts are saved under `data/processed` to keep the full table, summaries, and summary embeddings handy for tinkering or visualisation.

## Setup (local)
- Prereq: `uv` installed (or python 3.12 + pip). From repo root: `uv venv && . .venv/bin/activate && uv sync`.
- Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, adjust columns/paths if needed.
- Run commands from repo root; `uv run` will pick up the project (or set `PYTHONPATH=src`).

## Quickstart (docker-compose: Postgres + pgvector)
- Ensure `data/processed/orders_db_ready.parquet` exists (run steps in "Data pipeline" + `uv run python -m bubbola_gare.db_prepare`).
- Start everything: `OPENAI_API_KEY=... docker compose up --build`.
  - Services: `db` (Postgres + pgvector), `loader` (one-shot import of reduced vectors), `app` (FastAPI at :8000). Embeddings are reduced to <=1999 dims so an IVFFlat index is created automatically.
- Health check: `curl http://localhost:8000/health` → `{ "status": "ok" }`.
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
