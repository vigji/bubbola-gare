# Bubbola Gare — Orders Analytics Stack

FastAPI + Postgres/pgvector + MCP to answer natural-language questions over the procurement table. The pipeline ingests the Excel dump, builds LLM summaries + embeddings, loads a vectorized `orders` table, and serves:
- API on `:8000` (search, analytics, chat gateway, OpenAI-compatible shim)
- MCP server on `:8100/mcp` (tools over the DB)
- Postgres on `:5432` (table `orders`)
- Optional Open WebUI on `:3000` (profile `ui`)

All source columns are preserved in the DB alongside derived fields, summaries, and embeddings.

## Ports & Services
- `8000` — FastAPI app (search + analytics + chat gateway, OpenAI shim at `/v1/chat/completions`)
- `8100` — MCP server (`http://localhost:8100/mcp`)
- `5432` — Postgres/pgvector (`orders` database)
- `3000` — Open WebUI (only when `--profile ui`)

## Quickstart (single command)
```bash
# Starts db + loader (one-shot) + app + MCP. UI is optional.
docker compose up --build

# To include the UI as well:
docker compose --profile ui up --build
```
Notes:
- First run loads `data/processed/orders_db_ready.parquet` into Postgres (≈51k rows) and can take a few minutes. The loader truncates and recreates the `orders` table each time.
- `.env` is read automatically; set your keys/models there (see “Config” below).
- App and MCP containers have `restart: unless-stopped`; the loader does not restart.

## API Endpoints (port 8000)
- Health: `GET /health` → `{"status":"ok"}`
- Vector search: `POST /search` with `query`, optional `top_k`, filters (`region`, `vendor`, `category`, `contract_type`, `order_type`, `min_amount`, `max_amount`, `date_from`, `date_to`)
- SQL (read-only): `POST /analytics/sql` with `{"sql": "..."}`
- NL-to-SQL: `POST /analytics/nlq` with `{"question": "..."}`
- Semantic (NL) analytics: `POST /analytics/semantic` similar filters as `/search`
- Schema: `GET /analytics/schema`
- MCP-backed chat: `POST /chat` with `{"question": "..."}` (runs MCP tool + LLM)
- OpenAI-compatible shim: `POST /v1/chat/completions` (base URL `http://localhost:8000`, any API key)

### Example queries (copy/paste)
- Vector search (BM25 + vector): `curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query":"guanti nitrile", "top_k":5, "filters":{"region":"Lombardia"}}'`
- Read-only SQL: `curl -X POST http://localhost:8000/analytics/sql -H "Content-Type: application/json" -d '{"sql":"SELECT vendor, SUM(amount) AS total FROM orders WHERE order_date >= ''2024-01-01'' GROUP BY vendor ORDER BY total DESC LIMIT 5"}'`
- NL → SQL analytics: `curl -X POST http://localhost:8000/analytics/nlq -H "Content-Type: application/json" -d '{"question":"Top 5 fornitori per importo nel 2024"}'`
- Semantic filters (structured): `curl -X POST http://localhost:8000/analytics/semantic -H "Content-Type: application/json" -d '{"query":"calcestruzzo per fondamenta", "top_k":5, "filters":{"region":"Lazio","min_amount":5000}}'`
- Chat (MCP-backed auto-tooling): `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question":"Fornitore dal quale abbiamo piazzato più ordini singoli?"}'`
- OpenAI shim: `curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dummy" -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"Dammi la spesa totale per il 2024"}]}'`

## MCP Server (port 8100)
- Endpoint: `http://localhost:8100/mcp`
- Tools: `list_schema`, `run_sql` (read-only with enforced LIMIT/allowlist), `ask_orders` (NL→SQL), `semantic_search`, `sample_questions`
- Connect any MCP client to that URL. In Open WebUI, add a “Custom OpenAI” provider pointing to `http://localhost:8000` (so chats route through the gateway that calls MCP).

## Data Pipeline (manual, if you need to regenerate)
```bash
uv run python -m bubbola_gare.pipeline ingest          # Excel -> data/processed/orders_raw.parquet
uv run python -m bubbola_gare.pipeline summarize       # LLM summaries -> data/processed/orders_summary.parquet
uv run python -m bubbola_gare.pipeline embed-summary   # Embed summaries -> data/processed/orders_summary_embeddings.parquet
uv run python -m bubbola_gare.db_prepare               # Join raw + summaries + embeddings -> orders_db_ready.parquet
uv run python -m bubbola_gare.db_loader --truncate     # Load Postgres (table recreated)
```
`docker compose up --build` already runs the loader against `orders_db_ready.parquet`.

## Table Schema Highlights
Derived/analysis fields:
- `record_id` (PK), `order_code` (from `ordine_n`), `order_date`, `delivery_date`
- `category`, `contract_type`, `order_type`, `requester`, `executor`, `vendor`
- `region`, `city`, `province`, `country`, `cap`, `amount` (numeric from `importo_ordine`)
- `text_raw`, `text_normalized`, `order_summary`, `summary_normalized`, `summary_embedding` (vector)

All source columns are also present (as text/typed where applicable):
`t_acquisti_cod_acquisto, ordine_n, data_ord, tipo_approv, tipologia, tipo_contratto, tipo_contratto_txt, tipo_ordine, richiedente, esecutore, scad_off, cond_pag, condiz_pagamento, banca_appoggio, oggetto, descrizione, spedizione, data_cons, luogo_cons, resa, cons_agg, note, allegati, clausola, data_spedizione, data_invio_contratto, data_ric_cont_firmato, cod_ind, fornitore, indirizzo, comune, provincia, cap, regione, nazione, prefisso_int, prefisso_naz, tel, tel2, email, sito_web, codice_fiscale, partita_iva, partita_iva_naz, importo_ordine, cod_fatt_dati, t_acquisti_dati_cod_acquisto, posiz_num, codice, desc, um, qta, importo_unit, sconto_aumento, importo_tot, data_consegna`.

## Config (.env keys)
- `OPENAI_API_KEY` (or set `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`)
- `OPENAI_MODEL`, `SQL_GENERATION_MODEL`, `CHAT_MODEL`, `EMBEDDING_MODEL`
- `MCP_HTTP_URL` (default `http://mcp:8100/mcp` in compose)
- Postgres: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`
- Embedding/search knobs: `EMBED_OUTPUT_DIM`, `DEFAULT_TOP_K`, `DEFAULT_ALPHA`

## Troubleshooting
- Loader OOM/slow? It now streams inserts in batches; ensure `orders_db_ready.parquet` exists and run `docker compose up --build` again.
- MCP not reachable? Confirm `bubbola-gare-mcp-1` is running and `MCP_HTTP_URL` points to `http://localhost:8100/mcp` when hitting from host.
- UI: only starts with `--profile ui`; otherwise the core stack (db + loader + app + MCP) is always included.
