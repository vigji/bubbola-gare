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
- MCP-backed chat (dataset-scoped):
  - Orders: `POST /chat/orders` with `{"question": "..."}` → routes to orders tools only
  - Commesse: `POST /chat/commesse` with `{"question": "..."}` → routes to commesse tools only
- OpenAI-compatible shim (base URL `http://localhost:8000/v1`, any API key):
  - `GET /v1/models`
  - `POST /v1/chat/completions` (orders dataset default)

### Example queries (copy/paste)
- Vector search (BM25 + vector): `curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query":"guanti nitrile", "top_k":5, "filters":{"region":"Lombardia"}}'`
- Read-only SQL: `curl -X POST http://localhost:8000/analytics/sql -H "Content-Type: application/json" -d '{"sql":"SELECT vendor, SUM(amount) AS total FROM orders WHERE order_date >= ''2024-01-01'' GROUP BY vendor ORDER BY total DESC LIMIT 5"}'`
- NL → SQL analytics: `curl -X POST http://localhost:8000/analytics/nlq -H "Content-Type: application/json" -d '{"question":"Top 5 fornitori per importo nel 2024"}'`
- Semantic filters (structured): `curl -X POST http://localhost:8000/analytics/semantic -H "Content-Type: application/json" -d '{"query":"calcestruzzo per fondamenta", "top_k":5, "filters":{"region":"Lazio","min_amount":5000}}'`
- Chat ordini: `curl -X POST http://localhost:8000/chat/orders -H "Content-Type: application/json" -d '{"question":"Fornitore dal quale abbiamo piazzato più ordini singoli?"}'`
- Chat commesse: `curl -X POST http://localhost:8000/chat/commesse -H "Content-Type: application/json" -d '{"question":"Elenca commesse con committente SNAM"}'`
- OpenAI shim: `curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dummy" -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"Dammi la spesa totale per il 2024"}]}'`

## MCP Server (port 8100)
- Endpoint: `http://localhost:8100/mcp`
- Tools: `list_schema`, `run_sql` (read-only with enforced LIMIT/allowlist), `ask_orders` (NL→SQL), `semantic_search`, `sample_questions`
- Connect any MCP client to that URL.

## Open WebUI / LibreChat connectivity (chat UI → gateway → MCP)
- Treat the FastAPI gateway as an OpenAI-compatible backend with base URL `http://<gateway-host>:8000/v1` (note the trailing `/v1`).
- The shim exposes `GET /v1/models` (lists your configured `CHAT_MODEL`) and `POST /v1/chat/completions`.
- If you run the provided profile: `docker compose --profile ui up --build`, Open WebUI reaches the app at `http://app:8000/v1` on the compose network (already set as `OPENAI_API_BASE` in `docker-compose.yml`). In the UI, add a “Custom OpenAI” connection pointing to that URL with any API key (e.g., `dummy-key`), then hit “Test”. You should see the model list returned.
- If you run Open WebUI in Docker while the gateway runs on the host, add `--add-host=host.docker.internal:host-gateway` to your `docker run` and set the connection URL to `http://host.docker.internal:8000/v1` with a dummy bearer key.
- Quick connectivity check from the UI container: `docker exec -it open-webui sh -c "curl -s http://host.docker.internal:8000/v1/models"`. You should see the model list JSON; if not, it is a network/URL issue.
- LibreChat follows the same pattern: add a “Custom OpenAI-compatible” provider with the same base URL + bearer.
- If you see “Network Problem” in the UI:
  - Confirm you are using the right hostname for your topology: `http://app:8000/v1` when both UI and gateway are in the same docker-compose; `http://localhost:8000/v1` if the UI is on your host hitting the published port; `http://host.docker.internal:8000/v1` if the UI is in Docker and the gateway is on the host.
  - From inside the UI container run `curl http://app:8000/v1/models` (compose) or `curl http://host.docker.internal:8000/v1/models` (UI in Docker, gateway on host). If it returns JSON, connectivity is fine and the issue is likely the URL configured in Admin → Connections.
  - If you enabled “Direct connection” in the connection modal (green toggle), the browser must resolve the host. In that case prefer `http://localhost:8000/v1` (published port) or `http://host.docker.internal:8000/v1`; using the internal service name `app` will fail from the browser even though it works inside the container.
- Running Open WebUI with plain Docker against a host-running gateway:
  1) Start the UI:\
  `docker run -d -p 3000:8080 --name open-webui --add-host=host.docker.internal:host-gateway ghcr.io/open-webui/open-webui:main`
  2) Open Admin → Connections → Add “Custom OpenAI” and set:\
     URL: `http://host.docker.internal:8000/v1`\
     API Key: `dummy-key` (or any string, we don’t enforce it)
  3) Click Test; the model list should include your `CHAT_MODEL` (e.g., `gpt-4.1-mini`). If the test fails, run the curl check above from inside the container.

## Data Pipeline (manual, if you need to regenerate)
```bash
uv run python -m bubbola_gare.pipeline ingest          # Excel -> data/processed/orders_raw.parquet
uv run python -m bubbola_gare.pipeline summarize       # LLM summaries -> data/processed/orders_summary.parquet
uv run python -m bubbola_gare.pipeline embed-summary   # Embed summaries -> data/processed/orders_summary_embeddings.parquet
uv run python -m bubbola_gare.db_prepare               # Join raw + summaries + embeddings -> orders_db_ready.parquet
uv run python -m bubbola_gare.db_loader --truncate     # Load Postgres (table recreated)
```
`docker compose up --build` already runs the loader against `orders_db_ready.parquet`.

### Commesse (projects) pipeline
- Source file: `data/gesa_dump_commesse.xlsx` (columns like `num commessa`, `nome commessa`, `Settore`, `committente`, `oggetto`, `importo`, dates, etc.).
- Preprocessing fills missing `oggetto` with `nome_commessa + settore + committente`, normalizes text, and keeps all source columns.
- Commands:
  ```bash
  uv run python -m bubbola_gare.pipeline commesse-ingest         # Excel -> data/processed/commesse_raw.parquet
  uv run python -m bubbola_gare.pipeline commesse-preprocess     # Fill oggetto/text -> data/processed/commesse_preprocessed.parquet
  uv run python -m bubbola_gare.pipeline commesse-embed          # Embed oggetto_filled -> data/processed/commesse_embeddings.parquet
  uv run python -m bubbola_gare.pipeline commesse-db-ready       # Rename embedding -> data/processed/commesse_db_ready.parquet
  uv run python -m bubbola_gare.db_loader --commesse-path data/processed/commesse_db_ready.parquet --truncate-commesse
  ```
- The loader creates a `commesse` table with the original columns, the filled `oggetto_filled`, normalized text, and an `oggetto_embedding` vector for semantic search/filtering.

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
- `MCP_HTTP_URL` (default `http://mcp:8100/mcp` in compose), `MCP_HTTP_URL_ORDERS`, `MCP_HTTP_URL_COMMESSE`
- Postgres: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`
- Embedding/search knobs: `EMBED_OUTPUT_DIM`, `DEFAULT_TOP_K`, `DEFAULT_ALPHA`

## Troubleshooting
- Loader OOM/slow? It now streams inserts in batches; ensure `orders_db_ready.parquet` exists and run `docker compose up --build` again.
- MCP not reachable? Confirm `bubbola-gare-mcp-1` is running and `MCP_HTTP_URL`/`MCP_HTTP_URL_ORDERS`/`MCP_HTTP_URL_COMMESSE` point to `http://localhost:8100/mcp` when hitting from host.
- UI: only starts with `--profile ui`; otherwise the core stack (db + loader + app + MCP) is always included.
