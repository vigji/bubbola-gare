from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Source column mapping (kept verbatim in the DB)
RAW_SOURCE_COLUMNS = [
    "t_acquisti_cod_acquisto",
    "ordine_n",
    "data_ord",
    "tipo_approv",
    "tipologia",
    "tipo_contratto",
    "tipo_contratto_txt",
    "tipo_ordine",
    "richiedente",
    "esecutore",
    "scad_off",
    "cond_pag",
    "condiz_pagamento",
    "banca_appoggio",
    "oggetto",
    "descrizione",
    "spedizione",
    "data_cons",
    "luogo_cons",
    "resa",
    "cons_agg",
    "note",
    "allegati",
    "clausola",
    "data_spedizione",
    "data_invio_contratto",
    "data_ric_cont_firmato",
    "cod_ind",
    "fornitore",
    "indirizzo",
    "comune",
    "provincia",
    "cap",
    "regione",
    "nazione",
    "prefisso_int",
    "prefisso_naz",
    "tel",
    "tel2",
    "email",
    "sito_web",
    "codice_fiscale",
    "partita_iva",
    "partita_iva_naz",
    "importo_ordine",
    "cod_fatt_dati",
    "t_acquisti_dati_cod_acquisto",
    "posiz_num",
    "codice",
    "desc",
    "um",
    "qta",
    "importo_unit",
    "sconto_aumento",
    "importo_tot",
    "data_consegna",
]
RAW_DATE_COLUMNS = [
    "data_ord",
    "data_cons",
    "data_spedizione",
    "data_invio_contratto",
    "data_ric_cont_firmato",
    "data_consegna",
]
RAW_NUMERIC_COLUMNS = [
    "importo_ordine",
    "qta",
    "importo_unit",
    "importo_tot",
]

# LLM provider selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", OPENAI_MODEL)
MCP_HTTP_URL = os.getenv("MCP_HTTP_URL", "http://localhost:8100/mcp")

# Input/output locations
RAW_EXCEL_PATH = Path(
    os.getenv("RAW_EXCEL_PATH", DATA_DIR / "gesa_dump.xlsx")
)
RAW_PARQUET_PATH = Path(
    os.getenv("RAW_PARQUET_PATH", PROCESSED_DIR / "orders_raw.parquet")
)
PREPROCESSED_PATH = Path(
    os.getenv("PREPROCESSED_PATH", PROCESSED_DIR / "orders_preprocessed.parquet")
)
SUMMARY_PATH = Path(
    os.getenv("SUMMARY_PATH", PROCESSED_DIR / "orders_summary.parquet")
)
SUMMARY_EMBEDDINGS_PATH = Path(
    os.getenv("SUMMARY_EMBEDDINGS_PATH", PROCESSED_DIR / "orders_summary_embeddings.parquet")
)
DB_READY_PATH = Path(
    os.getenv("DB_READY_PATH", PROCESSED_DIR / "orders_db_ready.parquet")
)
EMBED_CACHE_PATH = Path(
    os.getenv("EMBED_CACHE_PATH", CACHE_DIR / "embedding_cache.pkl")
)

# Text configuration
TEXT_COLUMNS = [
    c.strip()
    for c in os.getenv(
        "TEXT_COLUMNS", "oggetto,descrizione,desc,note"
    ).split(",")
    if c.strip()
]
ID_COLUMN = os.getenv("ID_COLUMN", "ordine_n")
VENDOR_COLUMN = os.getenv("VENDOR_COLUMN", "fornitore")

# Embedding + search tuning
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5-nano-2025-08-07")
SUMMARY_CONCURRENCY = int(os.getenv("SUMMARY_CONCURRENCY", "90"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "8192"))
EMBED_WARN_MARGIN = float(os.getenv("EMBED_WARN_MARGIN", "0.9"))
AVG_CHARS_PER_TOKEN = int(os.getenv("AVG_CHARS_PER_TOKEN", "4"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.4"))
EMBED_OUTPUT_DIM = int(os.getenv("EMBED_OUTPUT_DIM", "1999"))

# Analytics / SQL generation
SQL_GENERATION_MODEL = os.getenv(
    "SQL_GENERATION_MODEL",
    OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL,
)
SQL_DEFAULT_LIMIT = int(os.getenv("SQL_DEFAULT_LIMIT", "200"))
SQL_MAX_LIMIT = int(os.getenv("SQL_MAX_LIMIT", "500"))

# Search parameters
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# Database serving
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "orders")
PGUSER = os.getenv("PGUSER", "orders")
PGPASSWORD = os.getenv("PGPASSWORD", "orders")
PG_MIN_CONN = int(os.getenv("PG_MIN_CONN", "1"))
PG_MAX_CONN = int(os.getenv("PG_MAX_CONN", "4"))


def ensure_data_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
