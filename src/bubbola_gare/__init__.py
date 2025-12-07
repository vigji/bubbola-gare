"""Utilities for ingesting, preprocessing, embedding, and searching order data."""

from .config import (
    RAW_EXCEL_PATH,
    RAW_PARQUET_PATH,
    PREPROCESSED_PATH,
    EMBED_CACHE_PATH,
    EMBEDDING_MODEL,
    SUMMARY_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_ALPHA,
    TEXT_COLUMNS,
    ID_COLUMN,
    VENDOR_COLUMN,
    SUMMARY_PATH,
    SUMMARY_EMBEDDINGS_PATH,
)

__all__ = [
    "RAW_EXCEL_PATH",
    "RAW_PARQUET_PATH",
    "PREPROCESSED_PATH",
    "EMBED_CACHE_PATH",
    "EMBEDDING_MODEL",
    "SUMMARY_MODEL",
    "DEFAULT_TOP_K",
    "DEFAULT_ALPHA",
    "TEXT_COLUMNS",
    "ID_COLUMN",
    "VENDOR_COLUMN",
    "SUMMARY_PATH",
    "SUMMARY_EMBEDDINGS_PATH",
]
