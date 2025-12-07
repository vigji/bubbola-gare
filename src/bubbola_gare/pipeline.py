from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import (
    PREPROCESSED_PATH,
    RAW_EXCEL_PATH,
    RAW_PARQUET_PATH,
    SUMMARY_EMBEDDINGS_PATH,
    SUMMARY_PATH,
    ensure_data_dirs,
)
from .data_io import ingest_excel_to_parquet, load_parquet, save_parquet
from .embedding import compute_embeddings
from .preprocessing import preprocess_orders
from .summarization import summarize_orders, SUMMARY_COLUMN
from .search import HybridIndex, load_index_from_parquet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingest(source: Path = RAW_EXCEL_PATH, dest: Path = RAW_PARQUET_PATH) -> pd.DataFrame:
    logger.info("Ingesting Excel %s -> %s", source, dest)
    ensure_data_dirs()
    df = ingest_excel_to_parquet(source, dest)
    logger.info("Saved raw table with %d rows and %d columns", len(df), len(df.columns))
    return df


def run_preprocess(source: Path = RAW_PARQUET_PATH, dest: Path = PREPROCESSED_PATH) -> pd.DataFrame:
    logger.info("Preprocessing %s -> %s", source, dest)
    ensure_data_dirs()
    df_raw = load_parquet(source)
    processed = preprocess_orders(df_raw)
    save_parquet(processed, dest)
    logger.info(
        "Saved preprocessed corpus with %d rows (from %d)",
        len(processed),
        len(df_raw),
    )
    return processed


def run_summarize(source: Path = RAW_PARQUET_PATH, dest: Path = SUMMARY_PATH) -> pd.DataFrame:
    logger.info("Summarizing orders %s -> %s (LLM)", source, dest)
    ensure_data_dirs()
    df_raw = load_parquet(source)
    return summarize_orders(df_raw, dest_path=dest)


def run_embeddings_summary(source: Path = SUMMARY_PATH, dest: Path = SUMMARY_EMBEDDINGS_PATH) -> pd.DataFrame:
    logger.info("Computing embeddings from summaries %s -> %s", source, dest)
    ensure_data_dirs()
    df_sum = load_parquet(source)
    if SUMMARY_COLUMN not in df_sum.columns:
        raise ValueError(f"{SUMMARY_COLUMN} column not found in {source}")
    df_sum = df_sum.rename(columns={"summary_normalized": "text_normalized", SUMMARY_COLUMN: "text_raw"})
    before = len(df_sum)
    df_sum = df_sum[df_sum["text_normalized"].fillna("").str.len() > 0].copy()
    dropped = before - len(df_sum)
    if dropped:
        logger.info("Skipped %d empty summary rows", dropped)
    if len(df_sum) == 0:
        raise ValueError("No non-empty summaries found. Run `uv run python -m bubbola_gare.pipeline summarize` first.")
    df_emb = compute_embeddings(df_sum, text_col="text_normalized")
    save_parquet(df_emb, dest)
    logger.info("Saved summary-based embeddings for %d rows", len(df_emb))
    return df_emb


def run_search(
    query: str,
    embeddings_path: Path = SUMMARY_EMBEDDINGS_PATH,
    top_k: int = 10,
    alpha: float = 0.4,
) -> None:
    index = load_index_from_parquet(embeddings_path)
    results = index.search(query, top_k=top_k, alpha=alpha)
    for i, r in enumerate(results, start=1):
        logger.info(
            "#%02d score=%.3f vec=%.3f bm25=%.3f vendor=%s order_id=%s",
            i,
            r.get("score_combined"),
            r.get("score_vector"),
            r.get("score_bm25"),
            r.get("vendor", ""),
            r.get("order_id", r.get("record_id")),
        )
        logger.info("    %s", r.get("text"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data pipeline for order search.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load Excel and export raw parquet")
    p_ingest.add_argument("--source", type=Path, default=RAW_EXCEL_PATH)
    p_ingest.add_argument("--dest", type=Path, default=RAW_PARQUET_PATH)

    p_sum = sub.add_parser("summarize", help="Generate LLM summaries per row and store parquet")
    p_sum.add_argument("--source", type=Path, default=RAW_PARQUET_PATH)
    p_sum.add_argument("--dest", type=Path, default=SUMMARY_PATH)

    p_emb_sum = sub.add_parser("embed-summary", help="Embed LLM summaries and export parquet")
    p_emb_sum.add_argument("--source", type=Path, default=SUMMARY_PATH)
    p_emb_sum.add_argument("--dest", type=Path, default=SUMMARY_EMBEDDINGS_PATH)

    p_search = sub.add_parser("search", help="Run a quick search over stored embeddings")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--alpha", type=float, default=0.4)
    p_search.add_argument("--embeddings", type=Path, default=SUMMARY_EMBEDDINGS_PATH)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        run_ingest(args.source, args.dest)
    elif args.command == "summarize":
        run_summarize(args.source, args.dest)
    elif args.command == "embed-summary":
        run_embeddings_summary(args.source, args.dest)
    elif args.command == "search":
        run_search(args.query, embeddings_path=args.embeddings, top_k=args.top_k, alpha=args.alpha)


if __name__ == "__main__":
    main()
