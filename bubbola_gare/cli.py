"""Command-line interface for the PDF table extraction pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .costs import MODEL_COSTS, UsageTracker
from .llm import LLMTableExtractor
from .pdf_pipeline import PDFTableProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a paginated PDF table into a structured CSV file using an LLM."
    )
    parser.add_argument("pdf", type=Path, help="Input PDF file path")
    parser.add_argument("csv", type=Path, help="Destination CSV file path")
    parser.add_argument(
        "--model",
        default="5o-mini",
        help="OpenAI model name to use (default: 5o-mini)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    usage_tracker = UsageTracker()
    extractor = LLMTableExtractor(model=args.model, usage_tracker=usage_tracker)
    processor = PDFTableProcessor(extractor)

    rows = processor.process(args.pdf)
    processor.export_csv(rows, args.csv)
    logging.getLogger(__name__).info(
        "Successfully wrote %s rows to %s", len(rows), args.csv
    )

    usage_tracker.log_summary(logging.getLogger(__name__), MODEL_COSTS)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
