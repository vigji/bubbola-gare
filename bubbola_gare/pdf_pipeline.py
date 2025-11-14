"""Core pipeline that orchestrates PDF parsing and CSV export."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pdfplumber

from .llm import LLMTableExtractor
from .models import TableItem


LOGGER = logging.getLogger(__name__)


class PDFTableProcessor:
    """High level orchestrator that runs the extraction page by page."""

    def __init__(self, extractor: LLMTableExtractor) -> None:
        self.extractor = extractor

    def process(self, pdf_path: Path) -> List[TableItem]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        rows: List[TableItem] = []
        pending_description: Optional[str] = None

        for page_number, page_text in self._iterate_pages(pdf_path):
            LOGGER.info("Processing page %s", page_number)
            extraction = self.extractor.extract_page(page_text, pending_description)
            rows.extend(extraction.items)
            pending_description = extraction.continuation.pending_description

        if pending_description:
            LOGGER.warning(
                "The last page ended with a pending description: %s",
                pending_description,
            )

        return rows

    def export_csv(self, rows: Sequence[TableItem], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["code", "description", "quantity", "unit_price", "total_price"]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.code or "",
                        row.description or "",
                        row.quantity or "",
                        row.unit_price or "",
                        row.total_price or "",
                    ]
                )

    def _iterate_pages(self, pdf_path: Path) -> Iterable[tuple[int, str]]:
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=1, y_tolerance=1)
                if not text:
                    LOGGER.warning("Page %s did not contain extractable text.", index)
                    text = ""
                yield index, text
