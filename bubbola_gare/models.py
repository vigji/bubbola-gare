"""Pydantic models that define the structure of the LLM responses."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TableItem(BaseModel):
    """Single row extracted from the tender table."""

    code: Optional[str] = Field(
        default=None,
        description=(
            "Product or service identifier. Keep it exactly as in the PDF, "
            "even if alphanumeric."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed Italian description of the item.",
    )
    quantity: Optional[str] = Field(
        default=None,
        description="Ordered quantity as a string to preserve decimal separators.",
    )
    unit_price: Optional[str] = Field(
        default=None,
        description="Unit price including currency, copied verbatim from the PDF.",
    )
    total_price: Optional[str] = Field(
        default=None,
        description="Total price including currency, copied verbatim from the PDF.",
    )


class ContinuationState(BaseModel):
    """State that allows the pipeline to stitch together split descriptions."""

    pending_description: Optional[str] = Field(
        default=None,
        description=(
            "Trailing text that belongs to the last item's description on the "
            "current page and needs to be prepended to the first item on the next "
            "page."
        ),
    )


class PageExtraction(BaseModel):
    """Structured payload returned by the LLM for each page."""

    items: List[TableItem] = Field(
        default_factory=list,
        description="Ordered list of rows that appear on the processed page.",
    )
    continuation: ContinuationState = Field(
        default_factory=ContinuationState,
        description="Continuation state for the next page.",
    )
