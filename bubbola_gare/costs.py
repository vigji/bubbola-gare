"""Utilities to keep track of token usage and costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

from openai.types.responses import Usage as ResponseUsage


ModelPrice = Dict[str, float]


# NOTE: prices are expressed in USD per million tokens and default to 0 so the user
# can easily overwrite them without editing the code.
MODEL_COSTS: Dict[str, ModelPrice] = {
    "5o-mini": {"input": 0.0, "output": 0.0},
}


@dataclass
class UsageSummary:
    """Aggregated usage information for a single model."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: ResponseUsage) -> None:
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: str, price_catalog: Mapping[str, ModelPrice]) -> float:
        pricing = price_catalog.get(model, {})
        in_price = pricing.get("input", 0.0)
        out_price = pricing.get("output", 0.0)
        return (self.input_tokens / 1_000_000) * in_price + (
            self.output_tokens / 1_000_000
        ) * out_price


@dataclass
class UsageTracker:
    """Tracks usage per model to make final reporting straightforward."""

    _stats: Dict[str, UsageSummary] = field(default_factory=dict)

    def add_usage(self, model: str, usage: ResponseUsage | None) -> None:
        if usage is None:
            return
        summary = self._stats.setdefault(model, UsageSummary())
        summary.add(usage)

    def summaries(self) -> Iterable[tuple[str, UsageSummary]]:
        return self._stats.items()

    def log_summary(self, logger, price_catalog: Mapping[str, ModelPrice]) -> None:
        if not self._stats:
            logger.info("No LLM calls were executed; skipping cost report.")
            return
        logger.info("LLM usage summary (tokens and estimated USD costs):")
        for model, summary in self._stats.items():
            logger.info(
                "  - %s: input=%s tokens, output=%s tokens, total=%s tokens, cost=$%.6f",
                model,
                summary.input_tokens,
                summary.output_tokens,
                summary.total_tokens,
                summary.cost_usd(model, price_catalog),
            )
