# Bubbola Gare PDF Table Extractor

This project converts paginated PDF tables (without OCR) into structured CSV files
using an LLM with a JSON schema. The extractor keeps track of descriptions that are
split between pages through a continuation field.

## Features

- Page-by-page processing with `pdfplumber` to collect the raw text.
- LLM-powered parsing via the OpenAI Responses API with a strict JSON schema.
- Continuation handling to merge descriptions that span multiple pages.
- Configurable model name (default: `5o-mini`).
- Token usage aggregation with hooks to estimate costs once prices are known.

## Installation

```bash
pip install -r requirements.txt
```

The script expects the `OPENAI_API_KEY` environment variable to be set so it can
authenticate with the OpenAI API.

## Usage

```bash
python -m bubbola_gare <input.pdf> <output.csv> [--model MODEL] [--log-level LEVEL]
```

Example:

```bash
python -m bubbola_gare data/bando.pdf output/items.csv --model 5o-mini
```

After the conversion completes, the program logs the aggregated token usage and
estimated USD cost. Update `MODEL_COSTS` in `bubbola_gare/costs.py` with the prices
for the models you intend to use.

## Output Format

The resulting CSV always contains the columns `code`, `description`, `quantity`,
`unit_price`, and `total_price`. Empty values are emitted as blank cells so that
post-processing scripts can handle them consistently.

## Development Notes

- The LLM response schema is defined via Pydantic models in `bubbola_gare/models.py`.
- `UsageTracker` can be extended if you need to persist or visualize the token
  consumption metrics.
- The PDF processor logs warnings whenever a page does not contain extractable text
  or when the last page leaves a pending continuation.
