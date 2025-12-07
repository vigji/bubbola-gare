from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Optional

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from .config import (
    ID_COLUMN,
    SUMMARY_CONCURRENCY,
    SUMMARY_MODEL,
    SUMMARY_PATH,
    VENDOR_COLUMN,
)
from .preprocessing import normalize_text

logger = logging.getLogger(__name__)

SUMMARY_COLUMN = "order_summary"


def _row_signature(row: pd.Series) -> str:
    # Stable signature of the row contents to avoid recomputing identical summaries.
    return json.dumps(row.to_dict(), sort_keys=True, default=str)


def _format_prompt(row: pd.Series) -> str:
    return (
        "You are given a single purchase order row. "
        "Produce a concise, single-sentence summary of what was ordered, "
        "Avoiding refs to the specific order (nunmber, confirmation, vendor) "
        "keeping technical specs (codes, units, quantities, models, materials) "
        "and omitting boilerplate or contractual clutter. "
        "Do not include geographical info unless very specifically linked to the item. "
        "Do not include price information, neither total price nor per item cost, canone, etc."
        "Just summarize what whas ordered!"
        "If description is very long (> 2 sentences/rows) summarize it in 2 concise sentencies."
        "Inn general be concise"
        "Respond in Italian. Do not add explanations.\n\n"
        f"Riga ordine (JSON):\n{row.to_json(force_ascii=False)}"
    )


def _maybe_load_existing(path: Optional[str | bytes]) -> Dict[int, Dict[str, str]]:
    if not path:
        return {}
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return {}
    existing = {}
    if {"record_id", "summary_key", SUMMARY_COLUMN}.issubset(df.columns):
        for _, r in df.iterrows():
            existing[int(r["record_id"])] = {
                "summary_key": r["summary_key"],
                SUMMARY_COLUMN: r[SUMMARY_COLUMN],
            }
    return existing


def summarize_orders(df_raw: pd.DataFrame, dest_path=SUMMARY_PATH) -> pd.DataFrame:
    """
    Generate LLM summaries for each order row (caching by signature) and persist to Parquet.
    """
    df = df_raw.reset_index(drop=False).rename(columns={"index": "source_row"})
    df["record_id"] = df.index.astype("int64")
    df["order_id"] = df[ID_COLUMN] if ID_COLUMN in df.columns else df["record_id"]
    df["vendor"] = df[VENDOR_COLUMN] if VENDOR_COLUMN in df.columns else ""

    existing = _maybe_load_existing(dest_path)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(SUMMARY_CONCURRENCY)

    rows = []
    for _, row in df.iterrows():
        rid = int(row["record_id"])
        sig = _row_signature(row)
        rows.append(
            {
                "record_id": rid,
                "source_row": int(row["source_row"]),
                "order_id": row["order_id"],
                "vendor": row["vendor"],
                "sig": sig,
                "row": row,
            }
        )

    async def _summarize_row(entry: Dict) -> Dict:
        rid = entry["record_id"]
        sig = entry["sig"]
        cached = existing.get(rid)
        if cached and cached.get("summary_key") == sig and cached.get(SUMMARY_COLUMN):
            summary_text = cached[SUMMARY_COLUMN]
        else:
            prompt = _format_prompt(entry["row"])
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=SUMMARY_MODEL,
                    messages=[
                        {"role": "system", "content": "Assistant summarises order lines."},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=5000,
                    reasoning_effort="minimal",
                )
            summary_text = resp.choices[0].message.content.strip()
            print(summary_text)

        return {
            "record_id": rid,
            "source_row": entry["source_row"],
            "order_id": entry["order_id"],
            "vendor": entry["vendor"],
            SUMMARY_COLUMN: summary_text,
            "summary_key": sig,
            "summary_normalized": normalize_text(summary_text),
        }

    async def _run() -> pd.DataFrame:
        coros = [_summarize_row(entry) for entry in rows]
        results = await tqdm_asyncio.gather(*coros)
        return pd.DataFrame(results)

    out_df = asyncio.run(_run())
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dest_path, index=False)
    logger.info("Saved %d summaries to %s", len(out_df), dest_path)
    return out_df
