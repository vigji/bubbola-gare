"""
Benchmark NL->SQL answers against local pandas ground truth.

What it does:
- Loads the same parquet used for DB loading (`orders_db_ready.parquet` by default).
- Builds ~50 natural-language questions (counts, sums, averages across years, vendors, regions, categories, contract types).
- Sends each question to the FastAPI NLQ endpoint (`/analytics/nlq`) and parses structured results.
- Compares the API answer to pandas-computed ground truth and reports a score.

Usage:
    PYTHONPATH=src uv run python analysis/benchmark_nlq.py \
        --data data/processed/orders_db_ready.parquet \
        --base-url http://localhost:8000

You can tweak the target dimension via .env; this script only reads the parquet and queries the API.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd
import requests


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_PATH = "data/processed/orders_db_ready.parquet"
TARGET_OPS = 100  # aim for at least this many diverse questions


Number = float | int


@dataclass
class Operation:
    name: str
    question: str
    expected_fn: Callable[[pd.DataFrame], Number]
    column: str
    max_rows: int = 20
    tolerance: float = 1e-2  # numeric tolerance


def _numeric(val) -> float:
    if val is None:
        return math.nan
    try:
        return float(val)
    except Exception:
        try:
            return float(str(val).replace(",", ""))
        except Exception:
            return math.nan


def _choose_top_values(series: pd.Series, n: int) -> List[str]:
    return [str(x) for x in series.dropna().value_counts().head(n).index.tolist()]


def build_operations(df: pd.DataFrame) -> List[Operation]:
    df = df.copy()
    vendors = _choose_top_values(df["vendor"], 5)
    regions = _choose_top_values(df["region"], 5)
    categories = _choose_top_values(df["category"], 5)
    contract_types = _choose_top_values(df["contract_type"], 5)
    years = [int(y) for y in sorted(df["year"].dropna().unique())]

    ops: List[Operation] = []

    # Overall scalars — baseline coverage of counts/distincts/totals.
    ops += [
        Operation(
            name="total_orders_all",
            question="How many purchase orders exist in total? Return column total_orders.",
            expected_fn=lambda d: int(len(d)),
            column="total_orders",
        ),
        Operation(
            name="distinct_vendors_all",
            question="How many distinct vendors exist across all orders? Return column distinct_vendors.",
            expected_fn=lambda d: int(d["vendor"].dropna().nunique()),
            column="distinct_vendors",
        ),
        Operation(
            name="distinct_categories_all",
            question="How many distinct categories exist? Return column distinct_categories.",
            expected_fn=lambda d: int(d["category"].dropna().nunique()),
            column="distinct_categories",
        ),
        Operation(
            name="distinct_regions_all",
            question="How many distinct regions appear in the orders? Return column distinct_regions.",
            expected_fn=lambda d: int(d["region"].dropna().nunique()),
            column="distinct_regions",
        ),
        Operation(
            name="total_amount_all",
            question="What is the total sum of amount across all orders? Return column total_amount.",
            expected_fn=lambda d: float(d["amount"].fillna(0).sum()),
            column="total_amount",
        ),
        Operation(
            name="average_amount_all",
            question="What is the average order amount overall? Return column avg_amount.",
            expected_fn=lambda d: float(d["amount"].mean()),
            column="avg_amount",
        ),
        Operation(
            name="median_amount_all",
            question="What is the median order amount overall? Return column median_amount.",
            expected_fn=lambda d: float(d["amount"].median()),
            column="median_amount",
        ),
        Operation(
            name="p90_amount_all",
            question="What is the 90th percentile (p90) of order amount overall? Return column p90_amount.",
            expected_fn=lambda d: float(d["amount"].quantile(0.90)),
            column="p90_amount",
        ),
        Operation(
            name="orders_over_10k",
            question="How many orders have amount greater than 10000? Return column total_orders.",
            expected_fn=lambda d: int((d["amount"] > 10000).sum()),
            column="total_orders",
        ),
        Operation(
            name="orders_between_500_2000",
            question="How many orders have amount between 500 and 2000 inclusive? Return column total_orders.",
            expected_fn=lambda d: int(((d["amount"] >= 500) & (d["amount"] <= 2000)).sum()),
            column="total_orders",
        ),
        Operation(
            name="orders_without_delivery_date",
            question="How many orders have a null delivery_date? Return column total_orders.",
            expected_fn=lambda d: int(d["delivery_date"].isna().sum()),
            column="total_orders",
        ),
        Operation(
            name="earliest_order_date",
            question="What is the earliest order_date? Return column earliest_date.",
            expected_fn=lambda d: d["order_date"].min().date().isoformat(),
            column="earliest_date",
        ),
        Operation(
            name="latest_order_date",
            question="What is the latest order_date? Return column latest_date.",
            expected_fn=lambda d: d["order_date"].max().date().isoformat(),
            column="latest_date",
        ),
    ]

    # Date-range and monthly style metrics (single-year/period to avoid repetition).
    if years:
        newest_year = years[-1]
        ops += [
            Operation(
                name=f"orders_in_{newest_year}",
                question=f"How many orders were placed in {newest_year}? Return column total_orders.",
                expected_fn=lambda d, y=newest_year: int((d['year'] == y).sum()),
                column="total_orders",
            ),
            Operation(
                name=f"amount_in_{newest_year}",
                question=f"What is the total amount of orders in {newest_year}? Return column total_amount.",
                expected_fn=lambda d, y=newest_year: float(d.loc[d['year'] == y, 'amount'].fillna(0).sum()),
                column="total_amount",
            ),
        ]
        # Pick a month with data if possible (forces agent to reason about grouping by period).
        month_counts = df.groupby(df["order_date"].dt.to_period("M")).size().sort_values(ascending=False)
        if not month_counts.empty:
            top_period = month_counts.index[0]  # Period
            y = top_period.year
            m = top_period.month
            ops.append(
                Operation(
                    name=f"orders_month_{y}_{m:02d}",
                    question=f"How many orders were placed in {y}-{m:02d}? Return column total_orders.",
                    expected_fn=lambda d, y=y, m=m: int(((d['order_date'].dt.year == y) & (d['order_date'].dt.month == m)).sum()),
                    column="total_orders",
                )
            )

    # Single best vendor by amount — top of leaderboard.
    if not df.empty and df["vendor"].notna().any():
        vendor_totals = (
            df.groupby("vendor")["amount"].sum().sort_values(ascending=False)
        )
        top_vendor = vendor_totals.index[0] if not vendor_totals.empty else None
        if top_vendor:
            ops.append(
                Operation(
                    name="top_vendor_by_amount",
                    question="Which vendor has the highest total order amount overall? Return vendor and total_amount, sorted by total_amount desc limit 1.",
                    expected_fn=lambda d, v=top_vendor: float(d.groupby("vendor")["amount"].sum().max()),
                    column="total_amount",
                )
            )

    # Vendor-focused (distinct flavors)
    for v in vendors[:3]:
        ops += [
            Operation(
                name=f"vendor_avg_ticket_{v}",
                question=f"What is the average order amount for vendor '{v}'? Return vendor and avg_amount.",
                expected_fn=lambda d, v=v: float(d.loc[d["vendor"] == v, "amount"].mean()),
                column="avg_amount",
            ),
            Operation(
                name=f"vendor_orders_over_5k_{v}",
                question=f"For vendor '{v}', how many orders exceed 5000 in amount? Return vendor and total_orders.",
                expected_fn=lambda d, v=v: int(((d["vendor"] == v) & (d["amount"] > 5000)).sum()),
                column="total_orders",
            ),
        ]

    # Region-focused variants
    for r in regions[:3]:
        ops += [
            Operation(
                name=f"region_order_count_{r}",
                question=f"How many orders are in region '{r}'? Return region and total_orders.",
                expected_fn=lambda d, r=r: int((d["region"] == r).sum()),
                column="total_orders",
            ),
            Operation(
                name=f"region_avg_amount_{r}",
                question=f"What is the average amount of orders in region '{r}'? Return region and avg_amount.",
                expected_fn=lambda d, r=r: float(d.loc[d["region"] == r, "amount"].mean()),
                column="avg_amount",
            ),
        ]

    # Category-focused variants
    for c in categories[:3]:
        ops += [
            Operation(
                name=f"category_order_count_{c}",
                question=f"How many orders are in category '{c}'? Return category and total_orders.",
                expected_fn=lambda d, c=c: int((d["category"] == c).sum()),
                column="total_orders",
            ),
            Operation(
                name=f"category_amount_sum_{c}",
                question=f"What is the total amount for category '{c}'? Return category and total_amount.",
                expected_fn=lambda d, c=c: float(d.loc[d["category"] == c, "amount"].fillna(0).sum()),
                column="total_amount",
            ),
            Operation(
                name=f"category_median_amount_{c}",
                question=f"What is the median amount for category '{c}'? Return category and median_amount.",
                expected_fn=lambda d, c=c: float(d.loc[d["category"] == c, "amount"].median()),
                column="median_amount",
            ),
        ]

    # Contract type variants
    for ct in contract_types[:3]:
        ops += [
            Operation(
                name=f"contract_order_count_{ct}",
                question=f"How many orders use contract type '{ct}'? Return contract_type and total_orders.",
                expected_fn=lambda d, ct=ct: int((d["contract_type"] == ct).sum()),
                column="total_orders",
            ),
            Operation(
                name=f"contract_amount_avg_{ct}",
                question=f"What is the average amount for contract type '{ct}'? Return contract_type and avg_amount.",
                expected_fn=lambda d, ct=ct: float(d.loc[d["contract_type"] == ct, "amount"].mean()),
                column="avg_amount",
            ),
        ]

    # Order type variants (if present)
    order_types = _choose_top_values(df.get("order_type", pd.Series(dtype=str)), 3)
    for ot in order_types:
        ops += [
            Operation(
                name=f"order_type_count_{ot}",
                question=f"How many orders have order_type '{ot}'? Return order_type and total_orders.",
                expected_fn=lambda d, ot=ot: int((d["order_type"] == ot).sum()),
                column="total_orders",
            ),
            Operation(
                name=f"order_type_amount_sum_{ot}",
                question=f"What is the total amount for order_type '{ot}'? Return order_type and total_amount.",
                expected_fn=lambda d, ot=ot: float(d.loc[d["order_type"] == ot, "amount"].fillna(0).sum()),
                column="total_amount",
            ),
        ]

    # Ratio/quality checks
    ops += [
        Operation(
            name="share_orders_over_5k",
            question="What fraction of orders (0-1) have amount greater than 5000? Return column fraction_over_5k.",
            expected_fn=lambda d: float(((d["amount"] > 5000).sum()) / len(d)),
            column="fraction_over_5k",
            tolerance=1e-3,
        ),
        Operation(
            name="orders_with_summary",
            question="How many orders have a non-null order_summary? Return column total_orders.",
            expected_fn=lambda d: int(d["order_summary"].notna().sum()) if "order_summary" in d.columns else 0,
            column="total_orders",
        ),
        Operation(
            name="avg_amount_with_delivery",
            question="What is the average amount for orders that have a delivery_date set? Return column avg_amount.",
            expected_fn=lambda d: float(d.loc[d["delivery_date"].notna(), "amount"].mean()),
            column="avg_amount",
        ),
    ]

    # Additional harder/more contextual benchmarks to increase diversity and reasoning load.
    if len(years) >= 2:
        latest, prev = years[-1], years[-2]
        # Reasoning: year-over-year growth requires computing two aggregates and comparing.
        ops.append(
            Operation(
                name="yoy_amount_growth_latest_vs_prev",
                question=f"What is the year-over-year growth rate of total amount from {prev} to {latest}? Return column yoy_growth (ratio).",
                expected_fn=lambda d, a=latest, b=prev: float(
                    (d.loc[d["year"] == a, "amount"].fillna(0).sum() - d.loc[d["year"] == b, "amount"].fillna(0).sum())
                    / max(d.loc[d["year"] == b, "amount"].fillna(0).sum(), 1e-9)
                ),
                column="yoy_growth",
                tolerance=1e-3,
            )
        )
        # Reasoning: average monthly run-rate in the latest year.
        ops.append(
            Operation(
                name="latest_year_avg_monthly_spend",
                question=f"What is the average monthly spend in {latest}? Return column avg_monthly_amount.",
                expected_fn=lambda d, y=latest: float(
                    d.loc[d["year"] == y].groupby(d["order_date"].dt.month)["amount"].sum().mean()
                ),
                column="avg_monthly_amount",
            )
        )

    # Reasoning: find the region with the highest total amount — requires grouping and ordering.
    if regions:
        ops.append(
            Operation(
                name="top_region_by_amount",
                question="Which region has the highest total order amount? Return region and total_amount, sorted desc limit 1.",
                expected_fn=lambda d: float(d.groupby("region")["amount"].sum().max()),
                column="total_amount",
            )
        )
    # Reasoning: find the category with most orders.
    if categories:
        ops.append(
            Operation(
                name="top_category_by_count",
                question="Which category has the highest number of orders? Return category and total_orders, sorted desc limit 1.",
                expected_fn=lambda d: int(d.groupby("category").size().max()),
                column="total_orders",
            )
        )
    # Reasoning: find the contract type with highest average amount.
    if contract_types:
        ops.append(
            Operation(
                name="contract_type_highest_avg",
                question="Which contract type has the highest average order amount? Return contract_type and avg_amount, sorted desc limit 1.",
                expected_fn=lambda d: float(d.groupby("contract_type")["amount"].mean().max()),
                column="avg_amount",
            )
        )

    # Reasoning: ratio of orders with missing vendor (data quality).
    ops.append(
        Operation(
            name="share_missing_vendor",
            question="What fraction of orders have a missing vendor? Return column missing_vendor_ratio.",
            expected_fn=lambda d: float(d["vendor"].isna().mean()),
            column="missing_vendor_ratio",
            tolerance=1e-3,
        )
    )

    # Reasoning: absolute spread between max and min amounts.
    ops.append(
        Operation(
            name="amount_spread",
            question="What is the difference between the maximum and minimum non-null order amount? Return column amount_spread.",
            expected_fn=lambda d: float(d["amount"].max() - d["amount"].dropna().min()),
            column="amount_spread",
        )
    )

    # Reasoning: identify most volatile category (highest stddev).
    if categories:
        ops.append(
            Operation(
                name="category_highest_stddev",
                question="Which category has the highest standard deviation of order amounts? Return category and std_amount, sorted desc limit 1.",
                expected_fn=lambda d: float(d.groupby("category")["amount"].std().max()),
                column="std_amount",
            )
        )

    # Reasoning: find vendor with most distinct categories.
    if vendors:
        ops.append(
            Operation(
                name="vendor_most_categories",
                question="Which vendor spans the largest number of distinct categories? Return vendor and distinct_categories, sorted desc limit 1.",
                expected_fn=lambda d: int(
                    d.groupby("vendor")["category"].nunique().max()
                ),
                column="distinct_categories",
            )
        )

    # Reasoning: top vendor in a given region by amount (intersection filter).
    if vendors and regions:
        v = vendors[0]
        r = regions[0]
        ops.append(
            Operation(
                name=f"vendor_{v}_amount_in_region_{r}",
                question=f"What is the total amount for vendor '{v}' within region '{r}'? Return vendor, region, total_amount.",
                expected_fn=lambda d, v=v, r=r: float(d.loc[(d["vendor"] == v) & (d["region"] == r), "amount"].fillna(0).sum()),
                column="total_amount",
            )
        )

    # Reasoning: top category per top vendor (join-like thinking).
    if vendors:
        v = vendors[0]
        ops.append(
            Operation(
                name=f"top_category_for_vendor_{v}",
                question=f"For vendor '{v}', which category has the highest total amount? Return category and total_amount, sorted desc limit 1.",
                expected_fn=lambda d, v=v: float(
                    d.loc[d["vendor"] == v].groupby("category")["amount"].sum().max()
                ),
                column="total_amount",
            )
        )

    # Reasoning: average amount per region conditioned on non-null summary.
    ops.append(
        Operation(
            name="avg_amount_by_region_with_summary",
            question="Among orders that have an order_summary, what is the average amount per region? Return region and avg_amount, sorted desc.",
            expected_fn=lambda d: float(
                d.loc[d["order_summary"].notna()].groupby("region")["amount"].mean().max()
            ),
            column="avg_amount",
        )
    )

    # Reasoning: share of total spend from top 3 vendors.
    if len(vendors) >= 3:
        ops.append(
            Operation(
                name="share_top3_vendors",
                question="What fraction of total spend comes from the top 3 vendors by total amount? Return column share_top3 (0-1).",
                expected_fn=lambda d, vs=vendors[:3]: float(
                    d.loc[d["vendor"].isin(vs), "amount"].sum() / max(d["amount"].sum(), 1e-9)
                ),
                column="share_top3",
                tolerance=1e-3,
            )
        )

    # Reasoning: earliest order per category requires grouping and min.
    if categories:
        c = categories[0]
        earliest_c = df.loc[df["category"] == c, "order_date"].min()
        if pd.notna(earliest_c):
            ops.append(
                Operation(
                    name=f"earliest_order_for_category_{c}",
                    question=f"What is the earliest order_date for category '{c}'? Return category and earliest_date.",
                    expected_fn=lambda d, c=c: d.loc[d["category"] == c, "order_date"].min().date().isoformat(),
                    column="earliest_date",
                )
            )

    # Reasoning: max amount order code (top order).
    ops.append(
        Operation(
            name="order_code_with_max_amount",
            question="Which order_code has the maximum amount? Return order_code and amount, sorted desc limit 1.",
            expected_fn=lambda d: float(d["amount"].max()),
            column="amount",
        )
    )

    # Reasoning: count of orders where amount equals amount_tot (if present).
    if "importo_tot" in df.columns:
        ops.append(
            Operation(
                name="orders_amount_equals_importo_tot",
                question="How many orders have amount equal to importo_tot? Return column total_orders.",
                expected_fn=lambda d: int((d["amount"] == d["importo_tot"]).sum()),
                column="total_orders",
            )
        )

    # Reasoning: proportion of orders per contract type within a region.
    if regions and contract_types:
        r = regions[0]
        ct = contract_types[0]
        ops.append(
            Operation(
                name=f"contract_share_in_region_{r}_{ct}",
                question=f"In region '{r}', what fraction of orders use contract type '{ct}'? Return column contract_share.",
                expected_fn=lambda d, r=r, ct=ct: float(
                    ((d["region"] == r) & (d["contract_type"] == ct)).sum()
                    / max((d["region"] == r).sum(), 1e-9)
                ),
                column="contract_share",
                tolerance=1e-3,
            )
        )

    # Reasoning: median amount for orders without delivery_date.
    ops.append(
        Operation(
            name="median_amount_missing_delivery",
            question="What is the median amount for orders with missing delivery_date? Return column median_amount.",
            expected_fn=lambda d: float(d.loc[d["delivery_date"].isna(), "amount"].median()),
            column="median_amount",
        )
    )

    # Reasoning: highest average ticket among regions with at least 50 orders (threshold filter).
    ops.append(
        Operation(
            name="region_highest_avg_with_min_volume",
            question="Among regions with at least 50 orders, which has the highest average order amount? Return region and avg_amount, sorted desc limit 1.",
            expected_fn=lambda d: float(
                d.groupby("region").filter(lambda g: len(g) >= 50).groupby("region")["amount"].mean().max()
                if not d.groupby("region").filter(lambda g: len(g) >= 50).empty
                else 0.0
            ),
            column="avg_amount",
        )
    )

    # Reasoning: orders per vendor in the latest month (freshness).
    if not df.empty:
        latest_month = df["order_date"].max().month
        latest_year = df["order_date"].max().year
        ops.append(
            Operation(
                name="orders_latest_month_all_vendors",
                question=f"How many orders were placed in {latest_year}-{latest_month:02d} overall? Return column total_orders.",
                expected_fn=lambda d, y=latest_year, m=latest_month: int(
                    ((d["order_date"].dt.year == y) & (d["order_date"].dt.month == m)).sum()
                ),
                column="total_orders",
            )
        )

    # Fill with generated combo questions until we reach the target count.
    extra_ops: List[Operation] = []

    # Combine vendor-region pairs for joint totals.
    for v in vendors:
        for r in regions:
            extra_ops.append(
                Operation(
                    name=f"vendor_region_amount_{v}_{r}",
                    question=f"What is the total amount for vendor '{v}' in region '{r}'? Return vendor, region, total_amount.",
                    expected_fn=lambda d, v=v, r=r: float(
                        d.loc[(d["vendor"] == v) & (d["region"] == r), "amount"].fillna(0).sum()
                    ),
                    column="total_amount",
                )
            )
    # Combine category-year counts for trend slices.
    for c in categories:
        for y in years:
            extra_ops.append(
                Operation(
                    name=f"category_orders_{c}_{y}",
                    question=f"In year {y}, how many orders are in category '{c}'? Return category, year, total_orders.",
                    expected_fn=lambda d, c=c, y=y: int(((d["category"] == c) & (d["year"] == y)).sum()),
                    column="total_orders",
                )
            )
    # Combine contract type and category averages.
    for ct in contract_types:
        for c in categories[:3]:
            extra_ops.append(
                Operation(
                    name=f"contract_category_avg_{ct}_{c}",
                    question=f"For contract type '{ct}' and category '{c}', what is the average amount? Return contract_type, category, avg_amount.",
                    expected_fn=lambda d, ct=ct, c=c: float(
                        d.loc[(d["contract_type"] == ct) & (d["category"] == c), "amount"].mean()
                    ),
                    column="avg_amount",
                )
            )

    # Append extras until we hit the target variety count.
    for op in extra_ops:
        if len(ops) >= TARGET_OPS:
            break
        ops.append(op)

    # Fallback padding to guarantee TARGET_OPS by varying thresholds (keeps agent challenged).
    thresholds = [100, 250, 500, 1000, 2000, 5000, 10000, 20000]
    t_idx = 0
    while len(ops) < TARGET_OPS and t_idx < len(thresholds):
        thr = thresholds[t_idx]
        ops.append(
            Operation(
                name=f"orders_over_threshold_{thr}",
                question=f"How many orders have amount greater than {thr}? Return column total_orders.",
                expected_fn=lambda d, thr=thr: int((d['amount'] > thr).sum()),
                column="total_orders",
            )
        )
        t_idx += 1

    return ops


def call_nlq(base_url: str, question: str, max_rows: int) -> Dict:
    resp = requests.post(
        f"{base_url.rstrip('/')}/analytics/nlq",
        json={"question": question, "max_rows": max_rows},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def compare_scalar(op: Operation, rows: List[Dict], expected) -> Tuple[bool, str]:
    if not rows:
        return False, "No rows returned"
    row = rows[0]
    val = row.get(op.column)
    if val is None:
        # fallback: use first numeric value in the row
        for k, v in row.items():
            if isinstance(v, (int, float)):
                val = v
                break
    if val is None:
        return False, f"Column '{op.column}' missing and no usable fallback found"

    # String/date comparison path
    if isinstance(expected, (str, pd.Timestamp)) or isinstance(val, str):
        exp_str = expected if isinstance(expected, str) else str(expected)
        act_str = val if isinstance(val, str) else str(val)
        ok = exp_str == act_str
        return ok, f"expected={exp_str}, actual={act_str}"

    # Numeric comparison path
    actual = _numeric(val)
    exp = _numeric(expected)
    if math.isnan(actual) or math.isnan(exp):
        return False, f"Unable to parse numeric comparison (actual={actual}, expected={exp})"

    diff = abs(actual - exp)
    ok = diff <= op.tolerance or diff <= abs(exp) * 0.001  # relative slack
    return ok, f"expected={exp:.4f}, actual={actual:.4f}, diff={diff:.4f}"


def run_operations(df: pd.DataFrame, base_url: str, ops: List[Operation]) -> None:
    # Ensure derived columns present (e.g., year) for expected_fn lambdas
    if "year" not in df.columns:
        df = df.copy()
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            df["year"] = df["order_date"].dt.year

    total = len(ops)
    passed = 0
    results = []

    for idx, op in enumerate(ops, start=1):
        try:
            expected = op.expected_fn(df)
            api_resp = call_nlq(base_url, op.question, op.max_rows)
            ok, detail = compare_scalar(op, api_resp.get("rows", []), expected)
        except Exception as exc:
            ok = False
            detail = f"Exception: {exc}"
        results.append((op.name, ok, detail))
        status = "✅" if ok else "❌"
        print(f"[{idx:02d}/{total:02d}] {status} {op.name}: {detail}")
        if ok:
            passed += 1

    score = passed / total * 100
    print(f"\nCompleted {total} operations. Passed: {passed}. Score: {score:.1f}%")

    if passed < total:
        print("\nFailures:")
        for name, ok, detail in results:
            if not ok:
                print(f" - {name}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NLQ responses against pandas ground truth.")
    parser.add_argument("--data", default=DEFAULT_PATH, help="Path to DB-ready parquet.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the FastAPI service.")
    args = parser.parse_args()

    df_raw = pd.read_parquet(args.data)
    df = df_raw.copy()
    if "order_date" not in df.columns:
        print("order_date column missing in dataset; cannot benchmark time-based queries.", file=sys.stderr)
        sys.exit(1)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["year"] = df["order_date"].dt.year

    if df.empty:
        print(f"Dataset at {args.data} is empty; aborting.", file=sys.stderr)
        sys.exit(1)

    ops = build_operations(df)
    run_operations(df, args.base_url, ops)


if __name__ == "__main__":
    main()
