from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from textwrap import fill

from bubbola_gare.config import (
    RAW_PARQUET_PATH,
    SUMMARY_EMBEDDINGS_PATH,
    VENDOR_COLUMN,
)


def load_embeddings(path: Path, raw_path: Path = RAW_PARQUET_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure embeddings are numpy arrays for dimensionality reduction
    df["embedding_array"] = df["embedding"].apply(np.array)
    df["text_preview"] = df["text_raw"].fillna("").str.slice(stop=180)
    df["text_preview_wrapped"] = df["text_preview"].apply(lambda t: fill(t, width=70))

    # Enrich with tipologia from raw table if available
    try:
        raw = pd.read_parquet(raw_path)
        raw = raw.reset_index(drop=False).rename(columns={"index": "record_id"})
        if "tipologia" in raw.columns:
            df = df.merge(raw[["record_id", "tipologia"]], on="record_id", how="left")
    except Exception:
        df["tipologia"] = ""
    return df


def reduce_embeddings(df: pd.DataFrame, random_state: int = 42) -> np.ndarray:
    X = np.vstack(df["embedding_array"].to_list())
    reducer = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=min(30, max(5, len(df) // 10)),
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(X)


def build_figure(df: pd.DataFrame, coords: np.ndarray):
    df_plot = df.copy()
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]
    vendor_field = VENDOR_COLUMN if VENDOR_COLUMN in df_plot.columns else "vendor"
    df_plot[vendor_field] = df_plot[vendor_field].fillna("")
    if "tipologia" in df_plot.columns:
        df_plot["tipologia"] = df_plot["tipologia"].fillna("")
    else:
        df_plot["tipologia"] = ""
    counts = df_plot[vendor_field].value_counts(dropna=False)
    df_plot["vendor_label"] = df_plot[vendor_field].map(
        lambda v: f"{v} ({counts.get(v, 0)})"
    )
    color_col = "vendor_label"
    parts = []
    if "order_id" in df_plot.columns:
        parts.append(df_plot["order_id"].astype(str).radd("Order: "))
    parts.append(df_plot[vendor_field].astype(str).radd("Vendor: "))
    if "tipologia" in df_plot.columns:
        parts.append(df_plot["tipologia"].astype(str).radd("Tipo: "))
    parts.append(df_plot["text_preview_wrapped"].astype(str))
    df_plot["hovertext"] = parts[0]
    for p in parts[1:]:
        df_plot["hovertext"] += "<br>" + p

    # Build color arrays for vendor and tipologia using a fixed palette
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    def color_array(series: pd.Series) -> list:
        uniques = series.unique().tolist()
        color_map = {val: palette[i % len(palette)] for i, val in enumerate(uniques)}
        return [color_map[val] for val in series]

    vendor_colors = color_array(df_plot[color_col])
    tip_colors = color_array(df_plot["tipologia"])

    scatter_vendor = go.Scattergl(
        x=df_plot["x"],
        y=df_plot["y"],
        mode="markers",
        marker=dict(size=6, opacity=0.8, color=vendor_colors),
        customdata=df_plot["hovertext"],
        hovertemplate="%{customdata}<extra></extra>",
        name="Vendor",
        showlegend=False,
    )

    scatter_tip = go.Scattergl(
        x=df_plot["x"],
        y=df_plot["y"],
        mode="markers",
        marker=dict(size=6, opacity=0.8, color=tip_colors),
        customdata=df_plot["hovertext"],
        hovertemplate="%{customdata}<extra></extra>",
        name="Tipologia",
        visible=False,
        showlegend=False,
    )

    fig = go.Figure(data=[scatter_vendor, scatter_tip])
    fig.update_layout(title="Order embeddings (2D projection)")
    # Fix axes so isolating a vendor via legend does not rescale
    x_min, x_max = float(df_plot["x"].min()), float(df_plot["x"].max())
    y_min, y_max = float(df_plot["y"].min()), float(df_plot["y"].max())
    pad_x = (x_max - x_min) * 0.05 or 1.0
    pad_y = (y_max - y_min) * 0.05 or 1.0
    fig.update_xaxes(range=[x_min - pad_x, x_max + pad_x], autorange=False)
    fig.update_yaxes(range=[y_min - pad_y, y_max + pad_y], autorange=False)
    fig.update_layout(hoverlabel={"font": {"size": 9}}, hovermode="closest")

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="Vendor", method="update", args=[{"visible": [True, False]}, {"title": "Order embeddings (Vendor)"}]),
                    dict(label="Tipologia", method="update", args=[{"visible": [False, True]}, {"title": "Order embeddings (Tipologia)"}]),
                ],
                direction="down",
                showactive=True,
                x=1.05,
                xanchor="left",
                y=1,
                yanchor="top",
            )
        ]
    )
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Project embeddings to 2D and open an interactive scatter plot."
    )
    parser.add_argument("--embeddings", type=Path, default=SUMMARY_EMBEDDINGS_PATH)
    parser.add_argument("--raw", type=Path, default=RAW_PARQUET_PATH, help="Path to raw parquet (for tipologia)")
    parser.add_argument("--output", type=Path, default=Path("data/processed/embeddings_plot.html"))
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Optional cap on the number of points to plot (random sample).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    df = load_embeddings(args.embeddings, raw_path=args.raw)
    if args.limit and len(df) > args.limit:
        df = df.sample(args.limit, random_state=args.random_state)

    coords = reduce_embeddings(df, random_state=args.random_state)
    fig = build_figure(df, coords)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)
    print(f"Saved interactive plot to {args.output}")


if __name__ == "__main__":
    main()
