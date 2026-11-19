"""Build the new-product dataset from the H&M transactions and article tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coldstart.config import CAT_COLS, DEFAULT_CONFIG, Config


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``transactions_train.csv`` and ``articles.csv`` from the Kaggle competition folder."""
    raw = Path(raw_dir)
    articles_file = raw / "articles.csv"
    if not articles_file.exists():
        raise FileNotFoundError(
            f"{articles_file} not found. Attach the competition data to the notebook "
            "(Kaggle: 'Add Input' > Competitions > H&M Personalized Fashion Recommendations)."
        )
    tx = pd.read_csv(
        raw / "transactions_train.csv",
        usecols=["t_dat", "article_id", "price"],
        dtype={"article_id": "int64", "price": "float32"},
        parse_dates=["t_dat"],
    )
    articles = pd.read_csv(articles_file)
    return tx, articles


def build_new_product_dataset(
    tx: pd.DataFrame, articles: pd.DataFrame, cfg: Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """One row per new article with its first-``horizon_weeks`` unit sales.

    The launch date of an article is the date of its first sale. Articles already on sale when
    the data starts are left-censored (their true launch is unknown), and articles whose
    first ``horizon_weeks`` are not fully observed are dropped.

    Colour variants of a product share a ``product_code``; ``group_launch`` is the earliest
    launch date of the product, used to keep variants on the same side of a split.
    """
    data_start, data_end = tx["t_dat"].min(), tx["t_dat"].max()
    horizon = pd.Timedelta(weeks=cfg.horizon_weeks)

    launch = tx.groupby("article_id")["t_dat"].min().rename("launch_date")
    keep = launch[
        (launch >= data_start + pd.Timedelta(weeks=cfg.left_censor_weeks))
        & (launch + horizon <= data_end)
    ]
    tx = tx.merge(keep, left_on="article_id", right_index=True)
    tx = tx[tx["t_dat"] < tx["launch_date"] + horizon]

    df = (
        tx.groupby("article_id")
        .agg(units=("price", "size"), price=("price", "mean"), launch_date=("launch_date", "first"))
        .reset_index()
    )
    df = df.merge(
        articles[["article_id", "product_code", "detail_desc", *CAT_COLS]],
        on="article_id",
        how="left",
    )
    df = df.dropna(subset=["product_code"]).copy()
    df["product_code"] = df["product_code"].astype("int64")

    df["group_launch"] = df.groupby("product_code")["launch_date"].transform("min")
    df["launch_month"] = df["launch_date"].dt.month
    df["launch_week"] = df["launch_date"].dt.isocalendar().week.astype(int)
    df = _add_assortment_features(df, launch, articles, data_start, cfg)
    return df.reset_index(drop=True)


def _add_assortment_features(
    df: pd.DataFrame,
    launch: pd.Series,
    articles: pd.DataFrame,
    data_start: pd.Timestamp,
    cfg: Config,
) -> pd.DataFrame:
    """Features about the assortment around the launch, built without any sales information.

    * ``n_variants_week1``: colour variants launched within a week of the product launch.
      Variants added later may be a reaction to good sales, so they are not counted.
    * ``price_rel_type``: price relative to the median of the same product type launched in
      the same month.
    * ``n_launch_week`` and ``n_launch_week_section``: distinct products launched the same
      week (overall and in the same section). They are counted on every article with a known
      launch, including those too recent to have a complete first month, so the counts do
      not drop artificially at the end of the data.
    * ``month_sin`` and ``month_cos``: launch month on a circle.
    """
    eligible = launch[launch >= data_start + pd.Timedelta(weeks=cfg.left_censor_weeks)]
    known = (
        eligible.rename("launch_date")
        .reset_index()
        .merge(
            articles[["article_id", "product_code", "section_name"]], on="article_id", how="inner"
        )
    )
    known["week"] = known["launch_date"].dt.to_period("W").dt.start_time
    per_week = known.groupby("week")["product_code"].nunique().rename("n_launch_week")
    per_week_section = (
        known.groupby(["week", "section_name"])["product_code"]
        .nunique()
        .rename("n_launch_week_section")
        .reset_index()
    )

    df = df.copy()
    df["week"] = df["launch_date"].dt.to_period("W").dt.start_time
    df = df.merge(per_week, left_on="week", right_index=True, how="left")
    df = df.merge(per_week_section, on=["week", "section_name"], how="left")
    df = df.drop(columns="week")

    month = df["launch_date"].dt.to_period("M")
    df["price_rel_type"] = df["price"] / df.groupby(["product_type_name", month])[
        "price"
    ].transform("median")
    first_week = df["launch_date"] < df["group_launch"] + pd.Timedelta(days=7)
    df["n_variants_week1"] = first_week.groupby(df["product_code"]).transform("sum")
    angle = 2 * np.pi * df["launch_month"] / 12
    df["month_sin"] = np.sin(angle)
    df["month_cos"] = np.cos(angle)
    return df
