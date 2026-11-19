"""Leakage-free temporal splits at the product level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from coldstart.config import Config


@dataclass(frozen=True)
class Split:
    """Boolean masks over the rows of the dataset, plus the cutoff date of the origin."""

    origin: int
    cutoff: pd.Timestamp
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def train(self) -> np.ndarray:
        """Fit and validation rows together (used to fit scalers, PCA, neighbours...)."""
        return self.fit | self.val


def temporal_split(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_weeks: int,
    val_quantile: float = 0.85,
    test_end: pd.Timestamp | None = None,
    origin: int = 0,
) -> Split:
    """Split by product launch date with an embargo of one horizon before the cutoff.

    * test: products first launched in ``[cutoff, test_end)``;
    * train: products launched before ``cutoff - horizon`` whose own launch is also before
      that date, so every training target (the units of the first ``horizon_weeks``) is fully
      observed before the cutoff, even for colour variants launched later than the product;
    * validation: the latest ``1 - val_quantile`` share of the training products, used only
      for early stopping.

    Colour variants share a product code and the product launch date, so a product is never
    split between train and test.
    """
    horizon = pd.Timedelta(weeks=horizon_weeks)
    embargo = cutoff - horizon
    group_launch = df["group_launch"]

    test = group_launch >= cutoff
    if test_end is not None:
        test &= group_launch < test_end
    train = (group_launch < embargo) & (df["launch_date"] < embargo)

    val_cut = group_launch[train].quantile(val_quantile)
    val = train & (group_launch >= val_cut)
    fit = train & ~val
    return Split(origin, cutoff, fit.to_numpy(), val.to_numpy(), test.to_numpy())


def rolling_origin_splits(df: pd.DataFrame, cfg: Config) -> list[Split]:
    """One split per origin; the test window of an origin ends where the next origin starts."""
    cutoffs = [df["group_launch"].quantile(q) for q in cfg.origins]
    splits = []
    for i, cutoff in enumerate(cutoffs):
        test_end = cutoffs[i + 1] if i + 1 < len(cutoffs) else None
        split = temporal_split(df, cutoff, cfg.horizon_weeks, cfg.val_quantile, test_end, i)
        if not split.test.any() or not split.fit.any():
            raise ValueError(f"Origin {i} (cutoff {cutoff.date()}) has an empty train or test set")
        splits.append(split)
    return splits
