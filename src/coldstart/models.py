"""Baselines and gradient boosting (no deep learning dependency here)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from coldstart.splits import Split


def type_median_baseline(df: pd.DataFrame, split: Split) -> np.ndarray:
    """Median first-month units of the product type, computed on the training articles."""
    train = split.train
    medians = df[train].groupby("product_type_name", observed=True)["units"].median()
    fallback = df.loc[train, "units"].median()
    return (
        df.loc[split.test, "product_type_name"].map(medians).astype(float).fillna(fallback)
    ).to_numpy()


def knn_baseline(emb: np.ndarray, y: np.ndarray, split: Split, k: int = 10) -> np.ndarray:
    """Median units of the ``k`` visually closest training articles (cosine distance).

    Training and test products are disjoint, so a test article never finds its own colour
    variants among its neighbours.
    """
    train_idx = np.flatnonzero(split.train)
    test_idx = np.flatnonzero(split.test)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(emb[train_idx])
    _, neighbours = nn.kneighbors(emb[test_idx])
    return np.median(y[train_idx][neighbours], axis=1)


def fit_lgbm_predict(
    X: pd.DataFrame, y: np.ndarray, split: Split, seed: int = 0
) -> tuple[np.ndarray, lgb.LGBMRegressor]:
    """LightGBM with an L1 objective (aligned with WAPE), early-stopped on the validation set."""
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_estimators=3000,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X.loc[split.fit],
        y[split.fit],
        eval_set=[(X.loc[split.val], y[split.val])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return np.clip(model.predict(X.loc[split.test]), 0, None), model
