"""Feature preparation shared by the models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from coldstart.config import CAT_COLS, NUM_COLS


def tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical attributes (as pandas categories) and numeric columns.

    Category labels are the only information taken from the full table; they carry no target
    information.
    """
    X = df[[*CAT_COLS, *NUM_COLS]].copy()
    for col in CAT_COLS:
        X[col] = pd.Categorical(X[col])
    return X


def pca_features(
    emb: np.ndarray, train_mask: np.ndarray, n_components: int, seed: int = 0
) -> pd.DataFrame:
    """Reduce image embeddings with a PCA fitted on the training articles only."""
    pca = PCA(n_components=n_components, random_state=seed).fit(emb[train_mask])
    return pd.DataFrame(pca.transform(emb), columns=[f"img_{i}" for i in range(n_components)])


def align_embeddings(
    df: pd.DataFrame, article_ids: np.ndarray, emb: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep the articles that have an embedding, so every model sees the same articles."""
    pos = pd.Index(article_ids).get_indexer(df["article_id"])
    ok = pos >= 0
    return df[ok].reset_index(drop=True), emb[pos[ok]]
