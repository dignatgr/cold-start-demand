"""Feature preparation shared by the models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from coldstart.config import ASSORT_COLS, CAT_COLS, NUM_COLS


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


def assortment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assortment features (see ``data._add_assortment_features``), re-indexed from zero."""
    return df[list(ASSORT_COLS)].reset_index(drop=True)


def text_features(
    df: pd.DataFrame, train_mask: np.ndarray, n_components: int = 16, seed: int = 0
) -> pd.DataFrame:
    """TF-IDF of the product description reduced by an SVD, both fitted on training rows only."""
    text = df["detail_desc"].fillna("").astype(str).reset_index(drop=True)
    train_text = text[np.asarray(train_mask)]
    tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(tfidf.fit_transform(train_text))
    components = svd.transform(tfidf.transform(text))
    return pd.DataFrame(components, columns=[f"txt_{i}" for i in range(n_components)])
