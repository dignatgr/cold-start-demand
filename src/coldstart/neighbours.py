"""Target features from the visually closest training articles, without leakage."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def topk_cosine(
    query: np.ndarray, ref: np.ndarray, k: int, chunk: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine similarities of the ``k`` closest rows of ``ref``, most similar first."""
    q, r = _normalise(query), _normalise(ref)
    k = min(k, len(r))
    idx = np.empty((len(q), k), dtype="int64")
    sim = np.empty((len(q), k), dtype="float32")
    for start in range(0, len(q), chunk):
        block = q[start : start + chunk] @ r.T
        part = np.argpartition(block, -k, axis=1)[:, -k:]
        part_sim = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        idx[start : start + chunk] = np.take_along_axis(part, order, axis=1)
        sim[start : start + chunk] = np.take_along_axis(part_sim, order, axis=1)
    return idx, sim


def neighbour_features(
    emb: np.ndarray,
    units: np.ndarray,
    product_codes: np.ndarray,
    train_mask: np.ndarray,
    k: int = 10,
    n_extra: int = 30,
) -> pd.DataFrame:
    """Sales of the ``k`` visually closest *training* articles, for every article.

    For each article, the neighbours are searched among the training articles of *other
    products*: colour variants of the same product look alike and would otherwise leak the
    article's own sales into its features. Excluding them for every row (training rows
    included) makes the training features behave like those of an unseen test product, which
    is what the model meets at prediction time.

    Returns ``nn_median`` (median units), ``nn_mean_log`` (mean of log1p units) and
    ``nn_dist`` (mean cosine distance, a measure of how typical the article is).
    """
    train_idx = np.flatnonzero(train_mask)
    units = np.asarray(units, dtype=float)
    product_codes = np.asarray(product_codes)

    # fetch extra neighbours to be able to drop the same-product ones
    idx, sim = topk_cosine(emb, emb[train_idx], k + n_extra)
    rows = train_idx[idx]
    same = product_codes[rows] == product_codes[:, None]

    # keep the first k other-product neighbours, in similarity order
    keep = np.argsort(same, axis=1, kind="stable")[:, :k]
    rows = np.take_along_axis(rows, keep, axis=1)
    invalid = np.take_along_axis(same, keep, axis=1)
    values = np.where(invalid, np.nan, units[rows])
    dist = np.where(invalid, np.nan, 1.0 - np.take_along_axis(sim, keep, axis=1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # rows without valid neighbours
        return pd.DataFrame(
            {
                "nn_median": np.nanmedian(values, axis=1),
                "nn_mean_log": np.nanmean(np.log1p(values), axis=1),
                "nn_dist": np.nanmean(dist, axis=1),
            }
        )
