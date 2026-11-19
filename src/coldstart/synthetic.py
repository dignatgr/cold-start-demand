"""Small synthetic stand-in for the H&M data, for tests, CI and quick demos.

Sales depend on the product type, the price and the launch month, so a model that uses the
attributes can beat a type-median baseline. Image embeddings are noisy copies of a per-type
centroid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldstart.config import CAT_COLS


def make_synthetic(
    n_products: int = 900, seed: int = 0, emb_dim: int = 32
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return ``(transactions, articles, article_ids, embeddings)`` with the H&M schema."""
    rng = np.random.default_rng(seed)
    n_types = 8
    type_scale = rng.lognormal(2.3, 0.6, n_types)
    centroids = rng.normal(size=(n_types, emb_dim))
    start = pd.Timestamp("2018-09-20")

    tx_rows, art_rows, ids, embs = [], [], [], []
    for p in range(n_products):
        code = 100000 + p
        t = int(rng.integers(n_types))
        product_launch = start + pd.Timedelta(days=int(rng.integers(0, 700)))
        price_z = float(rng.normal())
        for v in range(int(rng.integers(1, 4))):
            article_id = code * 1000 + v + 1
            launch = product_launch + pd.Timedelta(days=int(rng.integers(0, 14)) if v else 0)
            season = 1 + 0.4 * np.sin(2 * np.pi * launch.month / 12)
            lam = type_scale[t] * np.exp(-0.5 * price_z) * season * rng.lognormal(0, 0.3)
            units = max(1, int(rng.poisson(lam)))
            offsets = np.sort(rng.integers(0, 28, size=units))
            offsets[0] = 0  # the first sale defines the launch date
            price = float(0.02 * np.exp(0.3 * price_z))
            for d in offsets:
                tx_rows.append((launch + pd.Timedelta(days=int(d)), article_id, price))
            tx_rows.append((launch + pd.Timedelta(days=40), article_id, price))  # later sale

            art = {"article_id": article_id, "product_code": code}
            for i, col in enumerate(CAT_COLS):
                art[col] = f"{col}_{(t * (i + 1) + v * (i % 2)) % (n_types + i)}"
            art["product_type_name"] = f"type_{t}"
            art_rows.append(art)
            ids.append(article_id)
            embs.append(centroids[t] + rng.normal(scale=0.5, size=emb_dim))

    tx = pd.DataFrame(tx_rows, columns=["t_dat", "article_id", "price"])
    tx["price"] = tx["price"].astype("float32")
    return tx, pd.DataFrame(art_rows), np.array(ids, dtype="int64"), np.vstack(embs)
