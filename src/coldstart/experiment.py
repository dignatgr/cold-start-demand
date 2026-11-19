"""Run every model on every forecast origin and collect the test predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldstart.config import (
    BASELINE,
    DEFAULT_CONFIG,
    KNN,
    LGB_IMG,
    LGB_TAB,
    MLP_IMG,
    MLP_TAB,
    Config,
)
from coldstart.features import pca_features, tabular_features
from coldstart.models import fit_lgbm_predict, knn_baseline, type_median_baseline
from coldstart.splits import rolling_origin_splits


def run_experiment(
    df: pd.DataFrame,
    emb: np.ndarray,
    cfg: Config = DEFAULT_CONFIG,
    use_mlp: bool = True,
    device: str | None = None,
) -> pd.DataFrame:
    """Predict the first-month units of the test articles of each origin with every model.

    Returns a long table with one row per (origin, article, model).
    """
    y = df["units"].to_numpy(dtype=float)
    X_tab = tabular_features(df)
    frames = []

    for split in rolling_origin_splits(df, cfg):
        print(
            f"origin {split.origin}: cutoff {split.cutoff.date()}  "
            f"fit={split.fit.sum()} val={split.val.sum()} test={split.test.sum()}"
        )
        preds = {
            BASELINE: type_median_baseline(df, split),
            KNN: knn_baseline(emb, y, split, cfg.knn_k),
            LGB_TAB: fit_lgbm_predict(X_tab, y, split, cfg.seed)[0],
        }
        X_img = pd.concat([X_tab, pca_features(emb, split.train, cfg.n_pca, cfg.seed)], axis=1)
        preds[LGB_IMG] = fit_lgbm_predict(X_img, y, split, cfg.seed)[0]

        if use_mlp:
            from coldstart import deep  # imported here: needs PyTorch

            inputs = deep.prepare_inputs(df, emb, split)
            preds[MLP_TAB], epochs_tab = deep.train_mlp(inputs, split, False, cfg.seed, device)
            preds[MLP_IMG], epochs_img = deep.train_mlp(inputs, split, True, cfg.seed, device)
            print(f"  MLP epochs (early stopping): tabular={epochs_tab}, + image={epochs_img}")

        test = df.loc[split.test, ["article_id", "product_code"]]
        for name, pred in preds.items():
            frames.append(
                pd.DataFrame(
                    {
                        "origin": split.origin,
                        "article_id": test["article_id"].to_numpy(),
                        "product_code": test["product_code"].to_numpy(),
                        "units": y[split.test],
                        "model": name,
                        "pred": pred,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)
