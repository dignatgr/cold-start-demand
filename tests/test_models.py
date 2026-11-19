import numpy as np
import pandas as pd

from coldstart.features import tabular_features
from coldstart.models import fit_lgbm_predict, knn_baseline, type_median_baseline
from coldstart.splits import Split


def _split(n_fit, n_val, n_test):
    n = n_fit + n_val + n_test
    idx = np.arange(n)
    return Split(
        0,
        pd.Timestamp("2020-01-01"),
        idx < n_fit,
        (idx >= n_fit) & (idx < n_fit + n_val),
        idx >= n_fit + n_val,
    )


def test_type_median_uses_training_rows_only():
    df = pd.DataFrame(
        {
            "product_type_name": ["a", "a", "a", "b", "a", "c"],
            "units": [2, 4, 6, 100, 1000, 7],
        }
    )
    split = _split(n_fit=3, n_val=1, n_test=2)  # test: type "a" (units 1000) and unseen "c"
    pred = type_median_baseline(df, split)
    assert pred[0] == 4  # median of the training "a" rows, the test row is ignored
    assert pred[1] == np.median([2, 4, 6, 100])  # unseen type -> global training median


def test_knn_returns_median_of_nearest_neighbours():
    emb = np.array(
        [[1, 0.0], [1, 0.1], [1, 0.2], [0.0, 1], [0.1, 1], [0.2, 1], [0.9, 0.1], [0.1, 0.9]]
    )
    y = np.array([5, 5, 5, 50, 50, 50, 0, 0], dtype=float)
    split = _split(n_fit=6, n_val=0, n_test=2)
    pred = knn_baseline(emb, y, split, k=3)
    assert pred.tolist() == [5.0, 50.0]


def test_lightgbm_learns_a_simple_signal(synthetic_dataset, cfg):
    from coldstart.splits import rolling_origin_splits

    df, _ = synthetic_dataset
    y = df["units"].to_numpy(dtype=float)
    split = rolling_origin_splits(df, cfg)[-1]
    pred, model = fit_lgbm_predict(tabular_features(df), y, split)
    assert pred.shape == (split.test.sum(),)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    assert model.best_iteration_ >= 1
