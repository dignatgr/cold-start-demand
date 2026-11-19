import numpy as np
import pandas as pd
import pytest

from coldstart.features import text_features
from coldstart.neighbours import neighbour_features, topk_cosine


def test_topk_cosine_matches_brute_force():
    rng = np.random.default_rng(0)
    query, ref = rng.normal(size=(40, 8)), rng.normal(size=(100, 8))
    idx, sim = topk_cosine(query, ref, k=5, chunk=16)

    qn = query / np.linalg.norm(query, axis=1, keepdims=True)
    rn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    full = qn @ rn.T
    expected = np.argsort(-full, axis=1)[:, :5]
    np.testing.assert_array_equal(idx, expected)
    np.testing.assert_allclose(sim, np.take_along_axis(full, expected, axis=1), rtol=1e-5)


def _toy_neighbours():
    # product 1 (rows 0, 1) and product 2 (rows 2, 3) look alike, product 3 is different;
    # rows 6 and 7 are test products
    emb = np.array(
        [[1, 0.00], [1, 0.01], [1, 0.02], [1, 0.03], [0.0, 1], [0.01, 1], [1, 0.05], [0.0, 1.0]]
    )
    units = np.array([100, 100, 10, 10, 50, 50, 10_000, 10_000], dtype=float)
    products = np.array([1, 1, 2, 2, 3, 3, 4, 5])
    train = np.array([True] * 6 + [False] * 2)
    return emb, units, products, train


def test_neighbours_never_use_the_same_product():
    emb, units, products, train = _toy_neighbours()
    nn = neighbour_features(emb, units, products, train, k=2, n_extra=3)
    # row 0 is a variant of product 1: its neighbours must come from product 2, not product 1
    assert nn.loc[0, "nn_median"] == 10
    assert nn.loc[1, "nn_median"] == 10
    # and symmetrically for product 2, whose closest other-product rows are product 1
    assert nn.loc[2, "nn_median"] == 100


def test_neighbours_do_not_use_test_targets():
    emb, units, products, train = _toy_neighbours()
    base = neighbour_features(emb, units, products, train, k=2, n_extra=3)
    changed = units.copy()
    changed[~train] = 1.0  # change the sales of the test articles
    other = neighbour_features(emb, changed, products, train, k=2, n_extra=3)
    pd.testing.assert_frame_equal(base, other)
    # the test article (row 6) looks like products 1 and 2, never like itself
    assert base.loc[6, "nn_median"] in {10.0, 55.0, 100.0}


def test_text_features_are_fitted_on_training_rows_only():
    rng = np.random.default_rng(1)
    words = [f"word{i}" for i in range(30)]
    text = [" ".join(rng.choice(words, 6)) for _ in range(200)]
    df = pd.DataFrame({"detail_desc": text})
    train = np.arange(200) < 150

    base = text_features(df, train, n_components=5)
    changed = df.copy()
    changed.loc[~train, "detail_desc"] = "completely different unseen vocabulary here"
    other = text_features(changed, train, n_components=5)

    assert base.shape == (200, 5)
    # training rows are unaffected by what the test rows say
    pd.testing.assert_frame_equal(
        base[train].reset_index(drop=True), other[train].reset_index(drop=True)
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            base[~train].reset_index(drop=True), other[~train].reset_index(drop=True)
        )
