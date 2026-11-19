import numpy as np
import pytest

from coldstart.metrics import bias, bootstrap_metrics, bootstrap_wape_delta, wape


def test_wape_and_bias_values():
    y = np.array([10.0, 10.0])
    pred = np.array([5.0, 15.0])
    assert wape(y, pred) == pytest.approx(0.5)
    assert bias(y, pred) == pytest.approx(0.0)
    assert bias(y, np.array([5.0, 5.0])) == pytest.approx(-0.5)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.poisson(10, 400).astype(float) + 1
    pred = y * rng.lognormal(0, 0.4, 400)
    groups = np.repeat(np.arange(200), 2)
    res = bootstrap_metrics(y, pred, groups, n_boot=300)
    assert res["wape_lo"] <= res["wape"] <= res["wape_hi"]
    assert res["bias_lo"] <= res["bias"] <= res["bias_hi"]


def test_perfect_predictions_have_zero_error():
    y = np.arange(1.0, 21.0)
    res = bootstrap_metrics(y, y, np.arange(20), n_boot=50)
    assert res["wape"] == res["wape_lo"] == res["wape_hi"] == 0.0


def test_paired_delta_sign_and_identical_models():
    rng = np.random.default_rng(2)
    y = rng.poisson(20, 600).astype(float) + 1
    good = y * rng.lognormal(0, 0.1, 600)
    bad = y * rng.lognormal(0, 0.6, 600)
    groups = np.repeat(np.arange(300), 2)

    better = bootstrap_wape_delta(y, good, bad, groups, n_boot=300)
    assert better["delta_wape"] < 0 and better["delta_hi"] < 0

    same = bootstrap_wape_delta(y, good, good, groups, n_boot=100)
    assert same["delta_wape"] == same["delta_lo"] == same["delta_hi"] == 0.0


def test_resampling_whole_products_gives_wider_intervals():
    """Variants of a product share their error, so clustering widens the interval."""
    rng = np.random.default_rng(3)
    n_products = 150
    product_error = rng.lognormal(0, 0.5, n_products)
    groups = np.repeat(np.arange(n_products), 4)
    y = np.full(len(groups), 10.0)
    pred = y * np.repeat(product_error, 4)

    clustered = bootstrap_metrics(y, pred, groups, n_boot=500)
    naive = bootstrap_metrics(y, pred, np.arange(len(y)), n_boot=500)
    assert clustered["wape_hi"] - clustered["wape_lo"] > naive["wape_hi"] - naive["wape_lo"]
