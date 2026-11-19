"""Error metrics and product-level (cluster) bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum of absolute errors over the sum of actuals."""
    y_true, pred = np.asarray(y_true, dtype=float), np.asarray(pred, dtype=float)
    return float(np.abs(y_true - pred).sum() / y_true.sum())


def bias(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Relative bias of the total: (sum of predictions - sum of actuals) / sum of actuals."""
    y_true, pred = np.asarray(y_true, dtype=float), np.asarray(pred, dtype=float)
    return float((pred.sum() - y_true.sum()) / y_true.sum())


def _group_sums(groups: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    codes, uniques = pd.factorize(groups)
    return tuple(np.bincount(codes, weights=v, minlength=len(uniques)) for v in values)


def _boot_indices(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_groups, size=(n_boot, n_groups))


def _interval(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """WAPE and bias with 95% intervals from a bootstrap that resamples whole products.

    Colour variants of a product have correlated sales, so resampling individual articles
    would give intervals that are too narrow.
    """
    y_true, pred = np.asarray(y_true, dtype=float), np.asarray(pred, dtype=float)
    abs_err, signed_err, total = _group_sums(groups, np.abs(y_true - pred), pred - y_true, y_true)
    idx = _boot_indices(len(total), n_boot, seed)
    wape_s = abs_err[idx].sum(1) / total[idx].sum(1)
    bias_s = signed_err[idx].sum(1) / total[idx].sum(1)
    w_lo, w_hi = _interval(wape_s)
    b_lo, b_hi = _interval(bias_s)
    return {
        "wape": wape(y_true, pred),
        "wape_lo": w_lo,
        "wape_hi": w_hi,
        "bias": bias(y_true, pred),
        "bias_lo": b_lo,
        "bias_hi": b_hi,
    }


def bootstrap_wape_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """WAPE(a) - WAPE(b) with a paired 95% interval (negative: model a is better).

    Both models are evaluated on the same resampled products, so the interval reflects
    the difference between the models rather than the noise of the test set.
    """
    y_true = np.asarray(y_true, dtype=float)
    err_a = np.abs(y_true - np.asarray(pred_a, dtype=float))
    err_b = np.abs(y_true - np.asarray(pred_b, dtype=float))
    sum_a, sum_b, total = _group_sums(groups, err_a, err_b, y_true)
    idx = _boot_indices(len(total), n_boot, seed)
    samples = (sum_a[idx].sum(1) - sum_b[idx].sum(1)) / total[idx].sum(1)
    lo, hi = _interval(samples)
    return {
        "delta_wape": float((sum_a.sum() - sum_b.sum()) / total.sum()),
        "delta_lo": lo,
        "delta_hi": hi,
    }
