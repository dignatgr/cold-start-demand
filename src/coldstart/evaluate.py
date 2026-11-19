"""Aggregate predictions into metric tables, paired comparisons and a figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from coldstart.config import (  # noqa: E402
    BASELINE,
    DEFAULT_CONFIG,
    KNN,
    LGB_IMG,
    LGB_TAB,
    MLP_IMG,
    MLP_TAB,
    Config,
)
from coldstart.metrics import bias, bootstrap_metrics, bootstrap_wape_delta, wape  # noqa: E402

# (model a, model b): reported as WAPE(a) - WAPE(b); negative means a is better
PAIRS = [
    (LGB_TAB, BASELINE),
    (KNN, BASELINE),
    (LGB_IMG, LGB_TAB),
    (MLP_IMG, MLP_TAB),
    (LGB_IMG, MLP_IMG),
]


def metrics_by_origin(preds: pd.DataFrame) -> pd.DataFrame:
    """WAPE and bias of every model on every forecast origin."""
    rows = []
    for (origin, model), g in preds.groupby(["origin", "model"], sort=False):
        rows.append(
            {
                "origin": origin,
                "model": model,
                "n_articles": len(g),
                "wape": wape(g["units"], g["pred"]),
                "bias": bias(g["units"], g["pred"]),
            }
        )
    return pd.DataFrame(rows)


def pooled_metrics(preds: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Metrics over all origins with product-level bootstrap intervals."""
    rows = []
    for model, g in preds.groupby("model", sort=False):
        res = bootstrap_metrics(
            g["units"].to_numpy(),
            g["pred"].to_numpy(),
            g["product_code"].to_numpy(),
            n_boot=cfg.n_boot,
            seed=cfg.seed,
        )
        rows.append({"model": model, "n_articles": len(g), **res})
    return pd.DataFrame(rows)


def paired_deltas(preds: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Paired bootstrap of the WAPE difference for the comparisons in ``PAIRS``."""
    wide = preds.pivot_table(
        index=["origin", "article_id", "product_code", "units"], columns="model", values="pred"
    ).reset_index()
    rows = []
    for a, b in PAIRS:
        if a not in wide.columns or b not in wide.columns:
            continue
        res = bootstrap_wape_delta(
            wide["units"].to_numpy(),
            wide[a].to_numpy(),
            wide[b].to_numpy(),
            wide["product_code"].to_numpy(),
            n_boot=cfg.n_boot,
            seed=cfg.seed,
        )
        rows.append({"comparison": f"{a}  vs  {b}", **res})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (no extra dependency)."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def format_pooled(pooled: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": pooled["model"],
            "WAPE [95% CI]": [
                f"{r.wape:.3f} [{r.wape_lo:.3f}, {r.wape_hi:.3f}]" for r in pooled.itertuples()
            ],
            "Bias [95% CI]": [
                f"{r.bias:+.3f} [{r.bias_lo:+.3f}, {r.bias_hi:+.3f}]" for r in pooled.itertuples()
            ],
        }
    )


def format_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "WAPE difference (a vs b)": deltas["comparison"],
            "Delta [95% CI]": [
                f"{r.delta_wape:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]"
                for r in deltas.itertuples()
            ],
            "CI excludes 0": [
                "yes" if (r.delta_lo > 0 or r.delta_hi < 0) else "no" for r in deltas.itertuples()
            ],
        }
    )


def plot_wape(pooled: pd.DataFrame, path: str | Path) -> None:
    """Horizontal bars of the pooled WAPE with 95% bootstrap intervals."""
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(pooled) + 1.2))
    err = [pooled["wape"] - pooled["wape_lo"], pooled["wape_hi"] - pooled["wape"]]
    ax.barh(pooled["model"], pooled["wape"], xerr=err, color="#4c72b0", capsize=3)
    ax.invert_yaxis()
    ax.set(xlabel="WAPE (lower is better)", title="Test WAPE, all origins pooled")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
