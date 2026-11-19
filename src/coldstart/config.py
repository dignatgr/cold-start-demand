"""Shared constants and the experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass

CAT_COLS: tuple[str, ...] = (
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "department_name",
    "index_name",
    "section_name",
    "garment_group_name",
)
NUM_COLS: tuple[str, ...] = ("price", "launch_month", "launch_week")

# Model names, used as keys in the prediction tables
BASELINE = "Baseline (type median)"
KNN = "kNN on image embeddings"
LGB_TAB = "LightGBM, tabular"
LGB_IMG = "LightGBM, tabular + image"
MLP_TAB = "PyTorch MLP, tabular"
MLP_IMG = "PyTorch MLP, tabular + image"


@dataclass(frozen=True)
class Config:
    """Experiment settings.

    Attributes:
        horizon_weeks: Length of the target window after an article's first sale.
        left_censor_weeks: Articles first sold within this many weeks of the start of the
            data are dropped (they were probably already on sale).
        seed: Random seed for all models.
        n_pca: Number of PCA components of the image embedding used by LightGBM.
        knn_k: Number of neighbours of the visual kNN baseline.
        origins: Forecast origins, as quantiles of the product launch date. The test window
            of an origin runs until the next origin (the last one runs to the end of the data).
        val_quantile: The latest part of the training period, by launch date, used for
            early stopping.
        n_boot: Number of bootstrap replicates for confidence intervals.
    """

    horizon_weeks: int = 4
    left_censor_weeks: int = 2
    seed: int = 0
    n_pca: int = 32
    knn_k: int = 10
    origins: tuple[float, ...] = (0.5, 0.65, 0.8)
    val_quantile: float = 0.85
    n_boot: int = 1000


DEFAULT_CONFIG = Config()
