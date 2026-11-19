import numpy as np
import pandas as pd

from coldstart.splits import rolling_origin_splits, temporal_split


def test_products_never_shared_between_sets(synthetic_dataset, cfg):
    df, _ = synthetic_dataset
    for split in rolling_origin_splits(df, cfg):
        codes = {
            name: set(df.loc[mask, "product_code"])
            for name, mask in [("fit", split.fit), ("val", split.val), ("test", split.test)]
        }
        assert not codes["fit"] & codes["test"]
        assert not codes["val"] & codes["test"]
        assert not codes["fit"] & codes["val"]


def test_training_targets_are_observed_before_the_cutoff(synthetic_dataset, cfg):
    df, _ = synthetic_dataset
    horizon = pd.Timedelta(weeks=cfg.horizon_weeks)
    for split in rolling_origin_splits(df, cfg):
        train = df[split.train]
        # the first-month window of every training article ends before the cutoff
        assert (train["launch_date"] + horizon <= split.cutoff).all()
        assert (df.loc[split.test, "group_launch"] >= split.cutoff).all()


def test_validation_is_later_than_fit(synthetic_dataset, cfg):
    df, _ = synthetic_dataset
    split = rolling_origin_splits(df, cfg)[-1]
    assert df.loc[split.val, "group_launch"].min() >= df.loc[split.fit, "group_launch"].min()
    assert df.loc[split.val, "group_launch"].median() > df.loc[split.fit, "group_launch"].median()


def test_test_windows_do_not_overlap(synthetic_dataset, cfg):
    df, _ = synthetic_dataset
    splits = rolling_origin_splits(df, cfg)
    seen = np.zeros(len(df), dtype=int)
    for split in splits:
        assert split.test.any()
        seen += split.test
    assert seen.max() == 1


def test_late_colour_variant_stays_out_of_training():
    """A variant launched after the embargo must not leak its sales into training."""
    cutoff = pd.Timestamp("2020-03-01")
    df = pd.DataFrame(
        {
            "product_code": [1, 1, 2, 3],
            "launch_date": pd.to_datetime(["2020-01-01", "2020-02-20", "2020-01-05", "2020-03-10"]),
            "group_launch": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-05", "2020-03-10"]
            ),
        }
    )
    split = temporal_split(df, cutoff, horizon_weeks=4, val_quantile=0.9)
    assert split.train.tolist() == [True, False, True, False]
    assert split.test.tolist() == [False, False, False, True]
