import pandas as pd

from coldstart.config import CAT_COLS, Config
from coldstart.data import build_new_product_dataset


def _articles(ids_and_codes):
    rows = []
    for article_id, code in ids_and_codes:
        row = {"article_id": article_id, "product_code": code}
        row.update({c: f"{c}_a" for c in CAT_COLS})
        row["detail_desc"] = "a description"
        rows.append(row)
    return pd.DataFrame(rows)


def test_censoring_and_first_month_units():
    start = pd.Timestamp("2020-01-01")

    def day(n):
        return start + pd.Timedelta(days=n)

    tx = pd.DataFrame(
        [
            # article 1: sold from the first day of the data -> left-censored, dropped
            (day(0), 1, 0.01),
            (day(3), 1, 0.01),
            (day(130), 1, 0.01),  # also defines the end of the data
            # article 2: launched on day 20, 3 sales in its first 4 weeks, 1 later
            (day(20), 2, 0.02),
            (day(21), 2, 0.02),
            (day(25), 2, 0.04),
            (day(60), 2, 0.02),
            # article 3: launched on day 120, first 4 weeks not fully observed -> dropped
            (day(120), 3, 0.03),
            # article 4: launched on day 90, fully observed
            (day(90), 4, 0.05),
        ],
        columns=["t_dat", "article_id", "price"],
    )
    articles = _articles([(1, 10), (2, 20), (3, 30), (4, 20)])

    df = build_new_product_dataset(tx, articles, Config())

    assert set(df["article_id"]) == {2, 4}
    units = df.set_index("article_id")["units"]
    assert units[2] == 3 and units[4] == 1
    # colour variants share the product launch date, which is the earliest one
    assert (df["group_launch"] == day(20)).all()
    assert df.set_index("article_id").loc[2, "price"] > 0.02  # mean of 0.02, 0.02, 0.04


def test_assortment_features_use_no_sales_information():
    start = pd.Timestamp("2020-01-01")

    def day(n):
        return start + pd.Timedelta(days=n)

    tx = pd.DataFrame(
        [
            (day(0), 1, 0.01),  # old article: defines the start of the data
            (day(150), 1, 0.01),  # end of the data
            # product 20: two variants launched the same week, a third one much later
            (day(20), 2, 0.02),
            (day(22), 6, 0.04),
            (day(90), 4, 0.05),
            # product 70 launched the same week as product 20
            (day(23), 7, 0.03),
            # product 30 launched on day 140: too recent for a full first month
            (day(140), 3, 0.03),
        ],
        columns=["t_dat", "article_id", "price"],
    )
    articles = _articles([(1, 10), (2, 20), (3, 30), (4, 20), (6, 20), (7, 70)])
    df = build_new_product_dataset(tx, articles, Config()).set_index("article_id")

    assert set(df.index) == {2, 4, 6, 7}  # article 3 is too recent, article 1 censored
    # the variant launched on day 90 is not counted as part of the launch
    assert (df.loc[[2, 6, 4], "n_variants_week1"] == 2).all()
    assert df.loc[7, "n_variants_week1"] == 1
    # two distinct products (20 and 70) were launched that week
    assert (df.loc[[2, 6, 7], "n_launch_week"] == 2).all()
    assert df.loc[4, "n_launch_week"] == 1
    # the column set is the documented one and has no missing value
    for col in ["price_rel_type", "month_sin", "month_cos", "n_launch_week_section"]:
        assert df[col].notna().all()
