import numpy as np
import pytest

from coldstart.config import BASELINE, KNN, LGB_ASSORT, LGB_IMG, LGB_NN, LGB_TAB, LGB_TEXT
from coldstart.evaluate import metrics_by_origin, paired_deltas, pooled_metrics
from coldstart.experiment import run_experiment
from coldstart.run import main


@pytest.fixture(scope="module")
def preds(synthetic_dataset, cfg):
    df, emb = synthetic_dataset
    return run_experiment(df, emb, cfg, use_mlp=False)


def test_every_model_predicts_the_same_articles(preds):
    for _, g in preds.groupby("origin"):
        sets = [set(m["article_id"]) for _, m in g.groupby("model")]
        assert all(s == sets[0] for s in sets)
    assert {BASELINE, KNN, LGB_TAB, LGB_IMG} <= set(preds["model"])
    assert {LGB_ASSORT, LGB_TEXT, LGB_NN} <= set(preds["model"])
    assert np.isfinite(preds["pred"]).all()


def test_attributes_beat_the_type_median_baseline(preds, cfg):
    pooled = pooled_metrics(preds, cfg).set_index("model")
    assert pooled.loc[LGB_TAB, "wape"] < pooled.loc[BASELINE, "wape"]
    assert (pooled["wape_lo"] <= pooled["wape"]).all()
    assert (pooled["wape"] <= pooled["wape_hi"]).all()


def test_metric_tables(preds, cfg):
    by_origin = metrics_by_origin(preds)
    assert set(by_origin["origin"]) == {0, 1, 2}
    deltas = paired_deltas(preds, cfg)
    assert {"delta_wape", "delta_lo", "delta_hi"} <= set(deltas.columns)
    attr = deltas[deltas["comparison"].str.startswith(LGB_TAB)].iloc[0]
    assert attr["delta_wape"] < 0


def test_cli_writes_results(tmp_path):
    main(["--synthetic", "--no-mlp", "--n-boot", "50", "--out", str(tmp_path)])
    for name in ["results.md", "metrics_pooled.csv", "paired_deltas.csv", "wape_pooled.png"]:
        assert (tmp_path / name).exists()
