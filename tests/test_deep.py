import numpy as np
import pytest

pytest.importorskip("torch")

from coldstart import deep  # noqa: E402
from coldstart.splits import rolling_origin_splits  # noqa: E402


def test_mlp_trains_and_predicts(synthetic_dataset, cfg):
    df, emb = synthetic_dataset
    split = rolling_origin_splits(df, cfg)[-1]
    inputs = deep.prepare_inputs(df, emb, split)
    for use_img in (False, True):
        pred, epochs = deep.train_mlp(inputs, split, use_img, seed=0, device="cpu", epochs=5)
        assert pred.shape == (split.test.sum(),)
        assert np.isfinite(pred).all() and (pred >= 0).all()
        assert 1 <= epochs <= 5
