import pytest

from coldstart.config import Config
from coldstart.data import build_new_product_dataset
from coldstart.features import align_embeddings
from coldstart.synthetic import make_synthetic


@pytest.fixture(scope="session")
def cfg():
    return Config(n_boot=100)


@pytest.fixture(scope="session")
def synthetic_dataset(cfg):
    """(dataset, embeddings) built from the synthetic H&M-like tables."""
    tx, articles, ids, emb = make_synthetic(n_products=900, seed=0)
    df = build_new_product_dataset(tx, articles, cfg)
    return align_embeddings(df, ids, emb)
