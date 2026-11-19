# Cold-start demand prediction for new fashion articles

![CI](https://github.com/dignatgr/cold-start-demand/actions/workflows/ci.yml/badge.svg)

**Question.** A fashion article has no sales history when it launches, and last year's similar
products are a poor guide because trends change. How many units will it sell in its first four
weeks, given its attributes and its picture?

This repository compares a type-median baseline, a visual nearest-neighbour baseline, gradient
boosting and a small neural network on the public H&M data, with an evaluation designed to avoid
the usual leaks: split by product launch date, embargo, colour variants kept together, and
confidence intervals that resample whole products.

It is a regression on a cross-section of launches, not time-series forecasting. Time only enters
through the train/test split.

## Data

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
(Kaggle): transactions from September 2018 to September 2020, the article table and the product
images. The data is not redistributed here; accept the competition rules on Kaggle to use it.

**Target.** Units sold during the first 4 weeks after an article's first sale. Articles already
on sale when the data starts are dropped (their true launch is unknown), as are articles whose
first four weeks are not fully observed.

## Models

| Model | Inputs |
|---|---|
| Baseline | Median units of the product type |
| kNN | Median units of the 10 visually closest training articles (cosine distance on ResNet50 embeddings) |
| LightGBM, tabular | 9 categorical attributes, price, launch month and week; L1 objective |
| LightGBM, tabular + image | Same, plus a 32-component PCA of the image embedding |
| PyTorch MLP, tabular | Learned embeddings for the categories plus numeric features; L1 loss on `log1p(units)` |
| PyTorch MLP, tabular + image | Same, plus a dropout + linear branch on the image embedding |

Image embeddings come from a ResNet50 pretrained on ImageNet, used as a frozen feature extractor.

## Evaluation protocol

- **Split by product launch date.** Test products launch after a cutoff; training products launch
  at least four weeks before it, and so do all their colour variants, so every training target is
  fully observed before the cutoff.
- **Colour variants stay together.** Variants share a `product_code` and are never split between
  train and test. A test article therefore never finds its own variants among its visual
  neighbours.
- **Rolling origin.** Three cutoffs (50th, 65th and 80th percentile of launch dates). The test
  window of an origin ends where the next one starts, so no article is tested twice.
- **Validation.** The latest 15% of the training products are used only for early stopping.
- **Metrics.** WAPE (sum of absolute errors over the sum of actuals) and relative bias of the
  total. 95% intervals come from a bootstrap that resamples whole products; differences between
  models use a paired bootstrap on the same resampled products.

The properties above are covered by tests (`tests/test_splits.py`): no product shared between
sets, training windows closed before the cutoff, non-overlapping test windows.

## Results

### First analysis (single temporal split)

Test: products first launched from 2020-01-14 (14,868 articles; 48,993 articles to fit and 8,681
for validation). Lower WAPE is better.

| Model | WAPE | Bias |
|---|---|---|
| Baseline (type median) | 0.881 | -0.764 |
| LightGBM, tabular | 0.655 | -0.308 |
| LightGBM, tabular + image | 0.650 | -0.309 |
| PyTorch MLP, tabular | 0.718 | -0.362 |
| PyTorch MLP, tabular + image | 0.692 | -0.253 |

What this shows, with the caveat that it is one split without intervals:

- Product attributes carry most of the signal: LightGBM cuts WAPE from 0.88 to 0.66.
- Image embeddings add little on top of the attributes for LightGBM (0.650 against 0.655) and a
  little more for the MLP (0.692 against 0.718). Neither difference is tested here.
- LightGBM beats the MLP in both variants.
- Every model under-predicts total volume (negative bias). This is expected from an L1 objective,
  which targets the median of a very right-skewed distribution.

### Rolling-origin evaluation with confidence intervals

<!-- RESULTS:START -->
Run `python -m coldstart.run` and paste `results/results.md` here.
<!-- RESULTS:END -->

## Limitations

- Only articles with at least one sale are observed, so the model predicts sales *given* that the
  article sold.
- Price is the observed average price, not a planned price; there is no price-elasticity model.
- Image embeddings are generic ImageNet features, not fashion-specific, and there is no text
  feature from the product description.
- An L1 objective predicts the median and under-predicts totals. For volume planning, a
  mean-targeting loss (Poisson or Tweedie) or quantile predictions would be worth testing.
- One retailer and one two-year window.

## Reproduce

### On Kaggle

Attach the competition data (*Add Input* > Competitions), turn on a GPU and Internet, then:

```python
!git clone https://github.com/dignatgr/cold-start-demand
!pip install -q -e "cold-start-demand[deep]"
!python -m coldstart.run --raw-dir /kaggle/input/competitions/h-and-m-personalized-fashion-recommendations --out results
```

Embedding the ~75,000 images is the slow step; embeddings are cached under `cache/` and only
missing ones are computed. For a quick check, add `--max-products 3000`.

### Locally, on synthetic data

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python -m coldstart.run --synthetic --no-mlp --out results_demo
```

Add the `deep` extra (`pip install -e ".[dev,deep]"`) to also run the MLP and image-embedding
tests.

### Outputs

`python -m coldstart.run` writes to `results/`: `results.md` (tables ready to paste),
`metrics_pooled.csv`, `metrics_by_origin.csv`, `paired_deltas.csv`, `splits.csv`,
`wape_pooled.png` and the raw predictions. `notebooks/results.ipynb` reads these files.

## Repository layout

```
src/coldstart/
  data.py          new-product dataset from transactions and articles
  splits.py        leak-free temporal splits, rolling origins
  features.py      tabular features, PCA of embeddings
  embeddings.py    cached ResNet50 embeddings
  models.py        type-median and kNN baselines, LightGBM
  deep.py          PyTorch MLP
  metrics.py       WAPE, bias, product-level bootstrap
  experiment.py    runs every model on every origin
  evaluate.py      metric tables, paired comparisons, figure
  synthetic.py     small synthetic data for tests and demos
  run.py           command line entry point
tests/             unit tests and an end-to-end test on synthetic data
```

## License

MIT
