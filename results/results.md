Articles: 74203; origins: 3

| origin | cutoff | n_fit | n_val | n_test |
|---|---|---|---|---|
| 0 | 2019-04-16 | 20935 | 3714 | 11085 |
| 1 | 2019-08-15 | 31633 | 5697 | 11209 |
| 2 | 2020-01-14 | 43694 | 7713 | 14868 |

Pooled test metrics (95% intervals from a bootstrap over products):

| Model | WAPE [95% CI] | Bias [95% CI] |
|---|---|---|
| Baseline (type median) | 0.892 [0.889, 0.896] | -0.798 [-0.804, -0.791] |
| kNN on image embeddings | 0.833 [0.827, 0.839] | -0.543 [-0.557, -0.528] |
| LightGBM, tabular | 0.710 [0.703, 0.717] | -0.471 [-0.484, -0.458] |
| LightGBM, tabular + image | 0.711 [0.704, 0.717] | -0.479 [-0.492, -0.466] |
| PyTorch MLP, tabular | 0.775 [0.768, 0.783] | -0.486 [-0.501, -0.470] |
| PyTorch MLP, tabular + image | 0.743 [0.735, 0.750] | -0.471 [-0.485, -0.457] |

Paired differences in WAPE (negative: the first model is better):

| WAPE difference (a vs b) | Delta [95% CI] | CI excludes 0 |
|---|---|---|
| LightGBM, tabular  vs  Baseline (type median) | -0.183 [-0.188, -0.177] | yes |
| kNN on image embeddings  vs  Baseline (type median) | -0.059 [-0.065, -0.053] | yes |
| LightGBM, tabular + image  vs  LightGBM, tabular | +0.001 [-0.001, +0.002] | no |
| PyTorch MLP, tabular + image  vs  PyTorch MLP, tabular | -0.033 [-0.040, -0.026] | yes |
| LightGBM, tabular + image  vs  PyTorch MLP, tabular + image | -0.032 [-0.039, -0.026] | yes |
