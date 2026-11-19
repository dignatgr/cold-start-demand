"""PyTorch MLP trained end to end on the new-product task (requires the ``deep`` extra)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from coldstart.config import CAT_COLS, NUM_COLS
from coldstart.splits import Split


@dataclass
class MLPInputs:
    cat_codes: np.ndarray  # int64, category code + 1 so that 0 means missing
    cat_sizes: list[int]
    num: np.ndarray  # standardised numeric features
    img: np.ndarray  # standardised image embedding
    y_log: np.ndarray  # log1p(units)


class NewProductMLP(nn.Module):
    """Learned embeddings for categorical attributes, numeric features, optional image branch."""

    def __init__(self, cat_sizes, n_num, n_img, use_img, emb_dim=8, hidden=128, p=0.3):
        super().__init__()
        self.use_img = use_img
        self.embs = nn.ModuleList([nn.Embedding(s + 1, emb_dim) for s in cat_sizes])
        d = emb_dim * len(cat_sizes) + n_num
        if use_img:
            self.img = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_img, 64), nn.ReLU())
            d += 64
        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, cat, num, img):
        parts = [e(cat[:, i]) for i, e in enumerate(self.embs)] + [num]
        if self.use_img:
            parts.append(self.img(img))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def prepare_inputs(df: pd.DataFrame, emb: np.ndarray, split: Split) -> MLPInputs:
    """Encode the inputs; scalers use the training articles of the split only."""
    cats = [df[c].astype("category") for c in CAT_COLS]
    cat_codes = np.stack([c.cat.codes.to_numpy() + 1 for c in cats], axis=1).astype("int64")
    cat_sizes = [len(c.cat.categories) for c in cats]
    train = split.train

    num = df[list(NUM_COLS)].to_numpy(dtype="float32")
    num = (num - num[train].mean(0)) / (num[train].std(0) + 1e-6)
    img = ((emb - emb[train].mean(0)) / (emb[train].std(0) + 1e-6)).astype("float32")
    y_log = np.log1p(df["units"].to_numpy(dtype="float32")).astype("float32")
    return MLPInputs(cat_codes, cat_sizes, num.astype("float32"), img, y_log)


def train_mlp(
    inputs: MLPInputs,
    split: Split,
    use_img: bool,
    seed: int = 0,
    device: str | None = None,
    epochs: int = 200,
    patience: int = 15,
    batch_size: int = 128,
) -> tuple[np.ndarray, int]:
    """Train with an L1 loss on log1p(units); early stopping on the validation set.

    The median is preserved by the monotone log transform, so the L1 loss targets the same
    quantity as the L1 LightGBM. Returns predictions in units for the test rows and the
    number of epochs run.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = NewProductMLP(inputs.cat_sizes, inputs.num.shape[1], inputs.img.shape[1], use_img).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.L1Loss()

    def tensors(mask):
        arrays = (inputs.cat_codes[mask], inputs.num[mask], inputs.img[mask], inputs.y_log[mask])
        return tuple(torch.as_tensor(a).to(device) for a in arrays)

    data = {"fit": tensors(split.fit), "val": tensors(split.val), "test": tensors(split.test)}
    best, best_state, wait, epochs_run = np.inf, None, 0, 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        perm = torch.randperm(len(data["fit"][0]), device=device)
        for batch in perm.split(batch_size):
            c, n, im, t = (x[batch] for x in data["fit"])
            opt.zero_grad()
            loss_fn(model(c, n, im), t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            c, n, im, t = data["val"]
            val_loss = loss_fn(model(c, n, im), t).item()
        if val_loss < best - 1e-4:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        c, n, im, _ = data["test"]
        pred = np.expm1(model(c, n, im).cpu().numpy())
    return np.clip(pred, 0, None), epochs_run
