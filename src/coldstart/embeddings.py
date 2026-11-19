"""Frozen ResNet50 image embeddings, cached on disk (requires the ``deep`` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models
from tqdm.auto import tqdm

EMB_DIM = 2048


def image_path(image_dir: Path, article_id: int) -> Path:
    """H&M images are stored as ``images/<first 3 digits>/<10-digit article id>.jpg``."""
    aid = f"{article_id:010d}"
    return Path(image_dir) / aid[:3] / f"{aid}.jpg"


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, ids, image_dir, transform):
        self.ids, self.image_dir, self.transform = ids, image_dir, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(image_path(self.image_dir, self.ids[i])) as img:
            return self.transform(img.convert("RGB"))


def compute_embeddings(
    article_ids: np.ndarray,
    image_dir: str | Path,
    cache_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 2,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ids, embeddings)`` for the requested articles that have an image.

    A ResNet50 pretrained on ImageNet is used as a frozen feature extractor (no fine-tuning):
    each image becomes a 2048-dimensional vector. Embeddings are cached in ``cache_dir`` and
    only the missing ones are computed, so interrupted or repeated runs are cheap.
    """
    image_dir, cache_dir = Path(image_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_file, ids_file = cache_dir / "embeddings.npy", cache_dir / "article_ids.npy"

    if emb_file.exists() and ids_file.exists():
        cached, cached_ids = np.load(emb_file), np.load(ids_file)
    else:
        cached, cached_ids = np.zeros((0, EMB_DIM), dtype="float32"), np.zeros(0, dtype="int64")

    have = set(cached_ids.tolist())
    todo = [
        int(a) for a in article_ids if int(a) not in have and image_path(image_dir, int(a)).exists()
    ]
    if todo:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        loader = torch.utils.data.DataLoader(
            _ImageDataset(todo, image_dir, weights.transforms()),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        feats = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc="image embeddings"):
                feats.append(model(batch.to(device)).cpu().numpy())
        cached = np.vstack([cached, *feats]).astype("float32")
        cached_ids = np.concatenate([cached_ids, np.array(todo, dtype="int64")])
        np.save(emb_file, cached)
        np.save(ids_file, cached_ids)

    pos = pd.Index(cached_ids).get_indexer(np.asarray(article_ids))
    ok = pos >= 0
    return np.asarray(article_ids)[ok], cached[pos[ok]]
