import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")
pytest.importorskip("tqdm")
pytest.importorskip("torchvision")

from coldstart import embeddings  # noqa: E402


class _FakeResNet(torch.nn.Module):
    """Stand-in for ResNet50 so the test needs neither pretrained weights nor internet."""

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, embeddings.EMB_DIM),
        )
        self.fc = torch.nn.Linear(embeddings.EMB_DIM, 1)

    def forward(self, x):
        return self.fc(self.body(x))


def _write_image(image_dir, article_id, color):
    path = embeddings.image_path(image_dir, article_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL_Image.new("RGB", (64, 64), color).save(path)


def test_image_path_layout(tmp_path):
    path = embeddings.image_path(tmp_path, 108775015)
    assert path == tmp_path / "010" / "0108775015.jpg"


def test_embeddings_skip_missing_images_and_use_the_cache(tmp_path, monkeypatch):
    image_dir, cache_dir = tmp_path / "images", tmp_path / "cache"
    _write_image(image_dir, 108775015, (200, 10, 10))
    _write_image(image_dir, 108775016, (10, 200, 10))
    calls = []

    def fake_resnet50(weights=None):
        calls.append(weights)
        return _FakeResNet()

    monkeypatch.setattr(embeddings.models, "resnet50", fake_resnet50)
    requested = np.array([108775016, 999, 108775015])  # 999 has no image

    ids, emb = embeddings.compute_embeddings(
        requested, image_dir, cache_dir, batch_size=2, num_workers=0, device="cpu"
    )
    assert ids.tolist() == [108775016, 108775015]  # order kept, missing image dropped
    assert emb.shape == (2, embeddings.EMB_DIM)
    assert not np.allclose(emb[0], emb[1])  # different images, different embeddings
    assert len(calls) == 1

    ids2, emb2 = embeddings.compute_embeddings(
        requested, image_dir, cache_dir, batch_size=2, num_workers=0, device="cpu"
    )
    assert len(calls) == 1  # everything came from the cache
    assert ids2.tolist() == ids.tolist()
    np.testing.assert_allclose(emb2, emb)
