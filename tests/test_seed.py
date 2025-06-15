"""Multi-seed determinism: same seed → same shuffle; different seeds → different.

Catches the DataLoader-generator-omission bug where multi-seed runs all see
the same shuffle order and the seed-std collapses to ~0.
"""
import torch
from torch.utils.data import TensorDataset, DataLoader

from smartecg.utils.seed import make_torch_generator, set_seed


def _first_batch_indices(seed):
    """Return the indices of the first batch when DataLoader is seeded with `seed`."""
    set_seed(seed)
    ds = TensorDataset(torch.arange(200))
    g = make_torch_generator(seed)
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=g)
    return next(iter(loader))[0].tolist()


def test_same_seed_same_shuffle():
    a = _first_batch_indices(42)
    b = _first_batch_indices(42)
    assert a == b, f"same seed should reproduce shuffle, got {a} vs {b}"


def test_different_seed_different_shuffle():
    a = _first_batch_indices(42)
    b = _first_batch_indices(1337)
    assert a != b, (
        f"different seeds must produce different shuffles, got identical {a} — "
        "this means DataLoader is not seeing the per-seed generator and "
        "multi-seed std will collapse"
    )


def test_generator_independent_of_global_consumer():
    """If unrelated code consumes the global torch RNG between set_seed and
    DataLoader creation, the generator-based DataLoader should still be
    deterministic."""
    set_seed(42)
    g = make_torch_generator(42)
    # Some unrelated code burns through global RNG
    torch.randn(1000)
    ds = TensorDataset(torch.arange(200))
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=g)
    a = next(iter(loader))[0].tolist()

    set_seed(42)
    g = make_torch_generator(42)
    # No consumer this time
    ds = TensorDataset(torch.arange(200))
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=g)
    b = next(iter(loader))[0].tolist()

    assert a == b, "DataLoader shuffle should be isolated from global RNG state"
