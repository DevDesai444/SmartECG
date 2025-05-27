"""Forward + backward shape contracts for every model.

Runs on CPU with tiny tensors so it stays fast in CI.
"""
import torch
import pytest

from smartecg.models import build_model

B, N, T_IN, T_OUT, C = 2, 12, 500, 500, 5


def _cfg(name, **overrides):
    base = {
        "data": {"input_seconds": 5, "forecast_seconds": 5, "sampling_rate": 100},
        "num_leads": N,
        "classes": ["a", "b", "c", "d", "e"],
        "model": {"name": name, **overrides},
    }
    return base


@pytest.mark.parametrize("cfg", [
    _cfg("itransformer", d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0),
    _cfg("lstm", hidden=32, n_layers=1, dropout=0.0),
    _cfg("bilstm", hidden=16, n_layers=1, dropout=0.0),
    _cfg("cnn1d", channels=[16, 32, 32, 32], kernel=7, dropout=0.0),
    _cfg("transformer_t", patch_size=25, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0),
])
def test_forward_backward(cfg):
    model = build_model(cfg)
    x = torch.randn(B, N, T_IN, requires_grad=True)
    forecast, logits = model(x)
    assert forecast.shape == (B, N, T_OUT), f"{cfg['model']['name']} forecast wrong"
    assert logits.shape == (B, C), f"{cfg['model']['name']} logits wrong"
    (forecast.mean() + logits.mean()).backward()
    assert x.grad is not None


def test_itransformer_exposes_attention():
    cfg = _cfg("itransformer", d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0)
    model = build_model(cfg)
    x = torch.randn(B, N, T_IN)
    model(x)
    attns = model.get_last_attention()
    assert len(attns) == 2
    for a in attns:
        assert a.shape == (B, 4, N, N)
        # rows of attention should sum to ~1
        assert torch.allclose(a.sum(dim=-1), torch.ones_like(a.sum(dim=-1)), atol=1e-4)
