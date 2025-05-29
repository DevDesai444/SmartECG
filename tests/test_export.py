"""ONNX export + parity smoke test.

Skipped if onnx/onnxruntime aren't installed in the dev env.
"""
import importlib
import tempfile
from pathlib import Path
import pytest
import torch

from smartecg.models import build_model


def _cfg():
    return {
        "data": {"input_seconds": 5, "forecast_seconds": 5, "sampling_rate": 100},
        "num_leads": 12,
        "classes": ["a", "b", "c", "d", "e"],
        "model": {"name": "itransformer", "d_model": 32, "n_heads": 4,
                  "n_layers": 2, "d_ff": 64, "dropout": 0.0},
    }


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx/onnxruntime not installed",
)
def test_onnx_export_parity():
    from smartecg.deployment.export_onnx import export, parity_check
    torch.manual_seed(0)
    model = build_model(_cfg()).eval()
    x = torch.randn(1, 12, 500)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "m.onnx"
        export(model, x, str(out))
        ok, errs = parity_check(str(out), model, x, atol=1e-3)
        assert ok, errs
