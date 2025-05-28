"""Primary deployment target — ONNX.

ONNX is the most portable runtime for on-device inference and is the format
we benchmark against first.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def export(model, sample_input: torch.Tensor, out_path: str, opset: int = 17):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model, sample_input, out_path,
        input_names=["ecg"], output_names=["forecast", "logits"],
        dynamic_axes={"ecg": {0: "batch"},
                      "forecast": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )
    return out_path


def parity_check(onnx_path: str, model, sample_input: torch.Tensor, atol: float = 1e-3):
    """Run the ONNX model in onnxruntime and check it matches torch within atol."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        f_t, l_t = model(sample_input)
    f_o, l_o = sess.run(None, {"ecg": sample_input.numpy()})
    f_ok = np.allclose(f_t.numpy(), f_o, atol=atol)
    l_ok = np.allclose(l_t.numpy(), l_o, atol=atol)
    return f_ok and l_ok, {
        "forecast_max_err": float(np.abs(f_t.numpy() - f_o).max()),
        "logits_max_err": float(np.abs(l_t.numpy() - l_o).max()),
    }


def quantize_onnx_dynamic(in_path: str, out_path: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)
    return out_path
