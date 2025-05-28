"""Secondary target: Core ML mlprogram with INT8 weight quantization.

Demonstrates the mobile-neural-engine acceleration path on one platform
family. Not the primary export — see export_onnx.py.
"""
from __future__ import annotations
from pathlib import Path
import torch


def export(model, sample_input: torch.Tensor, out_path: str, quantize: bool = True):
    import coremltools as ct
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    traced = torch.jit.trace(model, sample_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="ecg", shape=sample_input.shape)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32,
    )

    if quantize:
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
        )
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear", weight_threshold=512)
        )
        mlmodel = linear_quantize_weights(mlmodel, config=cfg)

    mlmodel.save(out_path)
    return out_path
