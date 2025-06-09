"""Export the best iTransformer checkpoint and benchmark every runtime.

- ONNX (primary): export, parity check, INT8 dynamic quantize, latency benchmark.
- Core ML (secondary): export with weight quantization, latency benchmark.
- TFLite (secondary): skipped when onnx-tf isn't installed (common on M-series
  macs); benchmark just records "skipped" in that case.

Outputs go to:
    exports/itransformer.onnx
    exports/itransformer_int8.onnx
    exports/itransformer.mlpackage
    runs/deployment.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch

from smartecg.models import build_model
from smartecg.deployment import export_onnx, benchmark


REPO = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO / "exports"


def file_size_kb(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size / 1024
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = REPO / "checkpoints" / "itransformer" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    t_in = int(cfg["data"]["input_seconds"] * cfg["data"]["sampling_rate"])
    n_leads = cfg["num_leads"]
    sample = torch.randn(1, n_leads, t_in)

    results = {"input_shape": list(sample.shape)}

    # --- ONNX FP32 export + parity ---
    onnx_path = EXPORT_DIR / "itransformer.onnx"
    export_onnx.export(model, sample, str(onnx_path))
    ok, errs = export_onnx.parity_check(str(onnx_path), model, sample, atol=1e-3)
    print(f"ONNX FP32 parity: {ok}  max_err={errs}")
    results["onnx_fp32"] = {
        "parity_ok": ok,
        "max_err": errs,
        "size_kb": file_size_kb(onnx_path),
        **benchmark.bench_onnx(str(onnx_path), sample.numpy()),
    }

    # --- ONNX INT8 dynamic quantize ---
    onnx_int8 = EXPORT_DIR / "itransformer_int8.onnx"
    export_onnx.quantize_onnx_dynamic(str(onnx_path), str(onnx_int8))
    # parity vs FP32 ONNX (INT8 won't match torch exactly, but should match its source)
    import onnxruntime as ort
    s_fp = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(str(onnx_int8), providers=["CPUExecutionProvider"])
    f_fp, l_fp = s_fp.run(None, {"ecg": sample.numpy()})
    f_q, l_q = s_int8.run(None, {"ecg": sample.numpy()})
    int8_logit_err = float(np.abs(l_fp - l_q).max())
    print(f"ONNX INT8 logits max-err vs FP32: {int8_logit_err:.4f}")
    results["onnx_int8"] = {
        "logits_max_err_vs_fp32": int8_logit_err,
        "size_kb": file_size_kb(onnx_int8),
        **benchmark.bench_onnx(str(onnx_int8), sample.numpy()),
    }

    # --- Core ML ---
    try:
        from smartecg.deployment import export_coreml
        ml_path = EXPORT_DIR / "itransformer.mlpackage"
        export_coreml.export(model, sample, str(ml_path), quantize=True)
        results["coreml"] = {
            "size_kb": file_size_kb(ml_path),
            **benchmark.bench_coreml(str(ml_path), sample.numpy()),
        }
        print("CoreML exported")
    except Exception as e:
        print(f"CoreML export skipped: {e}")
        results["coreml"] = {"skipped": str(e)}

    # --- TFLite ---
    try:
        from smartecg.deployment import export_tflite
        tfl_path = EXPORT_DIR / "itransformer.tflite"
        export_tflite.export(str(onnx_path), str(tfl_path))
        results["tflite"] = {
            "size_kb": file_size_kb(tfl_path),
            **benchmark.bench_tflite(str(tfl_path), sample.numpy()),
        }
        print("TFLite exported")
    except Exception as e:
        print(f"TFLite export skipped: {e}")
        results["tflite"] = {"skipped": str(e)}

    out = REPO / "runs" / "deployment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults written to {out}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
