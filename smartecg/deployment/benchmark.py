"""Latency + size benchmark across runtimes.

Single-window inference on CPU. Reports p50/p95 latency and file size for:
  - ONNX (primary)
  - Core ML (secondary)
  - TFLite (secondary)
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np


def _stats(times):
    arr = np.array(times) * 1000.0  # to ms
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "n": len(arr),
    }


def bench_onnx(path: str, sample: np.ndarray, warmup=10, iters=100):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: sample})
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, {name: sample})
        ts.append(time.perf_counter() - t)
    return _stats(ts)


def bench_coreml(path: str, sample: np.ndarray, warmup=10, iters=100):
    import coremltools as ct
    m = ct.models.MLModel(path)
    inp_name = list(m.get_spec().description.input)[0].name
    feed = {inp_name: sample.astype(np.float32)}
    for _ in range(warmup):
        m.predict(feed)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        m.predict(feed)
        ts.append(time.perf_counter() - t)
    return _stats(ts)


def bench_tflite(path: str, sample: np.ndarray, warmup=10, iters=100):
    try:
        import tensorflow.lite as tflite
    except ImportError:
        import tflite_runtime.interpreter as tflite
    itp = tflite.Interpreter(model_path=path)
    itp.allocate_tensors()
    in_idx = itp.get_input_details()[0]["index"]
    itp.set_tensor(in_idx, sample.astype(np.float32))
    for _ in range(warmup):
        itp.invoke()
    ts = []
    for _ in range(iters):
        itp.set_tensor(in_idx, sample.astype(np.float32))
        t = time.perf_counter()
        itp.invoke()
        ts.append(time.perf_counter() - t)
    return _stats(ts)


def file_size_kb(path: str) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1024
    # mlpackage / saved-model dirs
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=None)
    p.add_argument("--coreml", default=None)
    p.add_argument("--tflite", default=None)
    p.add_argument("--n-leads", type=int, default=12)
    p.add_argument("--t-in", type=int, default=500)
    args = p.parse_args()

    sample = np.random.randn(1, args.n_leads, args.t_in).astype(np.float32)
    results = {}
    if args.onnx:
        results["onnx"] = bench_onnx(args.onnx, sample)
        results["onnx"]["size_kb"] = file_size_kb(args.onnx)
    if args.coreml:
        results["coreml"] = bench_coreml(args.coreml, sample)
        results["coreml"]["size_kb"] = file_size_kb(args.coreml)
    if args.tflite:
        results["tflite"] = bench_tflite(args.tflite, sample)
        results["tflite"]["size_kb"] = file_size_kb(args.tflite)

    for k, v in results.items():
        print(f"{k}: p50={v['p50_ms']:.2f}ms p95={v['p95_ms']:.2f}ms size={v['size_kb']:.1f}KB")


if __name__ == "__main__":
    main()
