#!/usr/bin/env bash
set -euo pipefail

CKPT_ROOT="checkpoints"
OUT_ROOT="exports"
mkdir -p "$OUT_ROOT"

for m in itransformer lstm bilstm cnn1d transformer_t; do
  ckpt="$CKPT_ROOT/$m/best.pt"
  [ -f "$ckpt" ] || { echo "skip $m (no checkpoint)"; continue; }
  python3 - <<PY
import torch
from pathlib import Path
from smartecg.models import build_model
from smartecg.deployment import export_onnx, export_coreml

ckpt = torch.load("$ckpt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = build_model(cfg)
model.load_state_dict(ckpt["model"])
model.eval()
x = torch.randn(1, cfg["num_leads"], int(cfg["data"]["input_seconds"] * cfg["data"]["sampling_rate"]))

onnx_path = "$OUT_ROOT/$m.onnx"
export_onnx.export(model, x, onnx_path)
export_onnx.quantize_onnx_dynamic(onnx_path, "$OUT_ROOT/${m}_int8.onnx")

try:
    export_coreml.export(model, x, "$OUT_ROOT/${m}.mlpackage", quantize=True)
except Exception as e:
    print("coreml export skipped:", e)
PY
done
