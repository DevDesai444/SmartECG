#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="exports"
for m in itransformer lstm bilstm cnn1d transformer_t; do
  for suffix in ".onnx" "_int8.onnx" ".mlpackage"; do
    p="$OUT_ROOT/${m}${suffix}"
    [ -e "$p" ] || continue
    flag=""
    case "$suffix" in
      *.onnx) flag="--onnx" ;;
      *.mlpackage) flag="--coreml" ;;
      *.tflite) flag="--tflite" ;;
    esac
    echo "--- $m $suffix ---"
    python3 -m smartecg.deployment.benchmark $flag "$p"
  done
done
