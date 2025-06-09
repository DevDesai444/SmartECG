#!/usr/bin/env bash
set -e
export SMARTECG_FORCE_CPU=1
export WANDB_DISABLED=true

for cfg in bilstm cnn1d transformer_t itransformer_s itransformer_l; do
    if [ -f "runs/${cfg}/test_predictions.npz" ]; then
        echo "==SKIP $cfg (already has results)=="
        continue
    fi
    echo "==START $cfg=="
    python3 -u -m smartecg.training.train \
        --config configs/${cfg}.yaml --epochs 6 --tag default 2>&1 \
        | tee /tmp/${cfg}.log
    echo "==DONE $cfg=="
done
echo "==ALL DONE=="
