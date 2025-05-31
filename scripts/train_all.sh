#!/usr/bin/env bash
# train every model with 3 seeds; size-ablation pass first, then baselines
set -euo pipefail

SEEDS=(42 1337 2024)

# iTransformer size ablation
for cfg in itransformer_s itransformer_m itransformer_l; do
  for s in "${SEEDS[@]}"; do
    python3 -m smartecg.training.train --config configs/${cfg}.yaml --tag "s${s}"
  done
done

# baselines on the selected default iTransformer size config + the four others
for cfg in itransformer lstm bilstm cnn1d transformer_t; do
  for s in "${SEEDS[@]}"; do
    python3 -m smartecg.training.train --config configs/${cfg}.yaml --tag "s${s}"
  done
done

# 500Hz ablation
python3 -m smartecg.training.train --config configs/itransformer.yaml --tag "fs500" \
  || echo "(needs a 500Hz config override; create configs/itransformer_500.yaml first)"
