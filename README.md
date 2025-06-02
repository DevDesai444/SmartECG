# SmartECG

Short-window cardiac event forecasting on 12-lead ECG, with an iTransformer
that attends across leads (not time). Built to explore whether variate-axis
attention is the right inductive bias for multivariate biosignals, and what it
costs to put such a model on a resource-constrained wearable.

> **Motivation.** Most published ECG models are *classifiers* that operate on
> a window that already contains the event. The clinical value of a prediction
> depends on how early it arrives. This project asks: can we *forecast* the
> next-5s 12-lead waveform from the previous 5s, jointly classify the events
> implied by that forecast, and ship the resulting model to a wearable-grade
> device?

## What this repo contains

- A from-scratch PyTorch implementation of **iTransformer** (Liu et al., ICLR
  2024) adapted for joint forecasting + multi-label classification. No
  third-party iTransformer libraries; no `nn.Transformer` /
  `nn.MultiheadAttention` — Q/K/V projections, multi-head split, softmax
  attention, and FFN are all written explicitly. See
  [`smartecg/models/itransformer.py`](smartecg/models/itransformer.py).
- Four baseline encoders that share the same dual heads as iTransformer, so
  the cross-architecture comparison isolates the choice of attention axis:
  LSTM, Bi-LSTM, 1D-CNN, and a time-axis Transformer.
- The full pipeline against **PTB-XL** (PhysioNet, 21,837 records, 10s,
  12-lead): preprocessing, 5-class label mapping, the official 10-fold
  stratified split, a PyTorch `Dataset` with caching.
- Three complementary interpretability views: variate-attention heatmaps
  (native to the model), SHAP per-lead importance, and Integrated Gradients
  temporal saliency — plus a Streamlit dashboard for per-class metrics broken
  down by age, sex, and recording site.
- On-device deployment: post-training INT8 quantization and exports to **ONNX
  (primary)**, **Core ML**, and **TFLite**. Latency p50/p95 + size benchmark
  per runtime.

## Architecture

```
            ┌────────────────────────────────────────────────────────┐
            │  ECG x ∈ ℝ^{12 × 500}   (12 leads × 5s @ 100Hz)        │
            └──────────────────────────┬─────────────────────────────┘
                                       │
                          per-variate Linear(T_in → D)
                                       │
                       z ∈ ℝ^{12 × D}   (one token per lead)
                                       │
            ┌──────────── L × encoder block (pre-LN) ───────────────┐
            │  MultiHeadVariateAttention  (attention over 12 leads) │
            │  FeedForward                                          │
            └────────────────────────────────────────────────────────┘
                                       │
                       ┌───────────────┴────────────────┐
                       │                                │
        Linear(D → T_out)                    mean over N leads → Linear(D → 5)
                       │                                │
            forecast ∈ ℝ^{12 × 500}              logits ∈ ℝ^5
            (next 5s of every lead)         (multi-label cardiac events)
```

Joint loss:

```
L = α · MSE(forecast, y_wave) + β · BCE_with_logits(logits, y_lab)
```

## Targets

Multi-label, mapped from PTB-XL's SCP-ECG statements at likelihood ≥ 50:

| Class | Source codes |
|---|---|
| `normal` | `NORM` |
| `af` | `AFIB`, `AFLT` |
| `stemi` | `AMI`, `IMI`, `ASMI`, `ALMI`, `ILMI`, `INJAS`, `INJAL`, `INJIN`, `INJIL`, `INJLA` |
| `arrhythmia` | `SBRAD`, `STACH`, `SARRH`, `PAC`, `PVC`, `BIGU`, `TRIGU`, `PACE` |
| `conduction` | `1AVB`, `2AVB`, `3AVB`, `CLBBB`, `CRBBB`, `IVCD`, `LAFB`, `LPFB`, `WPW` |

STEMI is mapped to the specific MI / injury codes rather than the broad STTC
superclass — that subset is the clinically actionable, time-critical signal
the forecasting framing is supposed to catch.

## Results

To be filled after the full training sweep finishes. Numbers below are
placeholders.

| Model | Macro AUROC | F1 macro | Sens (STEMI) | Spec (STEMI) | Params | Size FP32 / INT8 | ONNX p50 | Core ML p50 | TFLite p50 |
|---|---|---|---|---|---|---|---|---|---|
| LSTM | — | — | — | — | — | — / — | — | — | — |
| Bi-LSTM | — | — | — | — | — | — / — | — | — | — |
| 1D-CNN | — | — | — | — | — | — / — | — | — | — |
| Transformer-T | — | — | — | — | — | — / — | — | — | — |
| **iTransformer** | — | — | — | — | — | — / — | — | — | — |

### iTransformer size ablation

Decision rule, declared in advance to avoid post-hoc reasoning:

- If **Large** val AUROC > Medium by ≥ 0.01 *and* its overfitting gap ≤ Medium + 0.02 → ship Large.
- Else if **Medium** ≥ Small by ≥ 0.01 → ship Medium.
- Else ship **Small** (smallest is best for on-device).

| Variant | D | L | H | Params | Train AUROC | Val AUROC | Gap | INT8 size |
|---|---|---|---|---|---|---|---|---|
| Small | 64 | 2 | 4 | — | — | — | — | — |
| Medium | 128 | 4 | 4 | — | — | — | — | — |
| Large | 256 | 6 | 8 | — | — | — | — | — |

## Repository layout

```
configs/         per-model YAML, inherits from base.yaml
smartecg/
  data/          download, labels, preprocessing, splits, dataset
  models/        itransformer, lstm, bilstm, cnn1d, transformer_t (hand-written)
  training/      losses, metrics, AMP loop, train.py entry
  interpretability/  variate attention, SHAP, IG, streamlit dashboard
  deployment/    PTQ, ONNX/Core ML/TFLite export, latency benchmark
notebooks/       PTB-XL exploration, signal QA, results
scripts/         train_all.sh, export_all.sh, benchmark_all.sh
tests/           pytest — labels, preprocessing, model shapes, ONNX parity
```

## Quickstart

```bash
# install
pip install -e .[deploy,dev]

# secrets — fill in WANDB_API_KEY; .env is gitignored
cp .env.example .env

# fetch PTB-XL (~2 GB)
python -m smartecg.data.download

# tests
pytest -q

# smoke run on 100 records
python -m smartecg.training.train --config configs/itransformer.yaml \
    --max-records 100 --epochs 2

# full sweep (3 seeds × 5 architectures, plus size ablation)
bash scripts/train_all.sh

# quantize + export every model to every runtime
bash scripts/export_all.sh

# latency / size benchmark
bash scripts/benchmark_all.sh

# interpretability dashboard
streamlit run smartecg/interpretability/dashboard.py -- \
    --predictions runs/itransformer/test_predictions.npz \
    --metadata data/raw/ptbxl/ptbxl_database.csv
```

## Why these design choices

- **Variates as tokens.** ECG leads observe the same cardiac event from
  different spatial projections; cross-lead relationships carry diagnostic
  structure. Attention across leads makes that structure first-class.
- **Joint forecast + classify.** Forecasting is a real signal — if the
  forecast representation actually carries diagnostic content, the
  classification head should learn from it cheaply. The joint loss is also a
  natural regularizer.
- **Hand-written attention.** The math is in the source, not behind a
  framework wrapper. The interpretability code reads attention weights
  directly off the module.
- **100Hz primary.** Closer to the bandwidth of consumer wearable ECG
  hardware than 500Hz, and produces models small enough for an honest
  on-device benchmark. 500Hz is run as an ablation.
- **ONNX as primary export.** Runtime-agnostic; ports cleanly to almost any
  edge stack. Core ML and TFLite are run alongside to show the path is not
  tied to a single mobile platform.
- **Interpretability up front, not bolted on.** Three independent views
  (attention, SHAP, IG) so they can cross-validate; demographic breakdowns
  surface bias before it ships.

## References

- Liu, Y. et al. *iTransformer: Inverted Transformers Are Effective for Time
  Series Forecasting.* ICLR 2024.
- Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography
  dataset.* Scientific Data 7, 154 (2020).
- Strodthoff, N. et al. *Deep Learning for ECG Analysis: Benchmarks and
  Insights from PTB-XL.* IEEE Journal of Biomedical and Health Informatics,
  2021.
