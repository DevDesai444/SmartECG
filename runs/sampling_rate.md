# 100 Hz vs 500 Hz Medium — does 5× the sampling rate buy real AUROC?

Same iTransformer Medium (D=128, L=4, H=4), same canonical α=0.05 / β=2.0, same
3 seeds (42, 1337, 2024), same full PTB-XL — only the input sampling rate
differs. The 500 Hz variant reads PTB-XL's `records500/` instead of
`records100/`. The input window stays at 5 s either way: 500 samples at 100 Hz,
2500 samples at 500 Hz.

| sampling rate | params    | macro AUROC (mean ± std) | ONNX INT8 (MB) | ONNX INT8 p50 on M1 CPU (ms) |
|---|---|---|---|---|
| 100 Hz |   922,617 | 0.7484 ± 0.0065 | 1.01 | 0.221 |
| 500 Hz | 1,436,617 | 0.7553 ± 0.0043 | 1.50 | 0.244 |

## read

500 Hz buys +0.0069 macro AUROC over 100 Hz at the cost of 1.56× params and
1.10× INT8 latency on M1 CPU. The AUROC delta sits just barely above the
larger of the two per-seed stds (0.0065) — call it a marginal real gain rather
than noise, but only by about half a std. The 500 Hz INT8 also took a bigger
quantization hit on parity (logits max-err vs FP32 was 0.089 at 500 Hz vs
0.0084 at 100 Hz), suggesting the larger variate embedding's weight
distribution is less amenable to dynamic INT8 — calibrated static PTQ would
likely help, but that's a Phase 4 deployment-polish item, not a Phase 2
finding.

For the README's headline iTransformer number we keep 100 Hz: the 1.10× CPU
latency is tractable, but the AUROC gain is small relative to the
across-architecture gap (1D-CNN ahead of iTransformer by ~17 pp at 100 Hz —
the much bigger thing to fix), and the quantization-noise asymmetry would
require extra calibration work to ship the 500 Hz model honestly. The 500 Hz
checkpoint stays in `checkpoints/itransformer_500/` and the ONNX exports in
`exports/itransformer_500{,_int8}.onnx` so the comparison is reproducible.
