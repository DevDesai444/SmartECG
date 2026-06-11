# iTransformer size ablation, canonical α=0.05 / β=2.0

Three sizes, three seeds each (42, 1337, 2024), full PTB-XL @ 100 Hz, joint loss
weight α=0.05 / β=2.0 — the values that recover lead-specific attention in the
Medium. The point of this phase is to pick a shipping variant from the
mechanical rule in ROADMAP §0, on equal footing now that S and L share Medium's
canonical loss balance instead of inheriting the broken α=β=1.0 from base.yaml.

| variant | d_model | layers | params | test macro AUROC (mean ± std) | train macro AUROC (mean) | gap (train − test) | ONNX INT8 (MB) |
|---|---|---|---|---|---|---|---|
| Small  (S) |  64 | 2 |   164,985 | 0.7338 ± 0.0051 | N/A | N/A | — |
| Medium (M) | 128 | 4 |   922,617 | 0.7484 ± 0.0065 | N/A | N/A | 1.01 |
| Large  (L) | 256 | 6 | 4,997,113 | 0.7579 ± 0.0036 | N/A | N/A | — |

The Medium row reuses Phase 1's `runs/itransformer/seeds.json` and
`exports/itransformer_int8.onnx` unchanged — same canonical α/β, so it's the
correct reference for the size comparison.

## selection rule (locked, applied mechanically)

From ROADMAP §0:

```
if val_AUROC(L) > val_AUROC(M) + 0.01 and gap(L) ≤ gap(M) + 0.02:
    ship Large
elif val_AUROC(M) ≥ val_AUROC(S) + 0.01:
    ship Medium
else:
    ship Small
```

Substituting from the table (the aggregator persists the held-out fold 10 macro
as `macro_auroc` in seeds.json, which is what this rule reads — the rule's
labelling as "val" is from the ROADMAP, the numerical source is the held-out
fold for all three):

- `AUROC(L) − AUROC(M) = +0.0095`  (not > 0.01 — the Large branch short-circuits here)
- `AUROC(M) − AUROC(S) = +0.0146`  (≥ 0.01 — the Medium branch fires)

Decision: **ship Medium**.

The gap check (`gap(L) ≤ gap(M) + 0.02`) is never reached because the Large
branch fails on the AUROC delta first. Large is +0.0095 over Medium with tighter
std (0.0036 vs 0.0065), which is suggestive but does not clear the locked
+0.01 threshold. The rule is designed to refuse "looks bigger maybe a bit
better" — Phase 1's Medium had a +0.01 margin over Small under the broken loss
balance, and re-cleaning at canonical α/β reproduces the same margin
(+0.0146). The Large delta over Medium does not.

## method note on train AUROC

The selection rule's `gap = train_AUROC − val_AUROC` term needs per-seed train
macro AUROC, which `seeds.json` does not carry (the aggregator only persists
held-out test macros) and `smartecg/training/loop.py` does not log to disk per
epoch either. The two rescue paths Plan 03 documented are:

- (a) re-run inference on the train fold for each `best.pt` checkpoint.
- (b) parse per-epoch train logs.

Path (b) is unavailable — train metrics are only logged to W&B, never to disk.
Path (a) for S and L is feasible, but the Phase 1 Medium per-seed checkpoints
(`checkpoints/itransformer/seed_*/best.pt`) were not preserved locally — only a
single `checkpoints/itransformer/best.pt` remains from the pre-3-seed era. A
gap column populated for S and L but not for M would be asymmetric and could
not be used to apply the rule's gap branch.

Plan 03 documents the legitimate fallback for exactly this case ("(ii) accept
the rule as val-only and note that gap is N/A — the Medium-vs-Large branch
still resolves because the Medium baseline is 0.7484 and we'd need L > 0.7584
to even trigger the gap check"). The actual numbers are L = 0.7579, which is
+0.0005 below the +0.01 threshold — the val-only rule resolves cleanly without
the gap term ever being consulted, so the fallback is the documented choice.

This is recorded so Phase 4's README hygiene pass can decide whether to
re-aggregate the Medium gap before the headline writeup. The seeds-checkpoints
that would let us do it are not on local disk; they would have to be
re-trained or pulled from a fresh Kaggle run.

## why this matters

Phase 1 trained S and L under α=β=1.0 while M was rebalanced. The numbers
weren't on equal footing — the size-ablation row in the README would have read
as if Medium was meaningfully better than Small/Large when really it was just
the only one with a working loss. Re-cleaning at canonical α/β puts all three
on the same loss surface, so the selection rule resolves on signal, not on a
pre-existing handicap.

The shipping variant (Medium) feeds Phase 3 (interpretability figures
regenerated from the seed-42 Medium checkpoint at
`checkpoints/itransformer/seed_42/best.pt` if that gets restored — otherwise
the single `checkpoints/itransformer/best.pt` that's currently on disk) and
Phase 4 (the size-ablation sub-table in the README).
