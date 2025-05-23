"""Time-axis Transformer baseline, written from scratch.

The point of this baseline is to isolate the *attention axis* against the
iTransformer. Everything else (FFN, dual heads, hyperparam scale) is matched.

Patch the T-length time series into K = T // patch tokens of length `patch`,
apply a learnable per-token linear embedding, add sinusoidal positional
embeddings (written out, not imported), then run a stack of pre-LN encoder
blocks with hand-written multi-head self-attention across the K time tokens.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ForecastHeadFromPool, ClassifyHead


def sinusoidal_positional_embedding(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)        # (S, 1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))                    # (D/2,)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe                                                            # (S, D)


class _MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.h, self.dh).transpose(1, 2)

    def forward(self, x):
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        a = self.drop(a)
        o = torch.matmul(a, v).transpose(1, 2).contiguous()
        b, s, h, dh = o.shape
        return self.w_o(o.view(b, s, h * dh))


class _Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _MHA(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        y = self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.ln2(x))))))
        return x + y


class TransformerTEncoderModel(nn.Module):
    """Patch-token transformer. Variates (leads) go into the embedding;
    attention operates across the K time-patch tokens.
    """

    def __init__(self, n_leads=12, t_in=500, t_out=500, n_classes=5,
                 patch=25, d_model=128, n_heads=4, n_layers=4, d_ff=512,
                 dropout=0.1):
        super().__init__()
        assert t_in % patch == 0, f"t_in {t_in} not divisible by patch {patch}"
        self.patch = patch
        self.k = t_in // patch
        # each token packs `patch` time-steps from all N leads
        self.embed = nn.Linear(n_leads * patch, d_model)
        self.register_buffer(
            "pe", sinusoidal_positional_embedding(self.k, d_model), persistent=False
        )
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.forecast = ForecastHeadFromPool(d_model, n_leads, t_out)
        self.classify = ClassifyHead(d_model, n_classes)

    def forward(self, x):  # (B, N, T)
        b, n, t = x.shape
        # (B, N, K, patch) → (B, K, N*patch)
        x = x.view(b, n, self.k, self.patch).permute(0, 2, 1, 3).contiguous()
        x = x.view(b, self.k, n * self.patch)
        z = self.embed(x) + self.pe.unsqueeze(0)         # (B, K, D)
        for blk in self.blocks:
            z = blk(z)
        z = self.ln_f(z).mean(dim=1)                     # (B, D)
        return self.forecast(z), self.classify(z)
