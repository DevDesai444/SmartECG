"""iTransformer for multivariate ECG, written from scratch.

The single architectural commitment: attention operates over the N=12 leads as
tokens, with the entire T-length time series of each lead embedded into one
D-dim token. The model never attends across time directly — temporal structure
lives inside the per-variate embedding and inside the forecast head.

No use of nn.Transformer, nn.MultiheadAttention, or any external iTransformer
package. Q/K/V projections, multi-head split, softmax attention, and the FFN
block are all explicit so the math is visible end to end.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting", ICLR 2024.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ForecastHead, ClassifyHead


class VariateEmbedding(nn.Module):
    """Per-variate linear projection T_in → D.

    Input (B, N, T_in) → (B, N, D). No positional embedding: the *identity* of
    each variate is what carries meaning here, not its position in a sequence.
    """

    def __init__(self, t_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(t_in, d_model)

    def forward(self, x):  # (B, N, T_in)
        return self.proj(x)  # (B, N, D)


class MultiHeadVariateAttention(nn.Module):
    """Standard scaled dot-product multi-head attention, but over variate tokens.

    We write it by hand so the attention weights are introspectable for the
    interpretability code (self.last_attn) and to avoid the hidden defaults of
    nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # populated on every forward pass; consumed by interpretability hooks
        self.last_attn: torch.Tensor | None = None

    def _split_heads(self, x):  # (B, N, D) → (B, H, N, d_head)
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):  # (B, H, N, d_head) → (B, N, D)
        b, h, n, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * dh)

    def forward(self, x):  # x: (B, N, D)
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        # (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()
        attn = self.drop(attn)

        out = torch.matmul(attn, v)             # (B, H, N, d_head)
        out = self._merge_heads(out)            # (B, N, D)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class EncoderBlock(nn.Module):
    """Pre-LN block: residual around MHA, residual around FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadVariateAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class iTransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class iTransformer(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        t_in: int = 500,
        t_out: int = 500,
        n_classes: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.embed = VariateEmbedding(t_in, d_model)
        self.encoder = iTransformerEncoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.forecast = ForecastHead(d_model, t_out)
        self.classify = ClassifyHead(d_model, n_classes)

    def forward(self, x):  # x: (B, N, T_in)
        z = self.embed(x)            # (B, N, D)
        z = self.encoder(z)          # (B, N, D)
        forecast = self.forecast(z)  # (B, N, T_out)
        pooled = z.mean(dim=1)       # (B, D)
        logits = self.classify(pooled)
        return forecast, logits

    def get_last_attention(self):
        """Stack of per-layer (B, H, N, N) attention from the most recent forward."""
        return [blk.attn.last_attn for blk in self.encoder.blocks]
