"""Dual heads shared across all models so the comparison is apples-to-apples.

ForecastHead: project an encoded representation to a (12, T_out) waveform.
ClassifyHead: project a pooled representation to 5 class logits.
"""
import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """Per-variate forecast. Input (B, N, D) → (B, N, T_out)."""

    def __init__(self, d_model: int, t_out: int):
        super().__init__()
        self.proj = nn.Linear(d_model, t_out)

    def forward(self, x):  # x: (B, N, D)
        return self.proj(x)


class ForecastHeadFromPool(nn.Module):
    """For models that produce a (B, D) representation rather than (B, N, D).
    Predicts a flat (B, N*T_out) then reshapes."""

    def __init__(self, d_model: int, n_leads: int, t_out: int):
        super().__init__()
        self.n_leads = n_leads
        self.t_out = t_out
        self.proj = nn.Linear(d_model, n_leads * t_out)

    def forward(self, h):  # h: (B, D)
        b = h.size(0)
        return self.proj(h).view(b, self.n_leads, self.t_out)


class ClassifyHead(nn.Module):
    """Project a (B, D) representation to (B, n_classes) logits."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, h):  # h: (B, D)
        return self.proj(h)
