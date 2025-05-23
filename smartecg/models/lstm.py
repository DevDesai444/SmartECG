import torch
import torch.nn as nn

from .heads import ForecastHeadFromPool, ClassifyHead


class LSTMEncoderModel(nn.Module):
    """2-layer LSTM over time, leads as channels.

    Input (B, N, T_in) → permute → (B, T_in, N) → LSTM → take final hidden →
    forecast head emits (B, N, T_out); classify head emits (B, n_classes).
    """

    def __init__(self, n_leads=12, t_in=500, t_out=500, n_classes=5,
                 hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_leads, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.forecast = ForecastHeadFromPool(hidden, n_leads, t_out)
        self.classify = ClassifyHead(hidden, n_classes)

    def forward(self, x):  # (B, N, T_in)
        x = x.transpose(1, 2)              # (B, T_in, N)
        out, (h, _) = self.lstm(x)         # h: (L, B, H)
        h_last = h[-1]                     # (B, H)
        return self.forecast(h_last), self.classify(h_last)
