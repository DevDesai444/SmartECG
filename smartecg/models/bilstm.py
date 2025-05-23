import torch
import torch.nn as nn

from .heads import ForecastHeadFromPool, ClassifyHead


class BiLSTMEncoderModel(nn.Module):
    def __init__(self, n_leads=12, t_in=500, t_out=500, n_classes=5,
                 hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_leads, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        d = hidden * 2
        self.forecast = ForecastHeadFromPool(d, n_leads, t_out)
        self.classify = ClassifyHead(d, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)                     # (B, T, 2H)
        # mean-pool across time — symmetric for bidirectional reps
        h = out.mean(dim=1)
        return self.forecast(h), self.classify(h)
