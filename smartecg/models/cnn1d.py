import torch
import torch.nn as nn

from .heads import ForecastHeadFromPool, ClassifyHead


class CNN1DEncoderModel(nn.Module):
    """4 Conv1d blocks on (B, N, T). Strided pooling shrinks T; channels grow."""

    def __init__(self, n_leads=12, t_in=500, t_out=500, n_classes=5,
                 channels=(32, 64, 128, 128), kernel=7, dropout=0.1):
        super().__init__()
        layers = []
        in_c = n_leads
        for c in channels:
            layers += [
                nn.Conv1d(in_c, c, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            in_c = c
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        d = channels[-1]
        self.forecast = ForecastHeadFromPool(d, n_leads, t_out)
        self.classify = ClassifyHead(d, n_classes)

    def forward(self, x):                 # (B, N, T)
        h = self.backbone(x)              # (B, C, T')
        h = self.pool(h).squeeze(-1)      # (B, C)
        return self.forecast(h), self.classify(h)
