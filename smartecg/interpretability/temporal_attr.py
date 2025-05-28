"""Integrated Gradients over the input window for each (class, lead) pair."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .attention import LEAD_NAMES


class _ClassLogitWrapper(nn.Module):
    def __init__(self, model, class_idx):
        super().__init__()
        self.model = model
        self.class_idx = class_idx

    def forward(self, x):
        _, logits = self.model(x)
        return logits[:, self.class_idx]


def integrated_gradients(model, x: torch.Tensor, class_idx: int, device, steps: int = 32):
    """Returns IG attribution of shape (B, N, T) for a single class."""
    from captum.attr import IntegratedGradients
    wrapped = _ClassLogitWrapper(model, class_idx).to(device).eval()
    ig = IntegratedGradients(wrapped)
    attr = ig.attribute(x.to(device), baselines=torch.zeros_like(x).to(device),
                        n_steps=steps)
    return attr.detach().cpu().numpy()


def plot_temporal_saliency(x: np.ndarray, attr: np.ndarray, class_name: str, out_path: str):
    """x, attr: (N, T) for a single record. Plot waveform + saliency overlay per lead."""
    import matplotlib.pyplot as plt
    n, t = x.shape
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.0 * n), sharex=True)
    sal = np.abs(attr)
    sal /= (sal.max() + 1e-12)
    for i in range(n):
        ax = axes[i]
        ax.plot(x[i], color="black", linewidth=0.6)
        ax.fill_between(np.arange(t), x[i].min(), x[i].max(),
                        where=sal[i] > 0.2, alpha=0.3, color="red", linewidth=0)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axes[-1].set_xlabel("sample")
    fig.suptitle(f"temporal saliency — class={class_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
