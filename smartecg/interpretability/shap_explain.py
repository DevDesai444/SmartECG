"""SHAP attributions over the 12 leads.

We wrap the model so SHAP sees only the classification logits — the forecast
head is a regularization signal, not an attribution target.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .attention import LEAD_NAMES


class _ClassifierWrapper(nn.Module):
    """Strip the forecast head — SHAP wants a single output tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, logits = self.model(x)
        return logits


def per_lead_shap(model, background: torch.Tensor, samples: torch.Tensor,
                  device, classes):
    """Returns a (n_classes, n_leads) matrix of mean |SHAP|."""
    import shap
    wrapped = _ClassifierWrapper(model).to(device).eval()
    explainer = shap.GradientExplainer(wrapped, background.to(device))
    shap_values = explainer.shap_values(samples.to(device))   # list per class, each (M, N, T)
    if not isinstance(shap_values, list):
        shap_values = [shap_values[..., c] for c in range(len(classes))]
    out = np.zeros((len(classes), samples.shape[1]))
    for ci, sv in enumerate(shap_values):
        out[ci] = np.abs(sv).mean(axis=(0, 2))                # avg over samples and time
    return out


def per_lead_shap_multi(checkpoints, model_builder, background: torch.Tensor,
                        samples: torch.Tensor, device, classes):
    """Mean per-lead |SHAP| matrix across a list of checkpoints. Inputs held fixed."""
    mats = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = model_builder(ckpt["cfg"]).to(device).eval()
        m.load_state_dict(ckpt["model"])
        mats.append(per_lead_shap(m, background, samples, device, classes))
    return np.stack(mats, axis=0).mean(axis=0)


def plot_lead_importance(shap_mat: np.ndarray, classes, out_path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    n_classes, n_leads = shap_mat.shape
    width = 0.8 / n_classes
    x = np.arange(n_leads)
    for ci, name in enumerate(classes):
        ax.bar(x + ci * width, shap_mat[ci], width, label=name)
    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels(LEAD_NAMES, rotation=45)
    ax.set_ylabel("mean |SHAP|")
    ax.set_title("per-lead importance by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
