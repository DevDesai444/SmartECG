"""Extract and visualize iTransformer variate attention.

Native to the model — no monkey-patching, no post-hoc attribution method.
The attention matrix A ∈ R^{N x N} is *the* mechanism by which the model
relates cross-lead information, so visualizing it answers "which leads
inform the prediction" directly.
"""
from __future__ import annotations
import numpy as np
import torch

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


@torch.no_grad()
def collect_attention(model, loader, device, layer: int = -1, max_batches: int = 16):
    """Run a few batches and return per-sample attention + labels.

    Returns:
        attn (M, H, N, N), labels (M, C)
    """
    model.eval()
    A, Y = [], []
    n = 0
    for x_in, _y_wave, y_lab, _meta in loader:
        x_in = x_in.to(device)
        model(x_in)
        a = model.get_last_attention()[layer]   # (B, H, N, N)
        A.append(a.cpu().numpy())
        Y.append(y_lab.numpy())
        n += 1
        if n >= max_batches:
            break
    return np.concatenate(A, axis=0), np.concatenate(Y, axis=0)


def per_class_attention(attn: np.ndarray, labels: np.ndarray, classes: list[str]):
    """Mean attention over samples positive for each class. attn averaged over heads."""
    attn = attn.mean(axis=1)                    # (M, N, N) — avg over heads
    out = {}
    for ci, name in enumerate(classes):
        mask = labels[:, ci] == 1
        if mask.sum() == 0:
            out[name] = np.zeros_like(attn[0])
            continue
        out[name] = attn[mask].mean(axis=0)
    return out


def plot_attention_heatmap(A: np.ndarray, title: str, out_path: str):
    """Render a single 12x12 attention heatmap."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(A, cmap="viridis", vmin=0)
    ax.set_xticks(range(12)); ax.set_xticklabels(LEAD_NAMES, rotation=45)
    ax.set_yticks(range(12)); ax.set_yticklabels(LEAD_NAMES)
    ax.set_xlabel("attended-to lead (k)")
    ax.set_ylabel("query lead (q)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
