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


def integrated_gradients_multi(checkpoints, model_builder, x: torch.Tensor,
                               class_idx: int, device, steps: int = 32):
    """Mean IG attribution (B, N, T) across a list of checkpoints. Input held fixed."""
    attrs = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = model_builder(ckpt["cfg"]).to(device).eval()
        m.load_state_dict(ckpt["model"])
        attrs.append(integrated_gradients(m, x, class_idx, device, steps=steps))
    return np.stack(attrs, axis=0).mean(axis=0)


@torch.no_grad()
def select_top_examples_per_class(checkpoints, model_builder, loader, device, classes):
    """Pick one (x, ecg_id) per class by seed-averaged predicted probability.

    Runs each checkpoint over the loader, averages sigmoid(logits) across seeds,
    then per class picks the highest-prob record with label == 1.
    """
    n_classes = len(classes)
    per_seed_probs = []
    labels_all = None
    xs_all = None
    ids_all = None
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = model_builder(ckpt["cfg"]).to(device).eval()
        m.load_state_dict(ckpt["model"])
        probs, labs, xs, ids = [], [], [], []
        for x_in, _yw, y_lab, meta in loader:
            _, logits = m(x_in.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labs.append(y_lab.numpy())
            if labels_all is None:
                xs.append(x_in.numpy())
                ids.append(meta["ecg_id"].numpy())
        per_seed_probs.append(np.concatenate(probs, axis=0))
        if labels_all is None:
            labels_all = np.concatenate(labs, axis=0)
            xs_all = np.concatenate(xs, axis=0)
            ids_all = np.concatenate(ids, axis=0)
    avg_probs = np.stack(per_seed_probs, axis=0).mean(axis=0)  # (M, C)
    out = {}
    for ci, cname in enumerate(classes):
        mask = labels_all[:, ci] == 1
        if mask.sum() == 0:
            continue
        scored = np.where(mask, avg_probs[:, ci], -np.inf)
        idx = int(np.argmax(scored))
        out[cname] = (torch.from_numpy(xs_all[idx]), int(ids_all[idx]))
    return out


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
