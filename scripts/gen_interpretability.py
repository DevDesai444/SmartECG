"""Generate interpretability figures from the best iTransformer checkpoint.

Writes:
    figures/attention_overall.png            mean attention heatmap
    figures/attention_{class}.png            mean attention per positive class
    figures/shap_lead_importance.png         per-lead |SHAP| per class
    figures/saliency_{class}_{eid}.png       IG temporal saliency, one per class
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from torch.utils.data import DataLoader

from smartecg.models import build_model
from smartecg.utils.config import load_config
from smartecg.data.dataset import PTBXLDataset
from smartecg.data.labels import build_label_table, filter_to_available
from smartecg.interpretability.attention import (
    collect_attention, per_class_attention, plot_attention_heatmap, LEAD_NAMES,
)
from smartecg.interpretability.shap_explain import per_lead_shap, plot_lead_importance
from smartecg.interpretability.temporal_attr import (
    integrated_gradients, plot_temporal_saliency,
)


REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(REPO / "checkpoints" / "itransformer" / "best.pt",
                      map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    classes = cfg["classes"]
    device = torch.device("cpu")  # interpretability runs on CPU

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # build a small loader from the test fold
    root = cfg["data"]["root"]
    df = build_label_table(Path(root) / "ptbxl_database.csv",
                           cfg["data"]["label_threshold"])
    df = filter_to_available(df, root, cfg["data"]["sampling_rate"])
    test_ids = df.loc[df["strat_fold"].isin(cfg["splits"]["test_folds"]),
                      "ecg_id"].to_numpy()
    ds = PTBXLDataset(root=root, sampling_rate=cfg["data"]["sampling_rate"],
                      indices=test_ids, cache_dir=cfg["data"]["cache"])
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    # --- 1) Variate attention heatmaps ---
    print("collecting attention …")
    A, Y = collect_attention(model, loader, device, layer=-1, max_batches=16)
    print(f"  collected {len(A)} samples")
    per_class = per_class_attention(A, Y, classes)
    overall = A.mean(axis=(0, 1))  # avg over samples and heads
    plot_attention_heatmap(overall, "mean variate attention (overall)",
                           str(FIG / "attention_overall.png"))
    for c in classes:
        plot_attention_heatmap(
            per_class[c],
            f"mean variate attention — positive {c}",
            str(FIG / f"attention_{c}.png"),
        )

    # --- 2) SHAP per-lead importance ---
    print("computing SHAP …")
    background_batches, sample_batches = [], []
    for b, (x_in, _yw, _yl, _m) in enumerate(loader):
        if b < 1:
            background_batches.append(x_in)
        elif b < 3:
            sample_batches.append(x_in)
        else:
            break
    background = torch.cat(background_batches, dim=0)[:8]
    samples = torch.cat(sample_batches, dim=0)[:16]
    try:
        shap_mat = per_lead_shap(model, background, samples, device, classes)
        plot_lead_importance(shap_mat, classes, str(FIG / "shap_lead_importance.png"))
        print(f"  shap matrix shape: {shap_mat.shape}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")
        shap_mat = None

    # --- 3) Integrated Gradients temporal saliency ---
    print("computing IG saliency …")
    # find one positive example per class in the test set
    examples = {}
    for x_in, _yw, y_lab, meta in loader:
        for i in range(x_in.size(0)):
            for ci, cname in enumerate(classes):
                if cname not in examples and y_lab[i, ci] == 1.0:
                    examples[cname] = (x_in[i].clone(), int(meta["ecg_id"][i]))
        if len(examples) == len(classes):
            break

    for cname, (x_i, eid) in examples.items():
        ci = classes.index(cname)
        try:
            attr = integrated_gradients(
                model, x_i.unsqueeze(0), ci, device, steps=32,
            )[0]
            plot_temporal_saliency(
                x_i.numpy(), attr, cname,
                str(FIG / f"saliency_{cname}_{eid}.png"),
            )
            print(f"  saliency_{cname}_{eid}.png done")
        except Exception as e:
            print(f"  IG for {cname} skipped: {e}")

    # --- 4) Quick manifest ---
    manifest = sorted([str(p.relative_to(REPO)) for p in FIG.glob("*.png")])
    (REPO / "runs" / "interpretability.json").write_text(
        json.dumps({"figures": manifest}, indent=2)
    )
    print(f"\nwrote {len(manifest)} figures to {FIG}/")


if __name__ == "__main__":
    main()
