"""Evaluate a saved best.pt on the test fold and persist test_predictions.npz.

Used when a training run is killed before completing its built-in final eval.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from smartecg.models import build_model
from smartecg.data.dataset import PTBXLDataset
from smartecg.data.labels import build_label_table, filter_to_available
from smartecg.training.loop import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="model name (e.g. lstm)")
    args = p.parse_args()

    ckpt_path = Path("checkpoints") / args.name / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = cfg["data"]["root"]
    df = build_label_table(Path(root) / "ptbxl_database.csv",
                           cfg["data"]["label_threshold"])
    df = filter_to_available(df, root, cfg["data"]["sampling_rate"])
    test_ids = df.loc[df["strat_fold"].isin(cfg["splits"]["test_folds"]),
                      "ecg_id"].to_numpy()
    ds = PTBXLDataset(root=root, sampling_rate=cfg["data"]["sampling_rate"],
                      indices=test_ids, cache_dir=cfg["data"]["cache"])
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    cls, fct = evaluate(model, ld, device, cfg["classes"])
    print(f"{args.name}: macro_auroc={cls['macro']['auroc']:.4f} "
          f"f1={cls['macro']['f1']:.4f} mse={fct['mse']:.4f}")

    # collect raw predictions
    yt, ys, eids = [], [], []
    with torch.no_grad():
        for x_in, _yw, y_lab, meta in ld:
            x_in = x_in.to(device)
            _f, logits = model(x_in)
            yt.append(y_lab.numpy())
            ys.append(torch.sigmoid(logits).cpu().numpy())
            if isinstance(meta, dict) and "ecg_id" in meta:
                eids.append(np.asarray(meta["ecg_id"]))
    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "test_predictions.npz",
        y_true=np.concatenate(yt, axis=0),
        y_score=np.concatenate(ys, axis=0),
        ecg_id=np.concatenate(eids, axis=0) if eids else np.array([]),
        test_macro_auroc=cls["macro"]["auroc"],
        test_macro_f1=cls["macro"]["f1"],
        test_forecast_mse=fct["mse"],
        per_class=np.array([cls[c]["auroc"] for c in cfg["classes"]]),
        classes=np.array(cfg["classes"]),
    )
    print(f"wrote {out_dir / 'test_predictions.npz'}")


if __name__ == "__main__":
    main()
