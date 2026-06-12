"""Generate interpretability figures from the best iTransformer checkpoint.

Writes:
    figures/attention_overall.png            mean attention heatmap
    figures/attention_{class}.png            mean attention per positive class
    figures/shap_lead_importance.png         per-lead |SHAP| per class
    figures/saliency_{class}_{eid}.png       IG temporal saliency, one per class

With --seeds, all maps are averaged pixel-wise across per-seed Medium checkpoints
under --checkpoint-dir/seed_<n>/best.pt. Without --seeds, falls back to the
canonical single checkpoint at checkpoints/itransformer/best.pt.
"""
from __future__ import annotations
import argparse
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
    collect_attention, per_class_attention, plot_attention_heatmap,
    collect_attention_multi, average_per_class_attention, LEAD_NAMES,
)
from smartecg.interpretability.shap_explain import (
    per_lead_shap, plot_lead_importance, per_lead_shap_multi,
)
from smartecg.interpretability.temporal_attr import (
    integrated_gradients, plot_temporal_saliency,
    integrated_gradients_multi, select_top_examples_per_class,
)


REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures"


def _stratified_val_indices(df_val, classes, n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    chosen = set()
    for c in classes:
        pool = df_val[df_val[c] == 1]
        if len(pool) == 0:
            continue
        take = min(n_per_class, len(pool))
        ids = pool.sample(n=take, random_state=int(rng.integers(2**31))).ecg_id.tolist()
        chosen.update(ids)
    return sorted(chosen)


def _build_fixed_tensor(ds, ecg_ids):
    """Pull a fixed set of records as a tensor (M, 12, T)."""
    id_to_idx = {int(r["ecg_id"]): i for i, r in ds.df.iterrows()}
    xs = []
    for eid in ecg_ids:
        if eid not in id_to_idx:
            continue
        x_in, _yw, _yl, _meta = ds[id_to_idx[eid]]
        xs.append(x_in)
    return torch.stack(xs, dim=0)


def _suffix(title, multi_seed_label):
    if multi_seed_label:
        return f"{title} — averaged across {multi_seed_label}"
    return title


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default=None, help="comma-separated, e.g. 42,1337,2024")
    p.add_argument("--checkpoint-dir", default="checkpoints/itransformer_phase1",
                   help="parent dir; per-seed files at <dir>/seed_<n>/best.pt")
    p.add_argument("--shap-background", type=int, default=256,
                   help="stratified val records for SHAP background")
    p.add_argument("--shap-samples", type=int, default=64,
                   help="stratified val records for SHAP samples")
    args = p.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    if args.seeds is None:
        # backward-compat single-checkpoint flow
        ckpt = torch.load(REPO / "checkpoints" / "itransformer" / "best.pt",
                          map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        classes = cfg["classes"]
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _run_single(model, cfg, classes, device, args)
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
        ckpt_paths = [str(REPO / args.checkpoint_dir / f"seed_{s}" / "best.pt")
                      for s in seeds]
        for cp in ckpt_paths:
            assert Path(cp).is_file(), f"missing checkpoint: {cp}"
        cfg = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)["cfg"]
        classes = cfg["classes"]
        label = f"{len(seeds)} seeds ({', '.join(str(s) for s in seeds)})"
        _run_multi(ckpt_paths, cfg, classes, device, args, label)

    # quick manifest
    manifest = sorted([str(p.relative_to(REPO)) for p in FIG.glob("*.png")])
    (REPO / "runs" / "interpretability.json").write_text(
        json.dumps({"figures": manifest, "seeds": args.seeds}, indent=2)
    )
    print(f"\nwrote {len(manifest)} figures to {FIG}/")


def _build_loader(cfg, fold_key, batch_size=16, indices_override=None):
    root = cfg["data"]["root"]
    df = build_label_table(Path(root) / "ptbxl_database.csv",
                           cfg["data"]["label_threshold"])
    df = filter_to_available(df, root, cfg["data"]["sampling_rate"])
    if indices_override is not None:
        ids = indices_override
    else:
        ids = df.loc[df["strat_fold"].isin(cfg["splits"][fold_key]),
                     "ecg_id"].to_numpy()
    ds = PTBXLDataset(root=root, sampling_rate=cfg["data"]["sampling_rate"],
                      indices=ids, cache_dir=cfg["data"]["cache"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return ds, loader, df


def _run_single(model, cfg, classes, device, args):
    ds, loader, _ = _build_loader(cfg, "test_folds")
    print("collecting attention …")
    A, Y = collect_attention(model, loader, device, layer=-1, max_batches=16)
    print(f"  collected {len(A)} samples")
    per_class = per_class_attention(A, Y, classes)
    overall = A.mean(axis=(0, 1))
    plot_attention_heatmap(overall, _suffix("mean variate attention (overall)", None),
                           str(FIG / "attention_overall.png"))
    for c in classes:
        plot_attention_heatmap(per_class[c],
                               _suffix(f"mean variate attention — positive {c}", None),
                               str(FIG / f"attention_{c}.png"))

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

    print("computing IG saliency …")
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
            attr = integrated_gradients(model, x_i.unsqueeze(0), ci, device, steps=32)[0]
            plot_temporal_saliency(x_i.numpy(), attr, cname,
                                   str(FIG / f"saliency_{cname}_{eid}.png"))
            print(f"  saliency_{cname}_{eid}.png done")
        except Exception as e:
            print(f"  IG for {cname} skipped: {e}")


def _run_multi(ckpt_paths, cfg, classes, device, args, label):
    print(f"multi-seed flow over {len(ckpt_paths)} checkpoints")
    # build the test-fold loader for attention + saliency-example selection
    ds_test, loader_test, _ = _build_loader(cfg, "test_folds", batch_size=16)

    # --- 1) Variate attention ---
    print("collecting attention per seed …")
    per_seed = collect_attention_multi(ckpt_paths, build_model, loader_test, device,
                                       layer=-1, max_batches=16)
    per_seed_pc = [per_class_attention(A, Y, classes) for (A, Y) in per_seed]
    avg_pc = average_per_class_attention(per_seed_pc)
    overall_stack = np.stack([A.mean(axis=(0, 1)) for (A, _Y) in per_seed], axis=0)
    overall = overall_stack.mean(axis=0)
    plot_attention_heatmap(overall, _suffix("mean variate attention (overall)", label),
                           str(FIG / "attention_overall.png"))
    for c in classes:
        plot_attention_heatmap(avg_pc[c],
                               _suffix(f"mean variate attention — positive {c}", label),
                               str(FIG / f"attention_{c}.png"))

    # --- 2) SHAP, stratified val background + samples held fixed across seeds ---
    print(f"building stratified val background ({args.shap_background} records) "
          f"+ samples ({args.shap_samples}) …")
    root = cfg["data"]["root"]
    df = build_label_table(Path(root) / "ptbxl_database.csv",
                           cfg["data"]["label_threshold"])
    df = filter_to_available(df, root, cfg["data"]["sampling_rate"])
    val_df = df.loc[df["strat_fold"].isin(cfg["splits"]["val_folds"])].reset_index(drop=True)
    # build label columns matching CLASSES names so the stratifier reads cleanly
    for cname in classes:
        val_df[cname] = val_df[f"y_{cname}"].astype(int)
    bg_n_per_class = max(1, args.shap_background // len(classes) + 1)
    sm_n_per_class = max(1, args.shap_samples // len(classes) + 1)
    bg_ids = _stratified_val_indices(val_df, classes, bg_n_per_class, seed=0)[:args.shap_background]
    sm_ids = _stratified_val_indices(val_df, classes, sm_n_per_class, seed=1)
    sm_ids = [i for i in sm_ids if i not in set(bg_ids)][:args.shap_samples]
    # build a val-only dataset to look up by ecg_id
    val_ids = val_df["ecg_id"].to_numpy()
    ds_val = PTBXLDataset(root=root, sampling_rate=cfg["data"]["sampling_rate"],
                          indices=val_ids, cache_dir=cfg["data"]["cache"])
    background = _build_fixed_tensor(ds_val, bg_ids)
    samples = _build_fixed_tensor(ds_val, sm_ids)
    print(f"  background={tuple(background.shape)}, samples={tuple(samples.shape)}")
    try:
        shap_mat = per_lead_shap_multi(ckpt_paths, build_model, background, samples,
                                       device, classes)
        plot_lead_importance(shap_mat, classes, str(FIG / "shap_lead_importance.png"))
        print(f"  shap matrix shape: {shap_mat.shape}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    # --- 3) IG saliency — pick example by seed-averaged prob, then run IG per seed ---
    print("selecting saliency examples by seed-averaged predicted probability …")
    examples = select_top_examples_per_class(ckpt_paths, build_model, loader_test,
                                             device, classes)
    for cname, (x_i, eid) in examples.items():
        ci = classes.index(cname)
        try:
            attr = integrated_gradients_multi(ckpt_paths, build_model,
                                              x_i.unsqueeze(0), ci, device, steps=32)[0]
            plot_temporal_saliency(x_i.numpy(), attr,
                                   _suffix(cname, label),
                                   str(FIG / f"saliency_{cname}_{eid}.png"))
            print(f"  saliency_{cname}_{eid}.png done")
        except Exception as e:
            print(f"  IG for {cname} skipped: {e}")


if __name__ == "__main__":
    main()
