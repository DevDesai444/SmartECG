"""Training entry point.

Usage:
    python -m smartecg.training.train --config configs/itransformer.yaml
    python -m smartecg.training.train --config configs/itransformer.yaml --max-records 100 --epochs 2
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from smartecg.utils.config import load_config
from smartecg.utils.seed import set_seed
from smartecg.data.dataset import PTBXLDataset
from smartecg.data.splits import split_indices
from smartecg.data.labels import build_label_table, filter_to_available
from smartecg.models import build_model
from smartecg.training.loop import train_one_epoch, evaluate


def get_loaders(cfg, max_records=None):
    root = cfg["data"]["root"]
    df = build_label_table(Path(root) / "ptbxl_database.csv",
                           cfg["data"]["label_threshold"])
    n_total = len(df)
    df = filter_to_available(df, root, cfg["data"]["sampling_rate"])
    if len(df) < n_total:
        print(f"[data] using {len(df)}/{n_total} records "
              f"({len(df)/n_total*100:.1f}% on disk)")
    tr_ids = df.loc[df["strat_fold"].isin(cfg["splits"]["train_folds"]), "ecg_id"].to_numpy()
    va_ids = df.loc[df["strat_fold"].isin(cfg["splits"]["val_folds"]), "ecg_id"].to_numpy()
    te_ids = df.loc[df["strat_fold"].isin(cfg["splits"]["test_folds"]), "ecg_id"].to_numpy()

    if max_records is not None:
        rng = np.random.default_rng(cfg["seed"])
        tr_ids = rng.choice(tr_ids, size=min(max_records, len(tr_ids)), replace=False)
        va_ids = rng.choice(va_ids, size=min(max_records // 4, len(va_ids)), replace=False)

    def _mk(ids):
        return PTBXLDataset(
            root=root, sampling_rate=cfg["data"]["sampling_rate"],
            input_seconds=cfg["data"]["input_seconds"],
            forecast_seconds=cfg["data"]["forecast_seconds"],
            label_threshold=cfg["data"]["label_threshold"],
            indices=ids, cache_dir=cfg["data"]["cache"],
        )

    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    tr = DataLoader(_mk(tr_ids), batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    va = DataLoader(_mk(va_ids), batch_size=bs, shuffle=False, num_workers=nw)
    te = DataLoader(_mk(te_ids), batch_size=bs, shuffle=False, num_workers=nw)
    return tr, va, te


def cosine_with_warmup(optim, total_steps, warmup_ratio, min_warmup_steps: int = 50):
    """Cosine schedule with linear warmup. min_warmup_steps prevents zero-warmup
    when total_steps is small (smoke runs), which can cause MPS-side numerical
    instability when AdamW jumps straight to peak LR."""
    warmup = max(min_warmup_steps, int(total_steps * warmup_ratio))
    warmup = min(warmup, total_steps - 1)
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-records", type=int, default=None,
                   help="subsample for smoke runs")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    load_dotenv()  # reads .env (gitignored) for WANDB_API_KEY etc.
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["seed"])
    # Device selection: CUDA → MPS → CPU. We fall back from MPS to CPU when
    # SMARTECG_FORCE_CPU=1 or when the user knows MPS is unstable for this code
    # (torch 2.3 has an attention NaN bug we've verified on Apple Silicon —
    # CPU produces correct results, MPS does not).
    force_cpu = os.environ.get("SMARTECG_FORCE_CPU") == "1"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cfg["data"]["batch_size"] = cfg["data"].get("batch_size_cuda", cfg["data"]["batch_size"])
        cfg["data"]["num_workers"] = cfg["data"].get("num_workers", 4)
    elif torch.backends.mps.is_available() and not force_cpu:
        device = torch.device("mps")
        cfg["data"]["batch_size"] = cfg["data"].get("batch_size_mps", 32)
        cfg["data"]["num_workers"] = 0
    else:
        device = torch.device("cpu")
        cfg["data"]["batch_size"] = cfg["data"].get("batch_size_cpu", 16)
        cfg["data"]["num_workers"] = 0  # 0 is faster on macOS in our caching setup
    print(f"device: {device}, batch_size: {cfg['data']['batch_size']}")

    tr, va, te = get_loaders(cfg, max_records=args.max_records)
    model = build_model(cfg).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        eps=1e-7,  # default 1e-8 underflows on MPS at fp32 in some configs
    )
    total_steps = max(1, len(tr) * cfg["train"]["epochs"])
    sched = cosine_with_warmup(optim, total_steps, cfg["train"]["warmup_ratio"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    # Disable AMP on non-CUDA — MPS has limited AMP coverage in torch 2.3
    if device.type != "cuda":
        cfg["train"]["amp"] = False

    wandb_run = None
    if cfg["wandb"]["enabled"] and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", cfg["wandb"]["project"]),
            entity=os.environ.get("WANDB_ENTITY") or None,
            config=cfg,
            name=f"{cfg['model']['name']}_{args.tag}" if args.tag else None,
        )

    best_auroc = -1.0
    bad_epochs = 0
    step = 0
    # Output dir includes the config stem so size-ablation variants don't
    # collide with the default model run.
    cfg_stem = Path(args.config).stem
    ckpt_dir = Path("checkpoints") / cfg_stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, step = train_one_epoch(
            model, tr, optim, sched, device, cfg, scaler, step, wandb_run
        )
        cls, fct = evaluate(model, va, device, cfg["classes"])
        macro_auroc = cls["macro"]["auroc"]
        print(f"epoch {epoch} train_loss={train_loss:.4f} val_auroc={macro_auroc:.4f}")
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "val/macro_auroc": macro_auroc,
                "val/macro_f1": cls["macro"]["f1"],
                "val/forecast_mse": fct["mse"],
                **{f"val/{k}_auroc": v["auroc"] for k, v in cls.items() if k != "macro"},
            })
        if macro_auroc > best_auroc:
            best_auroc = macro_auroc
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "cfg": cfg},
                       ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["train"]["early_stop_patience"]:
                print("early stop")
                break

    # final test eval with best ckpt + persist predictions for the dashboard
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    cls, fct = evaluate(model, te, device, cfg["classes"])
    print("TEST", cls["macro"], "mse", fct["mse"])

    # collect test predictions for the dashboard
    import numpy as np
    model.eval()
    yt, ys, eids = [], [], []
    with torch.no_grad():
        for x_in, _yw, y_lab, meta in te:
            x_in = x_in.to(device)
            _f, logits = model(x_in)
            yt.append(y_lab.numpy())
            ys.append(torch.sigmoid(logits).cpu().numpy())
            # meta is a dict of lists when batched by the default collate
            if isinstance(meta, dict) and "ecg_id" in meta:
                eids.append(np.asarray(meta["ecg_id"]))
    if yt:
        runs_dir = Path("runs") / cfg_stem
        runs_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            runs_dir / "test_predictions.npz",
            y_true=np.concatenate(yt, axis=0),
            y_score=np.concatenate(ys, axis=0),
            ecg_id=np.concatenate(eids, axis=0) if eids else np.array([]),
            test_macro_auroc=cls["macro"]["auroc"],
            test_macro_f1=cls["macro"]["f1"],
            test_forecast_mse=fct["mse"],
            per_class=np.array([cls[c]["auroc"] for c in cfg["classes"]]),
            classes=np.array(cfg["classes"]),
        )
    if wandb_run is not None:
        wandb_run.log({
            "test/macro_auroc": cls["macro"]["auroc"],
            "test/macro_f1": cls["macro"]["f1"],
            "test/forecast_mse": fct["mse"],
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
