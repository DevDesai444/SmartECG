"""End-to-end sweep runner — trains every architecture, persists per-run
results to runs/{model}/results.json, then writes a summary at runs/summary.json.
Uses 1 seed by default to fit a single-session compute budget; seeds can be
overridden by SEEDS env var.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO)
    dt = time.time() - t0
    print(f"<<< exit={p.returncode} in {dt:.1f}s", flush=True)
    return p.returncode


def train(cfg, tag, epochs):
    env = os.environ.copy()
    env.setdefault("SMARTECG_FORCE_CPU", "1")
    env.setdefault("WANDB_DISABLED", "true")
    cmd = [
        "python3", "-u", "-m", "smartecg.training.train",
        "--config", cfg, "--epochs", str(epochs), "--tag", tag,
    ]
    p = subprocess.run(cmd, cwd=REPO, env=env)
    return p.returncode


def collect_results():
    out = {}
    runs = REPO / "runs"
    for run_dir in sorted(runs.glob("*")):
        if not run_dir.is_dir():
            continue
        npz = run_dir / "test_predictions.npz"
        if not npz.exists():
            continue
        import numpy as np
        z = np.load(npz, allow_pickle=True)
        out[run_dir.name] = {
            "test_macro_auroc": float(z["test_macro_auroc"]),
            "test_macro_f1": float(z["test_macro_f1"]),
            "test_forecast_mse": float(z["test_forecast_mse"]),
            "per_class": {
                c: float(z["per_class"][i])
                for i, c in enumerate(z["classes"])
            },
        }
    return out


def main():
    epochs = int(os.environ.get("EPOCHS", "8"))
    size_epochs = int(os.environ.get("SIZE_EPOCHS", "6"))
    plan = [
        # main cross-architecture sweep
        ("configs/itransformer.yaml", "default", epochs),
        ("configs/lstm.yaml", "default", epochs),
        ("configs/bilstm.yaml", "default", epochs),
        ("configs/cnn1d.yaml", "default", epochs),
        ("configs/transformer_t.yaml", "default", epochs),
        # iTransformer size ablation (S/M/L). _m is the default; we reuse it.
        ("configs/itransformer_s.yaml", "ablation_s", size_epochs),
        ("configs/itransformer_l.yaml", "ablation_l", size_epochs),
    ]
    skip_done = os.environ.get("SKIP_DONE", "1") == "1"

    summary = {}
    for cfg, tag, ep in plan:
        name = Path(cfg).stem
        if skip_done and (REPO / "runs" / name / "test_predictions.npz").exists():
            print(f"\n[skip] {name} already has results")
            results = collect_results()
            if name in results:
                summary[name] = {"exit": 0, **results[name]}
                print(f"   AUROC = {results[name]['test_macro_auroc']:.4f}")
            continue
        rc = train(cfg, tag, ep)
        summary[name] = {"exit": rc}
        results = collect_results()
        if name in results:
            summary[name].update(results[name])
            auroc = results[name]["test_macro_auroc"]
            print(f"\n======== {name}  AUROC={auroc:.4f} ========\n")

    out = REPO / "runs" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {out}\n")
    for name, r in summary.items():
        a = r.get("test_macro_auroc", "n/a")
        print(f"  {name:20s}  AUROC = {a}")


if __name__ == "__main__":
    main()
