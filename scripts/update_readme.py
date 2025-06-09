"""Rewrite README results sections with real numbers from runs/*."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def load_run(name):
    npz = REPO / "runs" / name / "test_predictions.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    classes = list(z["classes"])
    per_cls = {c: float(z["per_class"][i]) for i, c in enumerate(classes)}
    # compute STEMI sens/spec from raw predictions
    y_true = z["y_true"]; y_score = z["y_score"]
    stemi_idx = classes.index("stemi")
    pred = (y_score[:, stemi_idx] >= 0.5).astype(int)
    yt = y_true[:, stemi_idx].astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    # macro F1
    return {
        "macro_auroc": float(z["test_macro_auroc"]),
        "macro_f1": float(z["test_macro_f1"]),
        "stemi_sens": sens,
        "stemi_spec": spec,
        "forecast_mse": float(z["test_forecast_mse"]),
        "per_class": per_cls,
    }


def count_params(name):
    import torch
    p = REPO / "checkpoints" / name / "best.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    from smartecg.models import build_model
    m = build_model(ck["cfg"])
    m.load_state_dict(ck["model"])
    return sum(t.numel() for t in m.parameters())


def fmt(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) and not np.isnan(v) else "—"


def fmt_params(p):
    if p is None:
        return "—"
    if p >= 1e6:
        return f"{p/1e6:.2f}M"
    return f"{p/1e3:.0f}K"


def fmt_kb(kb):
    return f"{kb:.0f} KB" if kb < 1024 else f"{kb/1024:.2f} MB"


def main():
    models_main = [
        ("lstm", "LSTM"),
        ("bilstm", "Bi-LSTM"),
        ("cnn1d", "1D-CNN"),
        ("transformer_t", "Transformer-T"),
        ("itransformer", "**iTransformer**"),
    ]
    rows = []
    for name, label in models_main:
        r = load_run(name)
        p = count_params(name)
        if r is None:
            rows.append((label, *(["—"] * 7), fmt_params(p)))
            continue
        # ONNX latency / size — pulled from deployment.json only for the iTransformer
        rows.append((
            label,
            fmt(r["macro_auroc"], 3),
            fmt(r["macro_f1"], 3),
            fmt(r["stemi_sens"], 3),
            fmt(r["stemi_spec"], 3),
            fmt_params(p),
            fmt(r["per_class"].get("af", float("nan")), 3),
            fmt(r["per_class"].get("conduction", float("nan")), 3),
        ))

    # Deployment numbers — itransformer only (the shipped model)
    dep = REPO / "runs" / "deployment.json"
    if dep.exists():
        D = json.loads(dep.read_text())
    else:
        D = {}

    def dep_row(key):
        info = D.get(key, {})
        if info.get("skipped"):
            return ("skipped", "—", "—")
        size = info.get("size_kb")
        return (
            fmt_kb(size) if size else "—",
            f"{info.get('p50_ms', float('nan')):.2f} ms" if "p50_ms" in info else "—",
            f"{info.get('p95_ms', float('nan')):.2f} ms" if "p95_ms" in info else "—",
        )

    onnx_fp32 = dep_row("onnx_fp32")
    onnx_int8 = dep_row("onnx_int8")
    coreml = dep_row("coreml")
    tflite = dep_row("tflite")

    # Build the model comparison table
    main_hdr = (
        "| Model | Macro AUROC | F1 macro | Sens (STEMI) | Spec (STEMI) | "
        "Params | AF AUROC | CD AUROC |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    main_body = "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    )

    # Size ablation
    abl_rows = []
    for variant, label in [("itransformer_s", "Small"),
                           ("itransformer", "Medium (default)"),
                           ("itransformer_l", "Large")]:
        r = load_run(variant)
        p = count_params(variant)
        if r is None:
            abl_rows.append(f"| {label} | — | — | — | {fmt_params(p)} |")
        else:
            abl_rows.append(
                f"| {label} | {fmt(r['macro_auroc'])} | {fmt(r['macro_f1'])} | "
                f"{fmt(r['forecast_mse'])} | {fmt_params(p)} |"
            )
    abl_hdr = "| Variant | Val/Test AUROC | F1 | Forecast MSE | Params |\n|---|---|---|---|---|\n"

    # Deployment table
    dep_table = (
        "| Runtime | Size | p50 latency | p95 latency |\n"
        "|---|---|---|---|\n"
        f"| ONNX FP32 | {onnx_fp32[0]} | {onnx_fp32[1]} | {onnx_fp32[2]} |\n"
        f"| ONNX INT8 | {onnx_int8[0]} | {onnx_int8[1]} | {onnx_int8[2]} |\n"
        f"| Core ML (FP16+INT8 weights) | {coreml[0]} | {coreml[1]} | {coreml[2]} |\n"
        f"| TFLite | {tflite[0]} | {tflite[1]} | {tflite[2]} |\n"
    )

    # Figures
    figs = sorted((REPO / "figures").glob("*.png"))
    fig_md = "\n".join(
        f"![{f.stem}](figures/{f.name})" for f in figs[:8]
    ) if figs else "*Interpretability figures not yet generated.*"

    # Data subset note
    snap = REPO / "runs" / "snapshot_ecg_ids.csv"
    n_records = sum(1 for _ in open(snap)) - 1 if snap.exists() else 0
    subset_note = (
        f"\n> **Compute note.** Results are reported on a {n_records}-record snapshot "
        f"(~{n_records/21799*100:.0f}% of the full PTB-XL 100Hz set), trained on CPU "
        f"with a 1-seed budget per architecture. The full PTB-XL release contains "
        f"21,799 records; the snapshot reflects what was downloaded in the available "
        f"compute window. Numbers will move with full data + multi-seed averaging; "
        f"the ranking trends should be representative.\n"
    )

    text = README.read_text()

    # Replace "## Results" section through to "## Repository layout"
    new_results = (
        "## Results\n\n"
        + subset_note + "\n"
        + "### Cross-architecture comparison (test fold 10)\n\n"
        + main_hdr + main_body + "\n\n"
        + "### iTransformer size ablation\n\n"
        + abl_hdr + "".join(abl_rows) + "\n\n"
        + "### Deployment benchmarks (single-window inference, CPU)\n\n"
        + dep_table + "\n"
        + "### Interpretability figures\n\n"
        + fig_md + "\n\n"
    )

    text = re.sub(
        r"## Results.*?(?=## Repository layout)",
        new_results,
        text,
        count=1,
        flags=re.DOTALL,
    )

    README.write_text(text)
    print("README updated")
    print(f"  records: {n_records}")
    print(f"  figures: {len(figs)}")


if __name__ == "__main__":
    main()
