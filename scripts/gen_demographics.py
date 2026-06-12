"""Render per-class AUROC and F1 broken down by age, sex, and recording site.

Reads:
  runs/itransformer/seed_42/test_predictions.npz   (y_true, y_score, ecg_id, classes)
  data/raw/ptbxl/ptbxl_database.csv                (age, sex, site)

Writes:
  figures/demographics.png

Cells with fewer than MIN_POS=30 positives for a class are rendered as "n<30".
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "runs/itransformer/seed_42/test_predictions.npz"
META = REPO / "data/raw/ptbxl/ptbxl_database.csv"
OUT = REPO / "figures/demographics.png"

MIN_POS = 30


def age_bucket(a):
    if pd.isna(a):
        return "unknown"
    if a < 40:
        return "<40"
    if a < 60:
        return "40-60"
    if a < 80:
        return "60-80"
    return ">=80"


def sex_bucket(s):
    if pd.isna(s):
        return "unknown"
    return {0: "M", 1: "F"}.get(int(s), "unknown")


def metric_cell(y_true_col, y_score_col):
    n_pos = int(y_true_col.sum())
    if n_pos < MIN_POS:
        return None, None, n_pos
    auc = roc_auc_score(y_true_col, y_score_col)
    f1 = f1_score(y_true_col, (y_score_col >= 0.5).astype(int), zero_division=0)
    return auc, f1, n_pos


def render_panel(ax, mat_auc, mat_f1, row_labels, col_labels, title):
    n_rows, n_cols = mat_auc.shape
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(n_rows):
        for j in range(n_cols):
            a, f = mat_auc[i, j], mat_f1[i, j]
            if np.isnan(a):
                txt = "n<30"
                color = "#bbbbbb"
            else:
                txt = f"AUROC {a:.3f}\nF1    {f:.3f}"
                color = plt.cm.viridis(min(max((a - 0.5) / 0.5, 0), 1))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(a) and a > 0.7) else "black")
    ax.set_xticks(np.arange(n_cols + 1), minor=True)
    ax.set_yticks(np.arange(n_rows + 1), minor=True)
    ax.tick_params(which="minor", length=0)


def build_matrix(y_true, y_score, classes, groups, group_keys):
    n_groups = len(group_keys)
    n_classes = len(classes)
    mat_auc = np.full((n_groups, n_classes), np.nan)
    mat_f1 = np.full((n_groups, n_classes), np.nan)
    failed = []
    for i, g in enumerate(group_keys):
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        for c, cls in enumerate(classes):
            auc, f1, n_pos = metric_cell(y_true[idx, c], y_score[idx, c])
            if auc is None:
                failed.append((g, cls, n_pos))
                continue
            mat_auc[i, c] = auc
            mat_f1[i, c] = f1
    return mat_auc, mat_f1, failed


def main():
    data = np.load(PRED, allow_pickle=True)
    y_true = data["y_true"].astype(int)
    y_score = data["y_score"]
    ecg_id = data["ecg_id"].astype(int)
    classes = [str(c) for c in data["classes"]]

    meta = pd.read_csv(META, index_col="ecg_id")
    meta = meta.loc[ecg_id]
    age = meta["age"].apply(age_bucket).values
    sex = meta["sex"].apply(sex_bucket).values
    site = meta["site"].fillna(-1).astype(int).astype(str).values

    age_keys = ["<40", "40-60", "60-80", ">=80"]
    sex_keys = ["M", "F"]
    site_counts = pd.Series(site).value_counts()
    top_sites = site_counts.head(6).index.tolist()
    site_keys = top_sites

    age_auc, age_f1, age_fail = build_matrix(y_true, y_score, classes, age, age_keys)
    sex_auc, sex_f1, sex_fail = build_matrix(y_true, y_score, classes, sex, sex_keys)
    site_auc, site_f1, site_fail = build_matrix(y_true, y_score, classes, site, site_keys)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11),
                             gridspec_kw={"height_ratios": [4, 2, 6]})
    render_panel(axes[0], age_auc, age_f1, age_keys, classes,
                 "Age bucket x class")
    render_panel(axes[1], sex_auc, sex_f1, sex_keys, classes,
                 "Sex x class")
    render_panel(axes[2], site_auc, site_f1, [f"site {s}" for s in site_keys], classes,
                 f"Top-{len(site_keys)} recording sites x class")
    fig.suptitle("iTransformer Medium (seed 42) -- demographics breakdown",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_fail = len(age_fail) + len(sex_fail) + len(site_fail)
    print(f"wrote {OUT}")
    print(f"cells below n>={MIN_POS} positives: {n_fail}")
    for g, cls, n in age_fail:
        print(f"  age={g} class={cls} n_pos={n}")
    for g, cls, n in sex_fail:
        print(f"  sex={g} class={cls} n_pos={n}")
    for g, cls, n in site_fail:
        print(f"  site={g} class={cls} n_pos={n}")


if __name__ == "__main__":
    main()
