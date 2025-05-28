"""Streamlit dashboard for per-class metrics × demographics.

Run:
    streamlit run smartecg/interpretability/dashboard.py -- \
        --predictions runs/itransformer/test_predictions.npz \
        --metadata data/raw/ptbxl/ptbxl_database.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import streamlit as st

from smartecg.data.labels import CLASSES
from smartecg.training.metrics import classification_metrics


def _bucket_age(age):
    if pd.isna(age) or age < 0:
        return "unknown"
    if age < 30: return "<30"
    if age < 50: return "30–50"
    if age < 70: return "50–70"
    return "70+"


def _bucket_sex(s):
    return {0: "M", 1: "F"}.get(int(s), "unknown") if pd.notna(s) else "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="npz with y_true (M,C), y_score (M,C), ecg_id (M,)")
    p.add_argument("--metadata", required=True, help="ptbxl_database.csv")
    args = p.parse_args()

    st.title("SmartECG — per-class metrics × demographics")

    data = np.load(args.predictions, allow_pickle=True)
    y_true, y_score, ecg_id = data["y_true"], data["y_score"], data["ecg_id"]

    meta = pd.read_csv(args.metadata, index_col="ecg_id")
    meta = meta.loc[ecg_id.astype(int)]
    meta["age_bucket"] = meta["age"].apply(_bucket_age)
    meta["sex_bucket"] = meta["sex"].apply(_bucket_sex)

    breakdown = st.selectbox("Group by", ["age_bucket", "sex_bucket", "site"])
    rows = []
    for grp, mask in meta.groupby(breakdown).groups.items():
        idx = meta.index.get_indexer(mask)
        idx = idx[idx >= 0]
        if len(idx) < 20:
            continue
        m = classification_metrics(y_true[idx], y_score[idx], CLASSES)
        for c in CLASSES:
            rows.append({"group": grp, "class": c,
                         "auroc": m[c]["auroc"], "f1": m[c]["f1"],
                         "sens": m[c]["sens"], "spec": m[c]["spec"],
                         "n": int(len(idx))})

    df = pd.DataFrame(rows)
    st.dataframe(df)
    st.bar_chart(df.pivot(index="group", columns="class", values="auroc"))


if __name__ == "__main__":
    main()
