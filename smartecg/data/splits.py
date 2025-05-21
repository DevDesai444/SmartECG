"""PTB-XL official stratified folds: 1–8 train, 9 val, 10 test."""
from __future__ import annotations
import pandas as pd

DEFAULT_TRAIN = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_VAL = [9]
DEFAULT_TEST = [10]


def split_indices(df: pd.DataFrame, train=DEFAULT_TRAIN, val=DEFAULT_VAL, test=DEFAULT_TEST):
    tr = df[df["strat_fold"].isin(train)].index.to_numpy()
    va = df[df["strat_fold"].isin(val)].index.to_numpy()
    te = df[df["strat_fold"].isin(test)].index.to_numpy()
    return tr, va, te
