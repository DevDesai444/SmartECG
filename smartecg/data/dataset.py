"""PTB-XL torch Dataset for joint forecast + classify.

Each sample:
    x_in    (12, T_in)   first 5s, bandpassed + znormed
    y_wave  (12, T_out)  next 5s, same preprocessing
    y_lab   (5,)         multi-label binary targets
    meta    dict         age, sex, site_id, ecg_id
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from .labels import CLASSES, build_label_table
from .preprocessing import preprocess_record


class PTBXLDataset(Dataset):
    def __init__(
        self,
        root: str,
        sampling_rate: int = 100,
        input_seconds: float = 5.0,
        forecast_seconds: float = 5.0,
        label_threshold: float = 50.0,
        indices: np.ndarray | None = None,
        cache_dir: str | None = None,
    ):
        self.root = Path(root)
        self.fs = sampling_rate
        self.t_in = int(input_seconds * sampling_rate)
        self.t_out = int(forecast_seconds * sampling_rate)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        meta_path = self.root / "ptbxl_database.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"ptb-xl not found at {self.root}. run `python -m smartecg.data.download`."
            )
        df = build_label_table(meta_path, label_threshold)
        if indices is not None:
            df = df.loc[df["ecg_id"].isin(indices)].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def _record_path(self, row):
        fname = row["filename_lr"] if self.fs == 100 else row["filename_hr"]
        return str(self.root / fname)

    def _load_raw(self, row):
        sig, _ = wfdb.rdsamp(self._record_path(row))
        return sig  # (T, 12) float

    def _load_processed(self, idx):
        row = self.df.iloc[idx]
        ecg_id = int(row["ecg_id"])
        if self.cache_dir is not None:
            cpath = self.cache_dir / f"{ecg_id}_{self.fs}.pt"
            if cpath.exists():
                return torch.load(cpath, weights_only=True), row
        sig = self._load_raw(row)              # (T, 12)
        x = preprocess_record(sig.astype(np.float32), self.fs)  # (12, T)
        x_t = torch.from_numpy(x)
        if self.cache_dir is not None:
            torch.save(x_t, self.cache_dir / f"{ecg_id}_{self.fs}.pt")
        return x_t, row

    def __getitem__(self, idx):
        x, row = self._load_processed(idx)
        # split into input window + forecast target
        x_in = x[:, : self.t_in].contiguous()
        y_wave = x[:, self.t_in : self.t_in + self.t_out].contiguous()
        y_lab = torch.tensor([row[f"y_{c}"] for c in CLASSES], dtype=torch.float32)
        meta = {
            "ecg_id": int(row["ecg_id"]),
            "age": float(row["age"]) if pd.notna(row["age"]) else -1.0,
            "sex": int(row["sex"]) if pd.notna(row["sex"]) else -1,
            "site": int(row["site"]) if pd.notna(row["site"]) else -1,
        }
        return x_in, y_wave, y_lab, meta
