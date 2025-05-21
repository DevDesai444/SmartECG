"""Standard ECG preprocessing.

Bandpass 0.5–40Hz removes baseline wander and HF noise without distorting QRS.
Zero-phase filtering avoids morphology shifts important for ST-segment work.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt


def _butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype="band")


def bandpass(x: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """x: (T, N) or (N, T). Returns same shape, filtered along time axis."""
    b, a = _butter_bandpass(low, high, fs)
    # we operate along time → assume the longer axis is time
    if x.ndim != 2:
        raise ValueError(f"expected 2D, got {x.shape}")
    time_axis = 0 if x.shape[0] > x.shape[1] else 1
    return filtfilt(b, a, x, axis=time_axis).astype(np.float32)


def znorm_per_lead(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-lead z-score. x shape (N, T) — leads as rows."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return ((x - mu) / (sd + eps)).astype(np.float32)


def preprocess_record(x: np.ndarray, fs: int) -> np.ndarray:
    """End-to-end preprocessing → (N=12, T) float32, bandpassed + znormed."""
    # wfdb returns (T, N). Get to (N, T).
    if x.shape[0] > x.shape[1]:
        x = x.T
    x = bandpass(x.T, fs).T  # bandpass expects (T, N), then back
    x = znorm_per_lead(x)
    return x
