"""Per-class and aggregate evaluation metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


def _safe_auroc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _specificity(y_true, y_pred):
    # spec = TN / (TN + FP)
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    return tn / (tn + fp) if (tn + fp) else float("nan")


def classification_metrics(y_true_lab, y_score_lab, classes, threshold=0.5):
    """y_*: (M, C) numpy arrays. Returns nested dict {class: {metric: val}, ...}."""
    out = {}
    y_pred_lab = (y_score_lab >= threshold).astype(np.float32)
    per_class_auroc = []
    per_class_f1 = []
    for i, name in enumerate(classes):
        auroc = _safe_auroc(y_true_lab[:, i], y_score_lab[:, i])
        f1 = f1_score(y_true_lab[:, i], y_pred_lab[:, i], zero_division=0)
        sens = recall_score(y_true_lab[:, i], y_pred_lab[:, i], zero_division=0)
        spec = _specificity(y_true_lab[:, i], y_pred_lab[:, i])
        prec = precision_score(y_true_lab[:, i], y_pred_lab[:, i], zero_division=0)
        out[name] = {"auroc": auroc, "f1": f1, "sens": sens, "spec": spec, "prec": prec}
        per_class_auroc.append(auroc)
        per_class_f1.append(f1)
    out["macro"] = {
        "auroc": float(np.nanmean(per_class_auroc)),
        "f1": float(np.nanmean(per_class_f1)),
    }
    return out


def forecast_metrics(y_true_wave, y_pred_wave):
    """y_*: (M, N, T) numpy. Returns per-lead MSE and MAE plus overall."""
    err = y_pred_wave - y_true_wave
    mse = (err ** 2).mean(axis=(0, 2))
    mae = np.abs(err).mean(axis=(0, 2))
    return {
        "per_lead_mse": mse.tolist(),
        "per_lead_mae": mae.tolist(),
        "mse": float(mse.mean()),
        "mae": float(mae.mean()),
    }
