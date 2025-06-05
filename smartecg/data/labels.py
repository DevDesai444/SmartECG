"""SCP code → 5-class multi-label mapping for PTB-XL.

Classes:
    0 normal       — NORM
    1 af           — AFIB, AFLT
    2 stemi        — specific MI / injury codes (clinically actionable subset of STTC)
    3 arrhythmia   — non-AF rhythm disturbances
    4 conduction   — AV blocks, BBB, fascicular blocks, WPW
"""
from __future__ import annotations
import ast
import numpy as np
import pandas as pd

CLASSES = ["normal", "af", "stemi", "arrhythmia", "conduction"]
NUM_CLASSES = len(CLASSES)

# code lists — to be validated against scp_statements.csv in notebook 01
NORMAL_CODES = {"NORM"}
AF_CODES = {"AFIB", "AFLT"}
STEMI_CODES = {
    # diagnostic MI superclass codes (PTB-XL diagnostic_class == 'MI')
    "AMI", "IMI", "ASMI", "ALMI", "ILMI", "PMI", "LMI", "IPMI", "IPLMI",
    # subendocardial injury codes — same MI superclass
    "INJAS", "INJAL", "INJIN", "INJIL", "INJLA",
}
ARRHYTHMIA_CODES = {
    "SBRAD", "STACH", "SARRH", "PAC", "PVC", "BIGU", "TRIGU", "PACE",
    "SVTAC", "SVARR", "PSVT",
}
CONDUCTION_CODES = {
    "1AVB", "2AVB", "3AVB",
    "CLBBB", "CRBBB", "ILBBB", "IRBBB",
    "IVCD", "LAFB", "LPFB",
    "WPW",
}

CLASS_CODE_MAP = {
    0: NORMAL_CODES,
    1: AF_CODES,
    2: STEMI_CODES,
    3: ARRHYTHMIA_CODES,
    4: CONDUCTION_CODES,
}

# PTB-XL rhythm and form codes carry binary 0/100 likelihoods — 0.0 means the
# annotator noted the code but didn't assign a confidence, not "absent". So
# rhythm/form codes count as present whenever they appear in the dict.
# Diagnostic codes carry probabilistic likelihoods (e.g., 15, 35, 50, 80, 100)
# and require thresholding.
RHYTHM_CODES = {
    "SR", "AFIB", "STACH", "SARRH", "SBRAD", "PACE", "SVARR", "BIGU",
    "AFLT", "SVTAC", "PSVT", "TRIGU",
}
FORM_CODES = {
    "ABQRS", "PVC", "STD_", "VCLVH", "QWAVE", "LOWT", "NT_", "PAC", "LPR",
    "INVT", "LVOLT", "HVOLT", "TAB_", "STE_", "PRC(S)",
}
NONDIAG_CODES = RHYTHM_CODES | FORM_CODES


def parse_scp(s):
    """PTB-XL stores scp_codes as a python-literal dict string."""
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s:
        return {}
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def codes_to_labels(scp: dict, threshold: float = 50.0) -> np.ndarray:
    """Map a {code: likelihood} dict to a 5-dim binary vector.

    Hybrid rule: diagnostic codes require likelihood >= threshold; rhythm and
    form codes count as present whenever they appear (their 0/100 likelihood
    is a binary annotation marker, not a confidence).
    """
    y = np.zeros(NUM_CLASSES, dtype=np.float32)
    if not scp:
        return y
    active = set()
    for c, lk in scp.items():
        if c in NONDIAG_CODES:
            active.add(c)
        elif lk is None or lk >= threshold:
            active.add(c)
    for ci, codeset in CLASS_CODE_MAP.items():
        if active & codeset:
            y[ci] = 1.0
    return y


def build_label_table(ptbxl_csv_path, threshold=50.0) -> pd.DataFrame:
    """Read ptbxl_database.csv and produce a tidy table with labels + metadata."""
    df = pd.read_csv(ptbxl_csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(parse_scp)
    labels = np.stack([codes_to_labels(d, threshold) for d in df["scp_codes"]])
    for i, name in enumerate(CLASSES):
        df[f"y_{name}"] = labels[:, i]
    keep = [
        "patient_id", "age", "sex", "site", "strat_fold",
        "filename_lr", "filename_hr",
        *[f"y_{n}" for n in CLASSES],
    ]
    return df[keep].reset_index()
