"""Idempotent PTB-XL fetcher.

Usage:
    python -m smartecg.data.download [--root data/raw/ptbxl]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ptbxl")
    args = p.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if (root / "ptbxl_database.csv").exists():
        print(f"ptb-xl already present at {root}")
        return 0

    # wget is the simplest cross-platform option for recursive physionet fetch
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "-P", str(root.parent),
        PTBXL_URL,
    ]
    print("fetching ptb-xl, this takes a while...")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
