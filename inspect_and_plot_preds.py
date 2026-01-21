# inspect_preds_quick.py
"""
Quick inspector:
 - searches checkpoint folders listed in `folders`
 - loads preds & trues from CSV/NPZ where present
 - loads means/stds from sequences_ST3D_from_single.npz (if available) for denorm
 - prints shapes and first N rows
 - flags obvious shape/denorm mismatches
Usage: python inspect_preds_quick.py
"""

import os
from pathlib import Path
import numpy as np

# Edit this if your folder names differ
CHECKPOINT_FOLDERS = [
    "checkpoints_gru",
    "checkpoints_lstm",
    "checkpoints_multi",
    "checkpoints_hybrid",
    "checkpoints_st3d"
]

SEQ_NPZ = "sequences_ST3D_from_single.npz"  # used to fetch means/stds if present
N_ROWS_PRINT = 40

def load_if_exists(path):
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".npz":
        return dict(np.load(p, allow_pickle=True))
    else:
        try:
            return np.loadtxt(str(p), delimiter=",")
        except Exception:
            try:
                return np.genfromtxt(str(p), delimiter=",")
            except Exception:
                return None

def collapse_spatial_if_needed(arr):
    # Accept complicated shapes and collapse to (N, K)
    if arr is None:
        return None
    a = np.array(arr)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    # If there are extra dims (e.g., (N, K, H, W) or (N,1,K,H,W) etc.)
    # reshape into (N, K, rest...) and sum over rest
    N = a.shape[0]
    K = a.shape[1]
    collapsed = a.reshape(N, K, -1).sum(axis=2)
    return collapsed

def try_load_preds(folder):
    # Prefer CSVs saved by trainers; sometimes names vary
    folder = Path(folder)
    preds = None
    trues = None
    # common file names
    candidates = [
        ("test_preds.csv", "test_trues.csv"),
        ("test_preds_reg.csv", "test_trues_reg.csv"),
        ("preds.npz", "trues.npz"),
        ("preds.npy", "trues.npy"),
        ("preds.npz", None)
    ]
    for pfile, tfile in candidates:
        ppath = folder / pfile
        if ppath.exists():
            preds = load_if_exists(ppath)
            if isinstance(preds, dict) and "preds" in preds:
                preds = preds["preds"]
            if tfile:
                tpath = folder / tfile
                trues = load_if_exists(tpath) if tpath.exists() else None
            break

    # fallback: check best-known raw txt in folder
    if preds is None:
        for f in folder.iterdir():
            if f.name.startswith("test_preds") or f.name.startswith("preds"):
                preds = load_if_exists(f)
            if f.name.startswith("test_trues") or f.name.startswith("trues"):
                trues = load_if_exists(f)
    return preds, trues

def main():
    root = Path(".")
    seq_npz = root / SEQ_NPZ
    means = None
    stds = None
    if seq_npz.exists():
        try:
            data = np.load(seq_npz, allow_pickle=True)
            # common keys: 'means','stds' or 'means' 'stds' as scalars
            if "means" in data.files:
                means = np.array(data["means"]).reshape(-1)
            if "stds" in data.files:
                stds = np.array(data["stds"]).reshape(-1)
            print("Loaded sequences npz keys:", data.files)
            print("means:", means.shape if means is not None else None, "stds:", stds.shape if stds is not None else None)
        except Exception as e:
            print("Could not read sequences npz:", e)

    for folder in CHECKPOINT_FOLDERS:
        fpath = root / folder
        print("\n" + "="*30)
        print("Folder:", folder)
        print("="*30)
        if not fpath.exists():
            print("  MISSING")
            continue
        preds, trues = try_load_preds(fpath)
        if preds is None and trues is None:
            print("  No preds/trues found.")
            continue

        # If loaded as dict from .npz, unify
        if isinstance(preds, dict) and "preds" in preds:
            preds = preds["preds"]
        if isinstance(trues, dict) and "trues" in trues:
            trues = trues["trues"]

        # numpy array conversion
        if preds is not None:
            preds = np.array(preds)
        if trues is not None:
            trues = np.array(trues)

        # collapse spatial dims to (N,K)
        preds_coll = collapse_spatial_if_needed(preds) if preds is not None else None
        trues_coll = collapse_spatial_if_needed(trues) if trues is not None else None

        print("pred shape:", None if preds_coll is None else preds_coll.shape)
        print("true shape:", None if trues_coll is None else trues_coll.shape)

        # print few rows (first N)
        if preds_coll is not None and trues_coll is not None:
            nprint = min(N_ROWS_PRINT, preds_coll.shape[0])
            print("\nFirst %d rows (pred vs true):" % nprint)
            for i in range(nprint):
                print(f"{i:03d}: pred={preds_coll[i]}   true={trues_coll[i]}")

            # quick denorm check: if means/stds available and scalars, attempt to denorm first column
            if means is not None and stds is not None:
                if means.size == 1:
                    # scalar normalization
                    denorm0 = preds_coll[:,0]*stds[0] + means[0]
                    tr0 = trues_coll[:,0]
                    print("\nFirst 5 denorm(pred0) vs true0:", denorm0[:5], tr0[:5])
                    # compute naive absolute difference stats
                    diffs = np.abs(denorm0 - tr0)
                    print("denorm diff: min/max/mean:", diffs.min(), diffs.max(), diffs.mean())
                else:
                    print("Means/stds not scalar; skipping quick denorm. Provide per-channel means if needed.")

        else:
            print("Could not align preds/trues for printing: preds", preds is not None, "trues", trues is not None)

if __name__ == "__main__":
    main()
