# debug_preds_v2.py
import os
import numpy as np
import argparse

def load_array(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".csv") or path.endswith(".txt"):
        return np.loadtxt(path, delimiter=",")
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        for k in ("preds","pred","test_preds","y_pred"):
            if k in d.files:
                return d[k]
        return d[d.files[0]]
    return np.loadtxt(path, delimiter=",")

def flatten_true_row(row):
    row = np.array(row)
    if row.ndim == 0:
        return row.item()
    if row.ndim == 1:
        return row
    # if row has extra dims (e.g., (K,1,1)) -> squeeze then flatten
    return row.reshape(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="checkpoints_hybrid/test_preds.csv")
    parser.add_argument("--true", default="checkpoints_hybrid/test_trues.csv")
    parser.add_argument("--show", type=int, default=100)
    args = parser.parse_args()

    preds_raw = load_array(args.pred)
    trues_raw = load_array(args.true)
    if preds_raw is None or trues_raw is None:
        raise FileNotFoundError("Preds or trues not found at given paths.")

    print("raw shapes -> preds:", np.array(preds_raw).shape, "trues:", np.array(trues_raw).shape)

    # If preds is 2D with single column -> flatten
    preds = np.array(preds_raw)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.flatten()

    trues = np.array(trues_raw)

    # If trues are multi-step, keep them as-is; if they have spatial dims, we leave them
    N = min(len(preds), trues.shape[0])
    show_n = min(args.show, N)
    print(f"Showing first {show_n} entries (index: pred | true-row | pred-true)")

    for i in range(show_n):
        p = preds[i] if preds.ndim==1 else preds[i,0]  # if preds multi-step, show first step
        trow = trues[i]
        t_flat = flatten_true_row(trow)
        # If multi-step, print the whole row; else print scalar
        if np.ndim(t_flat) == 0:
            err = p - float(t_flat)
            print(f"{i:03d}: pred={float(p):.5f} | true={float(t_flat):.5f} | err={err:.5f}")
        else:
            # show first few elements of true row
            first_k = t_flat[:10]
            err0 = p - float(t_flat[0])
            print(f"{i:03d}: pred={float(p):.5f} | true_row(first10)={first_k} | err_at_h0={err0:.5f}")

    # summary stats
    if preds.ndim == 1:
        print("\nPred stats: min/max/mean/std", preds.min(), preds.max(), preds.mean(), preds.std())
    else:
        print("\nPred shape (multi):", preds.shape)

    print("True stats (h0 only):")
    # compute stats for horizon 0 (if multi-step)
    if trues.ndim == 1:
        t0 = trues[:N].astype(float)
    else:
        # take first horizon if second dimension exists
        if trues.ndim >= 2:
            t0 = trues[:N, 0].astype(float).reshape(-1)
        else:
            t0 = trues[:N].reshape(-1)
    print("min/max/mean/std:", t0.min(), t0.max(), t0.mean(), t0.std())
