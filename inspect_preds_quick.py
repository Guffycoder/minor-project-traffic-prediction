#!/usr/bin/env python3
"""
inspect_preds_quick.py
Load predictions/trues from your model checkpoint folders and print quick stats + simple plots.
By default it looks for:
  checkpoints_gru/test_preds.csv, test_trues.csv
  checkpoints_lstm/test_preds.csv, test_trues.csv
  checkpoints_hybrid/test_preds.csv, test_trues.csv
  checkpoints_multi/test_preds_reg.csv, test_trues_reg.csv
If files missing, it will skip them.
"""
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

folders = [
    ("GRU", "checkpoints_gru"),
    ("LSTM", "checkpoints_lstm"),
    ("HYBRID", "checkpoints_hybrid"),
    ("MULTI", "checkpoints_multi"),
    ("ST3D", "checkpoints_st3d"),
]

found = []

def try_load_preds(folder):
    # common possibilities
    candidates = [
        ("test_preds.csv", "test_trues.csv"),
        ("test_preds_reg.csv", "test_trues_reg.csv"),
        ("test_preds_multi.csv", "test_trues_multi.csv"),
    ]
    for pfile, tfile in candidates:
        ppth = os.path.join(folder, pfile)
        tpth = os.path.join(folder, tfile)
        if os.path.exists(ppth) and os.path.exists(tpth):
            p = np.loadtxt(ppth, delimiter=",")
            t = np.loadtxt(tpth, delimiter=",")
            return p, t, ppth, tpth
    return None, None, None, None

for name, folder in folders:
    p, t, pp, tp = try_load_preds(folder)
    if p is None:
        # fallback: maybe test_preds.csv/test_trues.csv
        ppth = os.path.join(folder, "test_preds.csv")
        tpth = os.path.join(folder, "test_trues.csv")
        if os.path.exists(ppth) and os.path.exists(tpth):
            p = np.loadtxt(ppth, delimiter=",")
            t = np.loadtxt(tpth, delimiter=",")
            pp, tp = ppth, tpth

    if p is None:
        print(f"Folder: {folder} — no preds/trues found, skipping.")
        continue

    p = np.atleast_2d(p)
    t = np.atleast_2d(t)
    # ensure shapes: (N, K)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    # align K
    K = min(p.shape[1], t.shape[1])
    p = p[:, :K]
    t = t[:, :K]

    found.append((name, folder, p, t))
    print(f"Loaded {name} from {folder} -> preds {p.shape} trues {t.shape}")

if len(found) == 0:
    print("No prediction files found in any checkpoint folders.")
    raise SystemExit(0)

# Choose a horizon step to inspect
step = 0

# Plot first N samples overlayed
N = 500
plt.figure(figsize=(12,4))
for idx,(name, folder, p, t) in enumerate(found):
    n = min(N, p.shape[0])
    # Plot true once (first model) and predictions for each model
    if idx == 0:
        plt.plot(t[:n, step], label="True", linewidth=2)
    plt.plot(p[:n, step], linestyle='--', label=f"{name} pred", alpha=0.9)
plt.legend()
plt.title(f"First {N} test points - horizon {step}")
out = os.path.join("checkpoints", f"compare_step{step}.png")
os.makedirs("checkpoints", exist_ok=True)
plt.savefig(out, dpi=140)
print("Saved time series comparison to", out)
plt.close()

# Print metrics and small table
for name, folder, p, t in found:
    mae = mean_absolute_error(t.flatten(), p.flatten())
    # compute RMSE in a sklearn-version-safe way:
    mse = mean_squared_error(t.flatten(), p.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(t.flatten(), p.flatten())
    abs_err = np.abs(t - p)
    acc1 = 100 * np.mean(abs_err <= 1)
    acc5 = 100 * np.mean(abs_err <= 5)
    acc10 = 100 * np.mean(abs_err <= 10)
    print(f"\n{name} metrics (all horizons flattened):")
    print(f" MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    print(f" ±1={acc1:.2f}%, ±5={acc5:.2f}%, ±10={acc10:.2f}%")
    # show first 8 rows at inspected step
    print(" First 8 rows (pred, true) at horizon", step)
    for i in range(8):
        print(f" {i:03d}: pred={p[i,step]:.3f}  true={t[i,step]:.3f}")

# Step-wise multi-step metrics (if K>1)
K = found[0][2].shape[1]
maes = []
rmses = []
r2s = []
for k in range(K):
    mm = []
    rr = []
    r22 = []
    for name, folder, p, t in found:
        m = mean_absolute_error(t[:,k], p[:,k])
        # safe rmse calculation:
        r = np.sqrt(mean_squared_error(t[:,k], p[:,k]))
        rr2 = r2_score(t[:,k], p[:,k])
        mm.append(m); rr.append(r); r22.append(rr2)
    maes.append(mm)
    rmses.append(rr)
    r2s.append(r22)

# Save step-wise MAE plot for first model (for visualization)
labels = [f"{name}" for (name,_,_,_) in found]
maes_arr = np.array(maes)  # (K, n_models)
plt.figure(figsize=(8,4))
for i,name in enumerate(labels):
    plt.plot(maes_arr[:, i], label=name, marker='o')
plt.xlabel("Horizon step")
plt.ylabel("MAE")
plt.title("Per-step MAE for models")
plt.legend()
out2 = os.path.join("checkpoints", "multi_step_mae.png")
plt.savefig(out2, dpi=140)
print("Saved step-wise MAE plot to", out2)
plt.close()
