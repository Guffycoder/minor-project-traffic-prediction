# multi_diag.py
import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

OUT = Path("minor_Eval")
OUT.mkdir(exist_ok=True)

# Paths to try
PRED_PATHS = [
    "checkpoints_multi/test_preds.csv",
    "checkpoints_multi/test_preds_no_denorm.csv",
    "checkpoints_multi/test_preds_reg.csv",
    "checkpoints_multi/preds.csv",
    "checkpoints_multi/test_preds_multi.csv",
]
TRUE_PATHS = [
    "checkpoints_multi/test_trues.csv",
    "checkpoints_multi/test_trues_reg.csv",
    "checkpoints_multi/test_trues_multi.csv",
    "checkpoints_multi/trues.csv",
    "checkpoints_multi/test_trues_no_denorm.csv",
    "checkpoints_hybrid/test_trues.csv",
    "checkpoints/test_trues.csv",
]

def find_first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_csv(p):
    arr = np.loadtxt(p, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

pred_file = find_first(PRED_PATHS)
true_file = find_first(TRUE_PATHS)

if pred_file is None:
    print("No MULTI preds file found in expected locations. Put your CSV at checkpoints_multi/test_preds.csv")
    sys.exit(1)

if true_file is None:
    print("No trues file found in expected locations. Put test_trues.csv alongside preds or in checkpoints_hybrid/test_trues.csv")
    sys.exit(1)

preds = load_csv(pred_file)
trues = load_csv(true_file)
print("Loaded preds:", pred_file, "shape", preds.shape)
print("Loaded trues:", true_file, "shape", trues.shape)

# Align shapes (take min samples and min K)
N = min(preds.shape[0], trues.shape[0])
K = min(preds.shape[1], trues.shape[1])
preds = preds[:N, :K]
trues = trues[:N, :K]

# Basic stats
def stats(a):
    return dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()), std=float(a.std()))
print("Preds stats:", stats(preds))
print("Trues stats:", stats(trues))

# Per-horizon metrics
mae = []
rmse = []
r2s = []
for h in range(K):
    p = preds[:,h]
    t = trues[:,h]
    mae.append(mean_absolute_error(t,p))
    rmse.append(math.sqrt(mean_squared_error(t,p)))
    try:
        r2s.append(r2_score(t,p))
    except:
        r2s.append(float("nan"))

print("\nPer-horizon results:")
for h in range(K):
    print(f" Horizon {h}: MAE={mae[h]:.4f}, RMSE={rmse[h]:.4f}, R2={r2s[h]:.4f}")

print("\nOverall (flattened):")
mae_all = mean_absolute_error(trues.flatten(), preds.flatten())
rmse_all = math.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
r2_all = r2_score(trues.flatten(), preds.flatten())
print(f" MAE={mae_all:.4f}, RMSE={rmse_all:.4f}, R2={r2_all:.4f}")

# Save small sample table (first 20 rows, horizon 0)
print("\nFirst 20 rows (horizon 0):")
for i in range(min(20, N)):
    print(f"{i:03d}: pred={preds[i,0]:.4f}  true={trues[i,0]:.4f}")

# Plot time-series horizon0 (first 500)
M = min(500, N)
plt.figure(figsize=(12,3))
plt.plot(trues[:M,0], label="True", linewidth=2)
plt.plot(preds[:M,0], label="MULTI", alpha=0.8)
plt.legend()
plt.title("MULTI vs True (h0, first {} samples)".format(M))
plt.tight_layout()
plt.savefig(OUT/"multi_time_h0.png", dpi=150)
plt.close()

# Scatter true vs pred per horizon + identity line
for h in range(K):
    plt.figure(figsize=(5,5))
    plt.scatter(trues[:,h], preds[:,h], s=10, alpha=0.4)
    mn = min(trues[:,h].min(), preds[:,h].min())
    mx = max(trues[:,h].max(), preds[:,h].max())
    plt.plot([mn,mx],[mn,mx], 'r--')
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.title(f"MULTI scatter h{h} (MAE={mae[h]:.2f})")
    plt.tight_layout()
    plt.savefig(OUT/f"multi_scatter_h{h}.png", dpi=150)
    plt.close()

# Residual histograms
for h in range(K):
    resid = preds[:,h] - trues[:,h]
    plt.figure(figsize=(6,3))
    plt.hist(resid, bins=80)
    plt.title(f"MULTI residual hist h{h} (mean {resid.mean():.2f}, std {resid.std():.2f})")
    plt.tight_layout()
    plt.savefig(OUT/f"multi_resid_hist_h{h}.png", dpi=150)
    plt.close()

# Row-wise std (to see if predictions are constant)
row_std = preds.std(axis=1)
print("\nPred row-wise std: min/median/mean/max =", float(row_std.min()), float(np.median(row_std)), float(row_std.mean()), float(row_std.max()))
np.savetxt(OUT/"multi_sample_mae_per_row.csv", np.mean(np.abs(preds-trues),axis=1), delimiter=",")

print("\nSaved diagnostics to", OUT)
