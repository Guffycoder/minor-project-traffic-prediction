# generate_all_plots_with_accuracy.py
"""
Generate and save all required diagnostic plots for your minor project.
Saves images into folder: final_all_required_image
Reads predictions from:
  checkpoints_gru/test_preds.csv, test_trues.csv
  checkpoints_lstm/test_preds.csv, test_trues.csv
  checkpoints_hybrid/test_preds.csv, test_trues.csv
  checkpoints_multi/test_preds.csv, test_trues.csv
If some files are missing the script will try common fallback names.
It also computes simple ensembles (avg and weighted by inverse RMSE) and
saves a metrics CSV with MAE, RMSE, R2 and a custom 'Accuracy' for regression.

Note: For regression there is no canonical 'accuracy'. We use a simple
proxy: Accuracy = 1 - (RMSE / range_of_true_values), clipped to [0,1].
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT_DIR = "final_all_required_image"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "GRU": "checkpoints_gru/test_preds.csv",
    "LSTM": "checkpoints_lstm/test_preds.csv",
    "HYBRID": "checkpoints_hybrid/test_preds.csv",
    "MULTI": "checkpoints_multi/test_preds.csv"
}
# try fallback filenames for multi or reg variants
FALLBACK_MAP = {
    "MULTI": ["checkpoints_multi/test_preds_reg.csv", "checkpoints_multi/test_preds.csv"]
}


def try_load(path):
    if os.path.exists(path):
        return np.loadtxt(path, delimiter=",")
    return None


def load_model_preds():
    preds = {}
    trues = {}
    # load trues common from multiple folders as baseline (first found)
    true_path_candidates = [
        "checkpoints_gru/test_trues.csv",
        "checkpoints_lstm/test_trues.csv",
        "checkpoints_hybrid/test_trues.csv",
        "checkpoints_multi/test_trues.csv"
    ]
    true_data = None
    for t in true_path_candidates:
        if os.path.exists(t):
            true_data = np.loadtxt(t, delimiter=",")
            break
    if true_data is None:
        raise FileNotFoundError("No test_trues.csv found in any checkpoints folder.")
    # ensure 2D
    if true_data.ndim == 1:
        true_data = true_data.reshape(-1, 1)
    # load each model preds
    for name, ppath in MODELS.items():
        arr = try_load(ppath)
        if arr is None and name in FALLBACK_MAP:
            for alt in FALLBACK_MAP[name]:
                if os.path.exists(alt):
                    arr = np.loadtxt(alt, delimiter=",")
                    break
        if arr is None:
            print(f"[WARN] Predictions for {name} not found at {ppath}. Skipping model.")
            preds[name] = None
            trues[name] = None
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        preds[name] = arr
        # align trues to same horizon length K (take prefix)
        K = min(arr.shape[1], true_data.shape[1])
        trues[name] = true_data[:, :K]
    return preds, trues


preds, trues = load_model_preds()

# Build ensembles (only for models that loaded)
available = [m for m in preds if preds[m] is not None]
print("Available models:", available)

# Align K across models: choose min horizon across available
Ks = [preds[m].shape[1] for m in available]
K = min(Ks) if Ks else 0
print("Using horizon K =", K)
if K == 0:
    raise RuntimeError("No available predictions with a positive horizon K.")

# Trim preds/trues to K
for m in available:
    preds[m] = preds[m][:, :K]
    trues[m] = trues[m][:, :K]

# Simple average ensemble
ens_avg = np.mean([preds[m] for m in available], axis=0)

# Weighted ensemble by inverse RMSE (flattened)
# Compute RMSE per model (against the chosen baseline trues array)
baseline_true = trues[available[0]]
rmses = {}
for m in available:
    rm = mean_squared_error(baseline_true.flatten(), preds[m].flatten()) ** 0.5
    rmses[m] = rm
# guard against zero RMSE
inv = {}
for m in rmses:
    inv[m] = 1.0 / (rmses[m] + 1e-12)
total = sum(inv.values())
weights = {m: inv[m] / total for m in inv}
ens_weighted = sum(preds[m] * weights[m] for m in preds if preds[m] is not None)

# Save ensembles into preds dict for plotting convenience
preds["ENS_AVG"] = ens_avg
preds["ENS_WEIGHTED"] = ens_weighted
trues["ENS_AVG"] = baseline_true[:, :K]
trues["ENS_WEIGHTED"] = baseline_true[:, :K]

# Build models_all: only those keys that have arrays (not None)
models_all = [m for m in list(preds.keys()) if preds[m] is not None]

# --- Metrics utilities ---

def accuracy_score_regression(pred, true):
    """Proxy accuracy for regression.
    acc = 1 - (RMSE / (max(true)-min(true))) clipped to [0,1].
    If true has zero range, returns 0.0.
    """
    t = true.flatten()
    p = pred.flatten()
    if t.size == 0:
        return 0.0
    rmse = mean_squared_error(t, p) ** 0.5
    data_range = t.max() - t.min()
    if data_range == 0:
        return 0.0
    acc = 1.0 - (rmse / data_range)
    # clip
    return float(max(0.0, min(1.0, acc)))


def metrics_for(pred, true):
    mae = mean_absolute_error(true.flatten(), pred.flatten())
    rmse = mean_squared_error(true.flatten(), pred.flatten()) ** 0.5
    r2 = r2_score(true.flatten(), pred.flatten())
    acc = accuracy_score_regression(pred, true)
    return mae, rmse, r2, acc

# Create metrics table
metrics = {}
for m in models_all:
    metrics[m] = metrics_for(preds[m], trues[m])

# Save CSV metrics
import csv
csv_path = os.path.join(OUT_DIR, "metrics_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "MAE", "RMSE", "R2", "ACCURACY"])
    for m in sorted(metrics.keys()):
        mae, rmse, r2v, acc = metrics[m]
        writer.writerow([m, mae, rmse, r2v, acc])
print("Saved metrics CSV:", csv_path)

# --- PLOT 1: Time series overlay (first N points) for True vs GRU vs LSTM (horizon 0) ---
N = 500
h = 0
plt.figure(figsize=(12, 4))
tarr = trues[available[0]][:N, h]
plt.plot(tarr, label="True", linewidth=2, color="k")
if "GRU" in preds and preds["GRU"] is not None:
    plt.plot(preds["GRU"][:N, h], label="GRU", alpha=0.9)
if "LSTM" in preds and preds["LSTM"] is not None:
    plt.plot(preds["LSTM"][:N, h], label="LSTM", alpha=0.9)
plt.title(f"True vs GRU vs LSTM (first {N}) - horizon {h}")
plt.xlabel("sample index")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"time_series_true_vs_gru_lstm_step{h}.png"), dpi=150)
plt.close()

# --- PLOT 2: Time series overlay True vs ALL models (first N, horizon 0) ---
plt.figure(figsize=(12, 4))
plt.plot(tarr, label="True", linewidth=2, color="k")
for m in ["GRU", "LSTM", "HYBRID", "MULTI", "ENS_AVG", "ENS_WEIGHTED"]:
    if m in preds and preds[m] is not None:
        plt.plot(preds[m][:N, h], label=m, alpha=0.9)
plt.title(f"True vs All Models (first {N}) - horizon {h}")
plt.xlabel("sample index")
plt.ylabel("value")
plt.legend(ncol=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"time_series_true_vs_all_models_step{h}.png"), dpi=150)
plt.close()

# --- PLOT 3: Scatter True vs Pred for horizon 0 (one subplot per model) ---
plt.figure(figsize=(14, 8))
ncols = 3
nrows = int(np.ceil(len(models_all) / ncols))
for i, m in enumerate(models_all):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.scatter(trues[m][:, h], preds[m][:, h], s=10, alpha=0.3)
    mn = min(trues[m][:, h].min(), preds[m][:, h].min())
    mx = max(trues[m][:, h].max(), preds[m][:, h].max())
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.8)
    mae, rmse, r2v, acc = metrics[m]
    ax.set_title(f"{m} (MAE={mae:.3f} RMSE={rmse:.3f} ACC={acc:.3f})")
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_scatter_true_vs_pred_step0.png"), dpi=150)
plt.close()

# --- PLOT 4: Residual histograms (horizon 0) overlay ---
plt.figure(figsize=(10, 5))
# automatic bin range based on data spread but with fallback
all_res = np.concatenate([(preds[m][:, h] - trues[m][:, h]) for m in models_all])
bins = np.linspace(all_res.min() - 1e-6, all_res.max() + 1e-6, 160)
for m in models_all:
    res = preds[m][:, h] - trues[m][:, h]
    plt.hist(res, bins=bins, alpha=0.4, label=m, density=True)
plt.title("Residual histograms (horizon 0) - overlay (density)")
plt.xlabel("Residual (pred - true)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_histograms_h0.png"), dpi=150)
plt.close()

# --- PLOT 5: Per-horizon MAE and RMSE line plots ---
horizons = list(range(K))
plt.figure(figsize=(10, 5))
for m in models_all:
    maes = [mean_absolute_error(trues[m][:, k], preds[m][:, k]) for k in horizons]
    plt.plot(horizons, maes, marker='o', label=m)
plt.xlabel("Horizon step")
plt.ylabel("MAE")
plt.title("Per-horizon MAE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_horizon_mae.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
for m in models_all:
    rmses_h = [mean_squared_error(trues[m][:, k], preds[m][:, k]) ** 0.5 for k in horizons]
    plt.plot(horizons, rmses_h, marker='o', label=m)
plt.xlabel("Horizon step")
plt.ylabel("RMSE")
plt.title("Per-horizon RMSE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_horizon_rmse.png"), dpi=150)
plt.close()

# --- PLOT 6: Boxplot of absolute errors per horizon for each model (arranged per-horizon) ---
plt.figure(figsize=(14, 8))
ncols = 3
nrows = int(np.ceil(len(models_all) / ncols))
for i, m in enumerate(models_all):
    ax = plt.subplot(nrows, ncols, i + 1)
    abs_errs = [np.abs(preds[m][:, k] - trues[m][:, k]) for k in horizons]
    ax.boxplot(abs_errs, labels=[str(k) for k in horizons], showfliers=False)
    ax.set_title(f"{m} abs-error by horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Abs error")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_horizon_boxplot_errors.png"), dpi=150)
plt.close()

# --- PLOT 7: Distribution of predictions vs true for horizon 0 (density hist) ---
plt.figure(figsize=(10, 5))
min_val = min(trues[available[0]][:, h].min(), min(preds[m][:, h].min() for m in models_all))
max_val = max(trues[available[0]][:, h].max(), max(preds[m][:, h].max() for m in models_all))
bins = np.linspace(min_val, max_val, 120)
plt.hist(trues[available[0]][:, h], bins=bins, alpha=0.6, label='True', density=True)
for m in models_all:
    plt.hist(preds[m][:, h], bins=bins, alpha=0.35, label=m, density=True)
plt.title("Distribution: predictions vs true (horizon 0)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "distribution_preds_vs_true_h0.png"), dpi=150)
plt.close()

print("Saved all plots to", OUT_DIR)
print("Metrics (saved to):", csv_path)
