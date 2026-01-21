# generate_all_plots.py
"""
Generate all evaluation plots for models (GRU, LSTM, HYBRID, MULTI, ENSEMBLE)
Uses the uploaded NPZ at /mnt/data/sequences_Tin12_pred1.npz (exists in this environment)
and prediction CSVs saved by training scripts.

Saves plots to ./minor_Eval/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# ---------- Config ----------
NPZ_PATH = "D:/minor final/sequences_Tin12_pred1.npz"    # uploaded NPZ (developer-provided)
OUT_DIR = Path("minor_Eval")   # <<< CHANGED FOLDER NAME HERE
OUT_DIR.mkdir(exist_ok=True)
MAX_PLOT_SAMPLES = 500   # number of points for time-series plotting

# candidate csv paths (tries in order)
CANDIDATE_PATHS = {
    "gru": ["checkpoints_gru/test_preds.csv", "checkpoints_gru/test_preds_no_denorm.csv",
            "test_preds_gru.csv", "gru_test_preds.csv"],
    "lstm": ["checkpoints_lstm/test_preds.csv", "checkpoints_lstm/test_preds_no_denorm.csv",
             "test_preds_lstm.csv", "lstm_test_preds.csv"],
    "hyb": ["checkpoints_hybrid/test_preds.csv", "checkpoints_hybrid/test_preds_no_denorm.csv",
            "test_preds_hybrid.csv", "hybrid_test_preds.csv"],
    "multi": ["checkpoints_multi/test_preds.csv", "checkpoints_multi/test_preds_no_denorm.csv",
              "test_preds_multi.csv", "multi_test_preds.csv"],
    "ensemble": ["checkpoints/ensemble_avg_preds.csv", "checkpoints/ensemble_preds.csv",
                 "ensemble_preds.csv", "ensemble_avg.csv"]
}

TRUES_CANDIDATES = ["checkpoints_hybrid/test_trues.csv", "checkpoints/test_trues.csv", "test_trues.csv"]

# ---------- Helpers ----------
def find_first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_csv_array(path):
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def try_load_preds():
    preds = {}
    for name, cand in CANDIDATE_PATHS.items():
        p = find_first(cand)
        if p:
            try:
                preds[name] = load_csv_array(p)
                print(f"Loaded {name} preds from {p} -> shape {preds[name].shape}")
            except Exception as e:
                print(f"Couldn't load {p}: {e}")
        else:
            print(f"No preds file found for {name}")
    return preds

def load_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data

def ensure_same_length_dict(preds_dict):
    n_min = min(v.shape[0] for v in preds_dict.values())
    for k in preds_dict:
        preds_dict[k] = preds_dict[k][:n_min]
    return preds_dict, n_min

def compute_per_horizon_metrics(preds, trues):
    K = trues.shape[1]
    maes = []
    rmses = []
    r2s = []
    for h in range(K):
        t = trues[:, h]
        p = preds[:, h]
        maes.append(mean_absolute_error(t, p))
        rmses.append(math.sqrt(mean_squared_error(t, p)))
        try:
            r2s.append(r2_score(t, p))
        except:
            r2s.append(float('nan'))
    return np.array(maes), np.array(rmses), np.array(r2s)

def plot_time_series_stepwise(preds_dict, trues, max_samples=500):
    K = trues.shape[1]
    N = min(len(trues), max_samples)
    for h in range(K):
        plt.figure(figsize=(14,4))
        x = np.arange(N)
        plt.plot(x, trues[:N, h], label=f"True (step {h})")
        for name, arr in preds_dict.items():
            plt.plot(x, arr[:N, h], label=f"{name.upper()} Pred (step {h})")
        plt.title(f"Step {h} — first {N} points")
        plt.xlabel("Index")
        plt.ylabel("Traffic")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"pred_step_{h}.png", dpi=150)
        plt.close()

def plot_scatter_per_model(preds_dict, trues):
    K = trues.shape[1]
    for name, arr in preds_dict.items():
        for h in range(K):
            plt.figure(figsize=(6,6))
            plt.scatter(trues[:, h], arr[:, h], s=12, alpha=0.6)
            mn = min(trues[:, h].min(), arr[:, h].min())
            mx = max(trues[:, h].max(), arr[:, h].max())
            plt.plot([mn, mx], [mn, mx], 'r--')
            plt.xlabel("True")
            plt.ylabel("Pred")
            plt.title(f"{name.upper()} Scatter h{h}")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{name}_scatter_h{h}.png", dpi=150)
            plt.close()

def plot_resid_histograms(preds_dict, trues):
    K = trues.shape[1]
    for name, arr in preds_dict.items():
        for h in range(K):
            resid = arr[:, h] - trues[:, h]
            plt.figure(figsize=(7,4))
            plt.hist(resid, bins=50)
            plt.title(f"{name.upper()} Residual Hist h{h}")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{name}_resid_hist_h{h}.png", dpi=150)
            plt.close()

def plot_resid_heatmap(preds, trues, name):
    resid = preds - trues
    plt.figure(figsize=(7,7))
    vmax = np.percentile(np.abs(resid), 99)
    plt.imshow(resid, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title(f"{name.upper()} Residual heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"resid_heatmap_{name}.png", dpi=150)
    plt.close()

def plot_per_horizon_metrics(all_preds, trues):
    K = trues.shape[1]

    # MAE
    plt.figure(figsize=(10,6))
    for name, arr in all_preds.items():
        mae, rmse, r2 = compute_per_horizon_metrics(arr, trues)
        plt.plot(range(K), mae, marker='o', label=name.upper())
    plt.title("Per-Horizon MAE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mae_by_horizon_all_models.png")
    plt.close()

    # RMSE
    plt.figure(figsize=(10,6))
    for name, arr in all_preds.items():
        mae, rmse, r2 = compute_per_horizon_metrics(arr, trues)
        plt.plot(range(K), rmse, marker='o', label=name.upper())
    plt.title("Per-Horizon RMSE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rmse_by_horizon_all_models.png")
    plt.close()

    # R2
    plt.figure(figsize=(10,6))
    for name, arr in all_preds.items():
        mae, rmse, r2 = compute_per_horizon_metrics(arr, trues)
        plt.plot(range(K), r2, marker='o', label=name.upper())
    plt.title("Per-Horizon R2")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "r2_by_horizon_all_models.png")
    plt.close()

def cumulative_sorted_mae_plot(preds, trues, name):
    mae_per_sample = np.mean(np.abs(preds - trues), axis=1)
    sorted_vals = np.sort(mae_per_sample)
    cum = np.cumsum(sorted_vals)
    plt.figure(figsize=(10,5))
    plt.plot(cum)
    plt.title(f"Cumulative sorted MAE — {name.upper()}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_cumulative_error.png", dpi=150)
    plt.close()


# ---------- MAIN ----------
if __name__ == "__main__":

    # Load NPZ
    data = load_npz(NPZ_PATH)

    # Load trues
    trues_path = find_first(TRUES_CANDIDATES)
    if trues_path:
        trues = load_csv_array(trues_path)
    else:
        trues = data["Y_test"].reshape((data["Y_test"].shape[0], -1))

    # Load model predictions
    preds = try_load_preds()

    # reshape & align
    for k in preds:
        if preds[k].ndim == 1:
            preds[k] = preds[k].reshape(-1,1)
        preds[k] = preds[k].reshape((preds[k].shape[0], -1))

    preds, nmin = ensure_same_length_dict(preds)
    trues = trues[:nmin]

    # ---------- Generate All Plots ----------
    plot_time_series_stepwise(preds, trues)
    plot_scatter_per_model(preds, trues)
    plot_resid_histograms(preds, trues)

    for name, arr in preds.items():
        plot_resid_heatmap(arr, trues, name)
        cumulative_sorted_mae_plot(arr, trues, name)

    plot_per_horizon_metrics(preds, trues)

    print("All images saved in folder:", OUT_DIR)
