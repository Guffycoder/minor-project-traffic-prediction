#!/usr/bin/env python3
"""
generate_all_plots_full.py
Robust script to load ground-truth (from sequences npz) and model prediction CSVs
and generate a large set of diagnostic and presentation-quality plots.
Outputs saved into ./final_all_required_image/
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path("final_all_required_image")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- CONFIG: paths (edit if your files are elsewhere) ----------
SEQ_NPZ = Path("sequences_Tin12_pred1.npz")   # your uploaded dataset file
CHECKPOINT_FOLDERS = {
    "GRU": Path("checkpoints_gru"),
    "LSTM": Path("checkpoints_lstm"),
    "HYBRID": Path("checkpoints_hybrid"),
    "MULTI": Path("checkpoints_multi"),
}
ENSEMBLE_AVG = Path("checkpoints/ensemble_avg_preds.csv")   # optional
ENSEMBLE_WEIGHTS = Path("checkpoints/ensemble_weights.json")  # optional (not required)

# ---------- helper functions ----------
def collapse_to_NK(arr):
    """Collapse variety of shapes into (N,K). Sum spatial dims if present and squeeze singletons."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim > 2:
        axes = tuple(range(2, a.ndim))
        a = a.sum(axis=axes)
    return a

def load_npz_Y_test(npz_path):
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"{npz_path} not found. Put your sequences npz in repo root or change SEQ_NPZ.")
    d = dict(np.load(npz_path, allow_pickle=True))
    # attempt to find Y_test key
    ykey = None
    keys_lower = {k.lower(): k for k in d.keys()}
    for key_try in ("y_test", "ytest", "y_test_reg", "y_reg_test", "y_reg"):
        if key_try in keys_lower:
            ykey = keys_lower[key_try]; break
    if ykey is None:
        # choose first key that starts with 'y' or contains 'test' and 'y'
        for k in d.keys():
            kl = k.lower()
            if kl.startswith("y") and "test" in kl:
                ykey = k; break
        if ykey is None:
            for k in d.keys():
                if kl.startswith("y"):
                    ykey = k; break
    if ykey is None:
        # fallback: try "Y_test" uppercase presence
        if "Y_test" in d:
            ykey = "Y_test"
    if ykey is None:
        raise KeyError(f"Could not find a Y_test key in {npz_path}. Keys: {list(d.keys())}")
    Y_test_raw = d[ykey]
    Y_test = collapse_to_NK(Y_test_raw)
    return Y_test, ykey, d

def try_load_preds(folder: Path):
    """Return preds (N,K) and trues (N,K) if present in folder, else (None,None)."""
    if not folder.exists():
        return None, None
    # common file names
    cand_pairs = [
        ("test_preds.csv", "test_trues.csv"),
        ("test_preds_reg.csv", "test_trues_reg.csv"),
        ("preds.csv", "trues.csv"),
        ("test_preds_no_denorm.csv", "test_trues_no_denorm.csv"),
    ]
    for pfile, tfile in cand_pairs:
        ppth = folder / pfile
        tpth = folder / tfile
        if ppth.exists() and tpth.exists():
            try:
                p = np.loadtxt(ppth, delimiter=",")
                t = np.loadtxt(tpth, delimiter=",")
                p = np.atleast_2d(p); t = np.atleast_2d(t)
                return p, t
            except Exception as e:
                print("Error loading", ppth, tpth, e)
    # fallback: maybe preds exist but trues not
    ppth = folder / "test_preds.csv"
    tpth = folder / "test_trues.csv"
    if ppth.exists():
        p = np.loadtxt(ppth, delimiter=","); p = np.atleast_2d(p)
        if tpth.exists():
            t = np.loadtxt(tpth, delimiter=","); t = np.atleast_2d(t)
        else:
            t = None
        return p, t
    return None, None

def align_preds_to_truth(preds, truth):
    """
    Ensure preds and truth shapes align as (N,K).
    If preds have fewer horizons, tile last column.
    If preds have more horizons, crop.
    If shapes mismatch in N, attempt transpose fallback.
    """
    preds = np.atleast_2d(preds)
    truth = np.atleast_2d(truth)
    N_true, K_true = truth.shape
    Np, Kp = preds.shape
    # try transpose if dimensions match when transposed
    if Np != N_true and Kp == N_true and Np == K_true:
        preds = preds.T
        Np, Kp = preds.shape
    # if row count mismatch, attempt simple repeat/drop (rare)
    if Np != N_true:
        raise ValueError(f"Row count mismatch preds {preds.shape} vs truth {truth.shape}")
    if Kp < K_true:
        # tile last col
        last = np.tile(preds[:, -1].reshape(-1,1), (1, K_true - Kp))
        preds = np.concatenate([preds, last], axis=1)
    elif Kp > K_true:
        preds = preds[:, :K_true]
    return preds, truth

# ---------- load truth ----------
print("Loading Y_test from:", SEQ_NPZ)
Y_test, used_key, full_npz = load_npz_Y_test(SEQ_NPZ)
N, K = Y_test.shape
print("Y_test loaded as", used_key, "-> shape", Y_test.shape)
# If the npz contains means/stds show them
means = full_npz.get("means", None)
stds = full_npz.get("stds", None)
if means is not None and stds is not None:
    print("Means/stds in npz found. means shape:", np.asarray(means).shape, "stds shape:", np.asarray(stds).shape)

# ---------- load model preds ----------
models = {}
for name, folder in CHECKPOINT_FOLDERS.items():
    p, t = try_load_preds(folder)
    if p is None:
        print(f"{name}: no preds found in {folder}")
        continue
    # if trues not present in folder, use global Y_test
    if t is None:
        t = Y_test
    # collapse t if needed
    t = collapse_to_NK(t)
    p = np.atleast_2d(p)
    # align shapes
    try:
        p, t = align_preds_to_truth(p, t)
    except Exception as e:
        print(f"Error aligning {name} preds/trues:", e)
        continue
    models[name] = {"preds": p, "trues": t}
    print(f"Loaded {name} preds shape {p.shape}")

# optional ensemble avg
ensemble = None
if ENSEMBLE_AVG.exists():
    ensemble = np.loadtxt(ENSEMBLE_AVG, delimiter=",")
    ensemble = np.atleast_2d(ensemble)
    if ensemble.shape[0] == N and ensemble.shape[1] >= K:
        ensemble = ensemble[:, :K]
    elif ensemble.shape[1] == K and ensemble.shape[0] == N:
        pass
    else:
        # try transpose
        if ensemble.T.shape[0] == N and ensemble.T.shape[1] >= K:
            ensemble = ensemble.T
    print("Loaded ensemble avg shape:", ensemble.shape)

# ---------- metrics helpers ----------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def per_horizon_metrics(preds, trues):
    preds = np.asarray(preds); trues = np.asarray(trues)
    K = trues.shape[1]
    mae = [mean_absolute_error(trues[:,i], preds[:,i]) for i in range(K)]
    rmse = [math.sqrt(mean_squared_error(trues[:,i], preds[:,i])) for i in range(K)]
    r2 = [r2_score(trues[:,i], preds[:,i]) for i in range(K)]
    return np.array(mae), np.array(rmse), np.array(r2)

# ---------- PLOTS ----------
def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)

# 1) Heatmap of truth
fig = plt.figure(figsize=(10,6))
plt.imshow(Y_test, aspect="auto")
plt.colorbar()
plt.title("True targets heatmap (samples × horizon)")
plt.xlabel("Horizon")
plt.ylabel("Sample index")
savefig(fig, "heatmap_true.png")

# 2) True distribution hist for each horizon
fig = plt.figure(figsize=(12,6))
for i in range(K):
    plt.subplot(1, K, i+1)
    plt.hist(Y_test[:,i], bins=40)
    plt.title(f"h{i} dist")
savefig(fig, "dist_true_by_horizon.png")

# 3) Multi-model overlay for first horizon (first 500 points)
step = 0
Nshow = min(500, N)
fig = plt.figure(figsize=(14,5))
plt.plot(Y_test[:Nshow, step], label="True", linewidth=2)
for name, data in models.items():
    plt.plot(data["preds"][:Nshow, step], label=name)
if ensemble is not None:
    plt.plot(ensemble[:Nshow, step], label="Ensemble", linestyle="--", linewidth=1.5)
plt.legend(); plt.title(f"True vs Models (first {Nshow}) h{step}")
savefig(fig, "true_vs_models_step0.png")

# 4) Multi-step example: show first sample forecast (true vs each model) across horizons
sample_idx = 0
fig = plt.figure(figsize=(8,4))
plt.plot(range(K), Y_test[sample_idx], marker='o', label="True")
for name, data in models.items():
    plt.plot(range(K), data["preds"][sample_idx], marker='x', label=name)
if ensemble is not None:
    plt.plot(range(K), ensemble[sample_idx], marker='s', label="Ensemble")
plt.xlabel("Horizon"); plt.ylabel("Value")
plt.title(f"Multi-step forecast example (sample {sample_idx})")
plt.legend(); savefig(fig, "multi_step_example_sample0.png")

# 5) Per-horizon MAE/RMSE/R2 plot for all models
fig = plt.figure(figsize=(10,5))
for name, data in models.items():
    mae, rmse, r2 = per_horizon_metrics(data["preds"], data["trues"])
    plt.plot(range(K), mae, marker='o', label=f"{name} MAE")
plt.xlabel("Horizon"); plt.ylabel("MAE"); plt.title("Per-horizon MAE"); plt.legend()
savefig(fig, "mae_by_horizon_all_models.png")

fig = plt.figure(figsize=(10,5))
for name, data in models.items():
    mae, rmse, r2 = per_horizon_metrics(data["preds"], data["trues"])
    plt.plot(range(K), rmse, marker='o', label=f"{name} RMSE")
plt.xlabel("Horizon"); plt.ylabel("RMSE"); plt.title("Per-horizon RMSE"); plt.legend()
savefig(fig, "rmse_by_horizon_all_models.png")

fig = plt.figure(figsize=(10,5))
for name, data in models.items():
    mae, rmse, r2 = per_horizon_metrics(data["preds"], data["trues"])
    plt.plot(range(K), r2, marker='o', label=f"{name} R2")
plt.xlabel("Horizon"); plt.ylabel("R2"); plt.title("Per-horizon R2"); plt.legend()
savefig(fig, "r2_by_horizon_all_models.png")

# 6) Residual heatmaps for each model
for name, data in models.items():
    resid = data["preds"] - data["trues"]
    fig = plt.figure(figsize=(8,6))
    plt.imshow(resid, aspect="auto", cmap="RdBu", vmin=-np.max(np.abs(resid)), vmax=np.max(np.abs(resid)))
    plt.colorbar(); plt.title(f"Residual heatmap: {name} (pred - true)")
    savefig(fig, f"resid_heatmap_{name}.png")

# 7) Residual distributions and boxplots (horizon-wise)
for name, data in models.items():
    resid = (data["preds"] - data["trues"])
    # histogram for h0
    fig = plt.figure(figsize=(6,4))
    plt.hist(resid[:,0], bins=50)
    plt.title(f"{name} residual histogram h0")
    savefig(fig, f"{name}_resid_hist_h0.png")
    # boxplot across horizons
    fig = plt.figure(figsize=(8,4))
    plt.boxplot([resid[:,i] for i in range(K)], labels=[f"h{i}" for i in range(K)])
    plt.title(f"{name} residuals boxplot by horizon")
    savefig(fig, f"{name}_resid_boxplot_by_horizon.png")

# 8) Scatter true vs pred (h0) and linear fit
for name, data in models.items():
    fig = plt.figure(figsize=(5,5))
    plt.scatter(data["trues"][:,0], data["preds"][:,0], s=6, alpha=0.6)
    mn = min(data["trues"][:,0].min(), data["preds"][:,0].min())
    mx = max(data["trues"][:,0].max(), data["preds"][:,0].max())
    plt.plot([mn,mx],[mn,mx], color='red')
    plt.xlabel("True h0"); plt.ylabel(f"{name} pred h0"); plt.title(f"{name} scatter h0")
    savefig(fig, f"{name}_scatter_h0.png")

# 9) Cumulative absolute error curve per model (sorted by sample index)
for name, data in models.items():
    abs_err = np.abs(data["preds"] - data["trues"]).mean(axis=1)  # mean across horizons per sample
    cum = np.cumsum(np.sort(abs_err))
    fig = plt.figure(figsize=(8,4))
    plt.plot(cum, label=name)
    plt.title(f"Cumulative sorted mean-abs-error per sample: {name}")
    plt.xlabel("sample rank"); plt.ylabel("cumulative error")
    savefig(fig, f"{name}_cumulative_error.png")

# 10) Multi-model per-sample small-multiples (first 5 samples) — show true vs preds across horizons
for i in range(5):
    fig = plt.figure(figsize=(8,3))
    plt.plot(range(K), Y_test[i], marker='o', label="True")
    for name, data in models.items():
        plt.plot(range(K), data["preds"][i], marker='x', label=name)
    if ensemble is not None:
        plt.plot(range(K), ensemble[i], marker='s', label="Ensemble")
    plt.title(f"Multi-step forecast sample {i}")
    plt.xlabel("Horizon"); plt.legend()
    savefig(fig, f"multi_step_samples_{i}.png")

# 11) Ensemble comparison if available
if ensemble is not None:
    # compute metrics vs truth
    mae = mean_absolute_error(Y_test.flatten(), ensemble.flatten())
    rmse = math.sqrt(mean_squared_error(Y_test.flatten(), ensemble.flatten()))
    fig = plt.figure(figsize=(12,4))
    plt.plot(Y_test[:Nshow,0], label="True")
    plt.plot(ensemble[:Nshow,0], label="Ensemble")
    plt.title(f"Ensemble vs True h0 (MAE={mae:.3f} RMSE={rmse:.3f})")
    plt.legend()
    savefig(fig, "ensemble_vs_true_h0.png")

# 12) Save metrics summary CSV
summary_path = OUT_DIR / "metrics_summary_all.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model","overall_MAE","overall_RMSE","horizon_MAEs"])
    for name, data in models.items():
        overall_mae = mean_absolute_error(data["trues"].flatten(), data["preds"].flatten())
        overall_rmse = math.sqrt(mean_squared_error(data["trues"].flatten(), data["preds"].flatten()))
        per_mae = per_horizon_metrics(data["preds"], data["trues"])[0]
        writer.writerow([name, f"{overall_mae:.4f}", f"{overall_rmse:.4f}", "|".join([f"{v:.4f}" for v in per_mae])])
    if ensemble is not None:
        overall_mae = mean_absolute_error(Y_test.flatten(), ensemble.flatten())
        overall_rmse = math.sqrt(mean_squared_error(Y_test.flatten(), ensemble.flatten()))
        writer.writerow(["ENSEMBLE", f"{overall_mae:.4f}", f"{overall_rmse:.4f}", ""])
print("Saved metrics summary to", summary_path)

# 13) List generated files
print("\nGenerated files in", OUT_DIR.resolve())
for p in sorted(OUT_DIR.iterdir()):
    print(" -", p.name)

print("\nDone. Open the folder final_all_required_image/ to see all plots.")
