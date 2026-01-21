#!/usr/bin/env python3
"""
plots_one_shot.py
Robust one-shot plotting script for your project.
Generates many diagnostic plots into ./final_all_required_image/

Run:
    python plots_one_shot.py
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

SEQ_NPZ = Path("sequences_Tin12_pred1.npz")  # your dataset file (change if needed)
CHECKPOINT_FOLDERS = {
    "GRU": Path("checkpoints_gru"),
    "LSTM": Path("checkpoints_lstm"),
    "HYBRID": Path("checkpoints_hybrid"),
    "MULTI": Path("checkpoints_multi"),
}
ENSEMBLE_AVG = Path("checkpoints/ensemble_avg_preds.csv")   # optional

# ---------------- helpers ----------------
def collapse_to_NK(arr):
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
        raise FileNotFoundError(f"{npz_path} not found.")
    d = dict(np.load(npz_path, allow_pickle=True))
    # try common names
    for candidate in ("Y_test","y_test","Ytest","Y_test_reg","Yreg_test"):
        if candidate in d:
            Y_test_raw = d[candidate]; return collapse_to_NK(Y_test_raw), d
    # else pick first key starting with 'Y' or 'y'
    for k in d.keys():
        if k.lower().startswith("y"):
            return collapse_to_NK(d[k]), d
    # fallback: if X_test present, infer nothing
    raise KeyError(f"No Y_test-like key in {npz_path}. Keys: {list(d.keys())}")

def try_load_preds(folder: Path):
    if not folder.exists():
        return None, None
    candidates = [
        ("test_preds.csv", "test_trues.csv"),
        ("test_preds_reg.csv","test_trues_reg.csv"),
        ("test_preds_no_denorm.csv","test_trues_no_denorm.csv"),
        ("preds.csv","trues.csv"),
    ]
    for pfile, tfile in candidates:
        ppth = folder / pfile
        tpth = folder / tfile
        if ppth.exists() and tpth.exists():
            try:
                p = np.loadtxt(ppth, delimiter=","); t = np.loadtxt(tpth, delimiter=",")
                return np.atleast_2d(p), np.atleast_2d(t)
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

def match_pred_to_truth(preds, truth):
    """
    Align preds (Np, Kp) and truth (Nt, Kt):
      - crop/pad horizons to match Kt
      - rows: use first min(Np,Nt) rows
    Returns (preds_aligned, truth_aligned)
    """
    preds = np.atleast_2d(preds)
    truth = np.atleast_2d(truth)
    Np, Kp = preds.shape
    Nt, Kt = truth.shape

    # Align horizons
    if Kp == Kt:
        preds2 = preds.copy()
    elif Kp > Kt:
        preds2 = preds[:, :Kt]
    else:
        # Kp < Kt -> tile last column
        last = preds[:, -1].reshape(-1,1)
        missing = Kt - Kp
        pad = np.tile(last, (1, missing))
        preds2 = np.concatenate([preds, pad], axis=1)

    # Align rows: take first min rows
    M = min(preds2.shape[0], truth.shape[0])
    if M != preds2.shape[0] or M != truth.shape[0]:
        print(f"Warning: row-count mismatch: preds {preds2.shape[0]} vs truth {truth.shape[0]} -> using first {M} rows")
    return preds2[:M, :], truth[:M, :]

def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)

# ---------------- load truth ----------------
print("Loading Y_test from:", SEQ_NPZ)
Y_test, npzdict = load_npz_Y_test(SEQ_NPZ)
N_true, K_true = Y_test.shape
print("Y_test shape:", Y_test.shape)
means = npzdict.get("means", None); stds = npzdict.get("stds", None)
if means is not None and stds is not None:
    print("Found means/stds in npz.")

# ---------------- load preds ----------------
models = {}
for name, folder in CHECKPOINT_FOLDERS.items():
    p, t = try_load_preds(folder)
    if p is None:
        print(f"{name}: no preds found in {folder} (skipping)")
        continue
    # if folder had its own trues, use them; else use global Y_test
    if t is None:
        t = Y_test
    # collapse any higher dims
    t = collapse_to_NK(t)
    p = np.atleast_2d(p)
    try:
        p_al, t_al = match_pred_to_truth(p, t)
    except Exception as e:
        print(f"Could not align {name}: {e}")
        continue
    models[name] = {"preds": p_al, "trues": t_al}
    print(f"Loaded {name} preds shape {p_al.shape}")

# optional ensemble
ensemble = None
if ENSEMBLE_AVG.exists():
    ensemble = np.loadtxt(ENSEMBLE_AVG, delimiter=","); ensemble = np.atleast_2d(ensemble)
    # align to global truth
    try:
        ensemble, Y_test_al = match_pred_to_truth(ensemble, Y_test)
        ensemble = ensemble
        print("Loaded ensemble avg shape:", ensemble.shape)
    except Exception as e:
        print("Could not align ensemble:", e)
        ensemble = None

# if no models loaded -> exit
if len(models) == 0:
    print("No model predictions found. Put test_preds.csv & test_trues.csv into checkpoint folders and re-run.")
    raise SystemExit(0)

# ---------------- metrics helpers ----------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def per_horizon_metrics(preds, trues):
    K = trues.shape[1]
    mae = [mean_absolute_error(trues[:,i], preds[:,i]) for i in range(K)]
    rmse = [math.sqrt(mean_squared_error(trues[:,i], preds[:,i])) for i in range(K)]
    r2 = []
    for i in range(K):
        try:
            r2.append(r2_score(trues[:,i], preds[:,i]))
        except Exception:
            r2.append(float("nan"))
    return np.array(mae), np.array(rmse), np.array(r2)

# ---------------- PLOTS ----------------
# 1. heatmap true
fig = plt.figure(figsize=(10,6))
plt.imshow(Y_test, aspect="auto")
plt.colorbar(); plt.title("True targets heatmap (samples × horizon)")
savefig(fig, "heatmap_true.png")

# 2. true dist by horizon
fig = plt.figure(figsize=(12,4))
for i in range(min(4, K_true)):
    plt.subplot(1, min(4,K_true), i+1)
    plt.hist(Y_test[:,i], bins=30)
    plt.title(f"h{i}")
savefig(fig, "dist_true_by_horizon.png")

# 3. true vs models (h0 first N)
step = 0
Nshow = min(500, Y_test.shape[0])
fig = plt.figure(figsize=(14,5))
plt.plot(Y_test[:Nshow, step], label="True", linewidth=2)
for name, data in models.items():
    plt.plot(data["preds"][:Nshow, step], label=name)
if ensemble is not None:
    plt.plot(ensemble[:Nshow, step], label="Ensemble", linestyle="--")
plt.legend(); plt.title(f"True vs Models (first {Nshow}) h{step}")
savefig(fig, "true_vs_models_step0.png")

# 4. multi-step example sample 0 (handles K mismatch)
sample_idx = 0
fig = plt.figure(figsize=(8,4))
plt.plot(range(K_true), Y_test[sample_idx, :], marker='o', label="True")
for name, data in models.items():
    pred_row = data["preds"][sample_idx]
    pred_row = pred_row[:K_true]
    plt.plot(range(K_true), pred_row, marker='x', label=name)
if ensemble is not None:
    plt.plot(range(K_true), ensemble[sample_idx,:K_true], marker='s', label="Ensemble")
plt.xlabel("Horizon"); plt.ylabel("Value"); plt.title(f"Multi-step forecast example (sample {sample_idx})"); plt.legend()
savefig(fig, "multi_step_example_sample0.png")

# 5. per-horizon MAE/RMSE/R2 for models
fig = plt.figure(figsize=(10,5))
for name,data in models.items():
    mae, rmse, r2 = per_horizon_metrics(data["preds"], data["trues"])
    plt.plot(range(len(mae)), mae, marker='o', label=f"{name} MAE")
plt.xlabel("Horizon"); plt.ylabel("MAE"); plt.legend(); plt.title("Per-horizon MAE")
savefig(fig, "mae_by_horizon_all_models.png")

fig = plt.figure(figsize=(10,5))
for name,data in models.items():
    mae, rmse, r2 = per_horizon_metrics(data["preds"], data["trues"])
    plt.plot(range(len(rmse)), rmse, marker='o', label=f"{name} RMSE")
plt.xlabel("Horizon"); plt.ylabel("RMSE"); plt.legend(); plt.title("Per-horizon RMSE")
savefig(fig, "rmse_by_horizon_all_models.png")

fig = plt.figure(figsize=(10,5))
for name,data in models.items():
    mae, rmse, r2 = per_horizon_metrics(data["preds"], data["trues"])
    plt.plot(range(len(r2)), r2, marker='o', label=f"{name} R2")
plt.xlabel("Horizon"); plt.ylabel("R2"); plt.legend(); plt.title("Per-horizon R2")
savefig(fig, "r2_by_horizon_all_models.png")

# 6. residual heatmaps per model
for name, data in models.items():
    resid = (data["preds"] - data["trues"])
    fig = plt.figure(figsize=(8,5))
    vmax = np.max(np.abs(resid))
    plt.imshow(resid, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    plt.colorbar(); plt.title(f"Residual heatmap {name} (pred - true)")
    savefig(fig, f"resid_heatmap_{name}.png")

# 7. residual hist & scatter (h0)
for name,data in models.items():
    resid = (data["preds"] - data["trues"])
    fig = plt.figure(figsize=(6,4)); plt.hist(resid[:,0], bins=40); plt.title(f"{name} resid hist h0"); savefig(fig, f"{name}_resid_hist_h0.png")
    fig = plt.figure(figsize=(5,5)); plt.scatter(data["trues"][:,0], data["preds"][:,0], s=6, alpha=0.6)
    mn = min(data["trues"][:,0].min(), data["preds"][:,0].min()); mx = max(data["trues"][:,0].max(), data["preds"][:,0].max())
    plt.plot([mn,mx],[mn,mx], color='red'); plt.xlabel("True h0"); plt.ylabel(f"{name} pred h0"); savefig(fig, f"{name}_scatter_h0.png")

# 8. cumulative error per model
for name,data in models.items():
    abs_err = np.abs(data["preds"] - data["trues"]).mean(axis=1)
    cum = np.cumsum(np.sort(abs_err))
    fig = plt.figure(figsize=(8,4)); plt.plot(cum); plt.title(f"Cumulative sorted mean-abs-error {name}"); savefig(fig, f"{name}_cumulative_error.png")

# 9. small multiples (first 5 samples)
for i in range(min(5, Y_test.shape[0])):
    fig = plt.figure(figsize=(8,3))
    plt.plot(range(K_true), Y_test[i,:K_true], marker='o', label="True")
    for name,data in models.items():
        plt.plot(range(K_true), data["preds"][i,:K_true], marker='x', label=name)
    if ensemble is not None:
        plt.plot(range(K_true), ensemble[i,:K_true], marker='s', label="Ensemble")
    plt.title(f"Multi-step sample {i}"); plt.legend(); savefig(fig, f"multi_step_sample_{i}.png")

# 10. save metrics summary
summary_path = OUT_DIR / "metrics_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f); writer.writerow(["model","overall_MAE","overall_RMSE","per_horizon_MAE"])
    for name,data in models.items():
        overall_mae = mean_absolute_error(data["trues"].flatten(), data["preds"].flatten())
        overall_rmse = math.sqrt(mean_squared_error(data["trues"].flatten(), data["preds"].flatten()))
        per_mae = per_horizon_metrics(data["preds"], data["trues"])[0]
        writer.writerow([name, f"{overall_mae:.4f}", f"{overall_rmse:.4f}", "|".join([f"{v:.4f}" for v in per_mae])])
    if ensemble is not None:
        overall_mae = mean_absolute_error(Y_test.flatten(), ensemble.flatten())
        overall_rmse = math.sqrt(mean_squared_error(Y_test.flatten(), ensemble.flatten()))
        writer.writerow(["ENSEMBLE", f"{overall_mae:.4f}", f"{overall_rmse:.4f}", ""])
print("Saved metrics summary to", summary_path)

# List outputs
print("\nGenerated files in", OUT_DIR.resolve())
for p in sorted(OUT_DIR.iterdir()):
    print(" -", p.name)

print("\nDone.")
