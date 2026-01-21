# make_all_plots.py
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

# -------------------------
OUTDIR = "ekdumfinal"
os.makedirs(OUTDIR, exist_ok=True)

# file locations (update if your files are in different places)
paths = {
    "multitask": {
        "pred": os.path.join("checkpoints_multi", "test_preds_reg_denorm.csv"),
        "true": os.path.join("checkpoints_multi", "test_trues_reg_denorm.csv"),
        "pred_cls": os.path.join("checkpoints_multi", "test_preds_clf.csv"),
        "true_cls": os.path.join("checkpoints_multi", "test_trues_clf.csv"),
    },
    "lstm": {
        "pred": os.path.join("checkpoints", "test_preds.csv"),
        "true": os.path.join("checkpoints", "test_trues.csv"),
    },
    "gru": {
        "pred": os.path.join("checkpoints", "test_preds.csv"),  # if GRU preds saved in different path, change here
        "true": os.path.join("checkpoints", "test_trues.csv"),
    }
}

# robust loader using pandas (handles headers and 1D/2D)
def load_csv_any(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, header=0)
    except Exception:
        df = pd.read_csv(path, header=None)
    arr = df.values
    if arr.ndim == 1 or arr.shape[1] == 1:
        return arr.reshape(-1)
    return arr

def flatten(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    return a.reshape(-1)

def safe_metrics(true, pred):
    t = flatten(true)
    p = flatten(pred)
    mae = mean_absolute_error(t, p)
    rmse = math.sqrt(mean_squared_error(t, p))
    r2 = r2_score(t, p)
    return mae, rmse, r2

def savefig(fig, name):
    p = os.path.join(OUTDIR, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("Saved:", p)
    plt.close(fig)

# -------------------------
# Plot utilities
# -------------------------
def plot_timeseries(true, pred, label, n=500):
    t = flatten(true)
    p = flatten(pred)
    n = min(n, len(t))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(range(n), t[:n], label="True", linewidth=1.2)
    ax.plot(range(n), p[:n], label=f"{label} Pred", linewidth=1.0, alpha=0.9)
    ax.set_title(f"Time series (first {n}) - {label}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.legend()
    savefig(fig, f"{label}_timeseries_first{n}.png")

def plot_scatter(true, pred, label):
    t = flatten(true)
    p = flatten(pred)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(t, p, s=6, alpha=0.6)
    mn = min(t.min(), p.min()); mx = max(t.max(), p.max())
    ax.plot([mn,mx],[mn,mx], '--', color='gray')
    ax.set_xlabel("True"); ax.set_ylabel("Pred")
    ax.set_title(f"True vs Pred scatter - {label}")
    savefig(fig, f"{label}_scatter_true_vs_pred.png")

def plot_residuals_hist(true, pred, label):
    t = flatten(true); p = flatten(pred)
    res = p - t
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(res, bins=80, density=False)
    ax.set_title(f"Residuals histogram (pred-true) - {label}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    savefig(fig, f"{label}_residuals_hist.png")

def plot_residuals_vs_true(true, pred, label):
    t = flatten(true); p = flatten(pred)
    res = p - t
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(t, res, s=6, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("True"); ax.set_ylabel("Residual (pred-true)")
    ax.set_title(f"Residuals vs True - {label}")
    savefig(fig, f"{label}_residuals_vs_true.png")

def plot_abs_error_boxplot(true, pred, label):
    t = flatten(true); p = flatten(pred)
    abs_err = np.abs(p - t)
    fig, ax = plt.subplots(figsize=(4,5))
    ax.boxplot(abs_err, vert=True, widths=0.6, patch_artist=True)
    ax.set_ylabel("Absolute error")
    ax.set_title(f"Absolute error boxplot - {label}")
    savefig(fig, f"{label}_abs_error_boxplot.png")

def plot_cumulative_abs_error(true, pred, label):
    t = flatten(true); p = flatten(pred)
    abs_err = np.abs(p - t)
    sorted_err = np.sort(abs_err)
    ecdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(sorted_err, ecdf, linewidth=1.2)
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Proportion <= x")
    ax.set_title(f"CDF of absolute error - {label}")
    ax.grid(True, linestyle=':', linewidth=0.6)
    # annotate where ±1 sits
    tol = 1.0
    prop = (sorted_err <= tol).mean()
    ax.axvline(tol, color='red', linestyle='--', label=f"±{tol}: {prop*100:.2f}%")
    ax.legend()
    savefig(fig, f"{label}_cumulative_abs_error.png")

def plot_error_ecdf(true, pred, label):
    # same as cumulative, but saved separately for naming
    plot_cumulative_abs_error(true, pred, label)  # reuse

# Combined plots
def combined_timeseries(trues_dict, preds_dict, n=500):
    # pick first model present as "True" source
    any_key = None
    for k in trues_dict:
        if trues_dict[k] is not None:
            any_key = k; break
    if any_key is None:
        print("No true arrays found for combined timeseries.")
        return
    t = flatten(trues_dict[any_key])
    n = min(n, len(t))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(range(n), t[:n], label="True", linewidth=1.2)
    for k,p in preds_dict.items():
        if p is None: continue
        ax.plot(range(n), flatten(p)[:n], label=f"{k} pred", alpha=0.9)
    ax.set_title(f"Combined Pred vs True (first {n})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.legend()
    savefig(fig, f"combined_timeseries_first{n}.png")

def combined_scatter(trues_dict, preds_dict):
    any_key = None
    for k in trues_dict:
        if trues_dict[k] is not None:
            any_key = k; break
    if any_key is None:
        print("No true arrays found for combined scatter.")
        return
    t = flatten(trues_dict[any_key])
    fig, ax = plt.subplots(figsize=(6,6))
    colors = ["C0","C1","C2","C3"]
    idx=0
    for k,p in preds_dict.items():
        if p is None: continue
        ax.scatter(t, flatten(p), s=6, alpha=0.5, label=k, c=colors[idx%len(colors)])
        idx += 1
    mn = min(t.min(), min((flatten(p).min() for p in preds_dict.values() if p is not None)))
    mx = max(t.max(), max((flatten(p).max() for p in preds_dict.values() if p is not None)))
    ax.plot([mn,mx],[mn,mx],'--', color='gray')
    ax.set_xlabel("True"); ax.set_ylabel("Pred")
    ax.set_title("Combined True vs Pred scatter")
    ax.legend()
    savefig(fig, f"combined_scatter_true_vs_pred.png")

# -------------------------
# MAIN: load and plot all
# -------------------------
preds = {}
trues = {}
for model_name, fdict in paths.items():
    ppath = fdict.get("pred")
    tpath = fdict.get("true")
    preds[model_name] = load_csv_any(ppath) if ppath else None
    trues[model_name] = load_csv_any(tpath) if tpath else None

# If LSTM and GRU use same folder and filenames, ensure you point correct files.
# (If GRU predictions are in a different file, change paths above.)

# Per-model plots
for name in preds.keys():
    p = preds[name]
    t = trues[name]
    if p is None or t is None:
        print(f"Skipping plots for {name} (missing files).")
        continue
    # metrics printed
    mae, rmse, r2 = safe_metrics = (mean_absolute_error(flatten(t), flatten(p)),
                                    math.sqrt(mean_squared_error(flatten(t), flatten(p))),
                                    r2_score(flatten(t), flatten(p)))
    print(f"{name} : MAE={safe_metrics[0]:.4f}, RMSE={safe_metrics[1]:.4f}, R2={safe_metrics[2]:.4f}")
    # generate plots
    plot_timeseries(t, p, name, n=500)
    plot_scatter(t, p, name)
    plot_residuals_hist(t, p, name)
    plot_residuals_vs_true(t, p, name)
    plot_abs_error_boxplot(t, p, name)
    plot_cumulative_abs_error(t, p, name)
    plot_error_ecdf(t, p, name)

# Combined plots
combined_timeseries(trues, preds, n=500)
combined_scatter(trues, preds)

# Confusion matrix for multitask (if exists)
mt_pred_cls = load_csv_any(paths["multitask"]["pred_cls"]) if os.path.exists(paths["multitask"]["pred_cls"]) else None
mt_true_cls = load_csv_any(paths["multitask"]["true_cls"]) if os.path.exists(paths["multitask"]["true_cls"]) else None

if mt_pred_cls is not None and mt_true_cls is not None:
    try:
        preds_cls = np.asarray(mt_pred_cls).reshape(-1).astype(int)
        trues_cls = np.asarray(mt_true_cls).reshape(-1).astype(int)
    except:
        preds_cls = np.loadtxt(paths["multitask"]["pred_cls"], dtype=int, delimiter=",")
        trues_cls = np.loadtxt(paths["multitask"]["true_cls"], dtype=int, delimiter=",")
    cm = confusion_matrix(trues_cls, preds_cls, labels=[0,1,2])
    # save ASCII + csv
    print("\nConfusion matrix (multitask):\n", cm)
    pd.DataFrame(cm, index=["true_0","true_1","true_2"], columns=["pred_0","pred_1","pred_2"]).to_csv(os.path.join(OUTDIR, "confusion_matrix_multitask.csv"))
    # image of confusion matrix (simple)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i,j])), ha="center", va="center", color="black")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_title("Confusion matrix (multitask)")
    savefig(fig, "confusion_matrix_multitask.png")
else:
    print("No classification files for multitask found, skipping confusion matrix image.")

print("\nAll plots saved to folder:", OUTDIR)
