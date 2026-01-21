# visualize_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

multi_dir = "checkpoints_multi"
lstm_dir = "checkpoints"
gru_dir = "checkpoints"

# -----------------------
# Robust CSV loader
# -----------------------
def safe_load(path):
    if not os.path.exists(path):
        print("Missing:", path)
        return None
    try:
        df = pd.read_csv(path, header=0)
        arr = df.values
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr
    except:
        try:
            arr = pd.read_csv(path, header=None).values
            if arr.shape[1] == 1:
                return arr[:, 0]
            return arr
        except Exception as e2:
            print("Error loading:", path, e2)
            return None

# load predictions
multi_preds = safe_load(os.path.join(multi_dir, "test_preds_reg_denorm.csv"))
multi_trues = safe_load(os.path.join(multi_dir, "test_trues_reg_denorm.csv"))

lstm_preds = safe_load(os.path.join(lstm_dir, "test_preds.csv"))
lstm_trues = safe_load(os.path.join(lstm_dir, "test_trues.csv"))

gru_preds  = safe_load(os.path.join(gru_dir, "test_preds.csv"))
gru_trues  = safe_load(os.path.join(gru_dir, "test_trues.csv"))

def flat(a):
    if a is None:
        return None
    a = np.asarray(a)
    return a.reshape(-1)

results = {}

# -----------------------
# Create comparison table
# -----------------------
for name, p, t in [
    ("Multitask", multi_preds, multi_trues),
    ("LSTM", lstm_preds, lstm_trues),
    ("GRU",  gru_preds,  gru_trues)
]:
    if p is None or t is None:
        print(f"Skipping {name} - missing data")
        continue

    p1 = flat(p)
    t1 = flat(t)

    mae = mean_absolute_error(t1, p1)
    rmse = math.sqrt(mean_squared_error(t1, p1))   # FIXED: compatible RMSE
    r2 = r2_score(t1, p1)
    acc_abs = ((np.abs(t1 - p1) <= 1.0).mean()) * 100.0

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "AccAbs(±1)": acc_abs
    }

if results:
    df = pd.DataFrame(results).T
    df.to_csv("model_comparison_table.csv")
    print("\nMODEL COMPARISON TABLE:\n", df)
else:
    print("No valid results.")

# -----------------------
# Time-series plot
# -----------------------
if multi_preds is not None and multi_trues is not None:
    n = 300
    plt.figure(figsize=(12,4))
    plt.plot(flat(multi_trues)[:n], label="True")
    plt.plot(flat(multi_preds)[:n], label="Multitask", alpha=0.8)
    if lstm_preds is not None:
        plt.plot(flat(lstm_preds)[:n], label="LSTM", alpha=0.8)
    if gru_preds is not None:
        plt.plot(flat(gru_preds)[:n], label="GRU", alpha=0.8)
    plt.legend()
    plt.title("Pred vs True (first 300 samples)")
    plt.savefig("pred_vs_true_timeseries.png", dpi=200)
    print("Saved: pred_vs_true_timeseries.png")
    plt.close()

# -----------------------
# Scatter plot (Multitask)
# -----------------------
if multi_preds is not None and multi_trues is not None:
    plt.figure(figsize=(5,5))
    plt.scatter(flat(multi_trues), flat(multi_preds), s=6)
    mn = min(flat(multi_trues).min(), flat(multi_preds).min())
    mx = max(flat(multi_trues).max(), flat(multi_preds).max())
    plt.plot([mn,mx],[mn,mx], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title("True vs Pred (Multitask)")
    plt.savefig("true_vs_pred_scatter_multitask.png", dpi=200)
    print("Saved: true_vs_pred_scatter_multitask.png")
    plt.close()

# -----------------------
# Confusion Matrix (ASCII)
# -----------------------
pred_cls_path = os.path.join(multi_dir, "test_preds_clf.csv")
true_cls_path = os.path.join(multi_dir, "test_trues_clf.csv")

if os.path.exists(pred_cls_path) and os.path.exists(true_cls_path):
    try:
        preds_cls = pd.read_csv(pred_cls_path, header=0).values.reshape(-1).astype(int)
        trues_cls = pd.read_csv(true_cls_path, header=0).values.reshape(-1).astype(int)
    except:
        preds_cls = np.loadtxt(pred_cls_path, delimiter=",", dtype=int)
        trues_cls = np.loadtxt(true_cls_path, delimiter=",", dtype=int)

    cm = confusion_matrix(trues_cls, preds_cls, labels=[0,1,2])
    print("\nCONFUSION MATRIX (Multitask):")
    print(cm)

    pd.DataFrame(cm, index=["true_0","true_1","true_2"],
                     columns=["pred_0","pred_1","pred_2"]).to_csv("confusion_matrix_multitask.csv")
    print("Saved: confusion_matrix_multitask.csv")
else:
    print("\nNo classification files found. Skipping confusion matrix.")
