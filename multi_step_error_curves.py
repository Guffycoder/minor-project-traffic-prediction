# multi_step_error_curves.py
import os, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("checkpoints", exist_ok=True)

# helper to load preds/trues CSV (either shape (N,) (N,1) or (N,K))
def load_csv(path):
    a = np.loadtxt(path, delimiter=",")
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a

models = {
    "GRU": ("checkpoints_gru/test_preds.csv", "checkpoints_gru/test_trues.csv"),
    "LSTM":("checkpoints_lstm/test_preds.csv","checkpoints_lstm/test_trues.csv"),
    "MULTI":("checkpoints_multi/test_preds_reg.csv","checkpoints_multi/test_trues_reg.csv"),
    "HYBRID":("checkpoints_hybrid/test_preds.csv","checkpoints_hybrid/test_trues.csv"),
}

# load data (skip models whose files missing)
data = {}
for name,(p_pred,p_true) in models.items():
    try:
        preds = load_csv(p_pred)
        trues = load_csv(p_true)
        # if preds columns differ from trues, try to align:
        if preds.shape[0] != trues.shape[0]:
            raise ValueError(f"Row mismatch for {name}")
        data[name] = (preds, trues)
        print("Loaded", name, preds.shape, trues.shape)
    except Exception as e:
        print("Skipping", name, "->", e)

if not data:
    raise SystemExit("No model data loaded.")

# decide K (prediction horizon). take min across models
Ks = [preds.shape[1] for preds,trues in data.values()]
K = int(min(Ks))
print("Using horizon K =", K)

# compute metrics per horizon
horizons = list(range(K))
plt.figure(figsize=(10,5))
for metric_name, metric_fn in [("MAE", lambda t,p: mean_absolute_error(t,p)),
                               ("RMSE", lambda t,p: math.sqrt(mean_squared_error(t,p))),
                               ("R2", lambda t,p: r2_score(t,p))]:
    plt.clf()
    for name,(preds,trues) in data.items():
        # ensure preds/trues have K columns
        p = preds[:, :K]
        t = trues[:, :K]
        vals = []
        for k in horizons:
            vals.append(metric_fn(t[:,k].flatten(), p[:,k].flatten()))
        plt.plot(horizons, vals, marker='o', label=name)
    plt.xlabel("Horizon step (0 = next step)")
    plt.xticks(horizons)
    plt.grid(alpha=0.3)
    plt.title(f"{metric_name} by horizon")
    plt.legend()
    plt.tight_layout()
    out = f"checkpoints/multi_step_{metric_name.lower()}.png"
    plt.savefig(out, dpi=150)
    print("Saved", out)
