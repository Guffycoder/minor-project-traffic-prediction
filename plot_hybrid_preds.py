# plot_hybrid_preds.py
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

preds_path = "checkpoints_hybrid/test_preds.csv"
trues_path = "checkpoints_hybrid/test_trues.csv"

assert os.path.exists(preds_path), preds_path + " missing"
assert os.path.exists(trues_path), trues_path + " missing"

preds = np.loadtxt(preds_path, delimiter=",")
trues = np.loadtxt(trues_path, delimiter=",")

# If multi-step, flatten to 1D series by comparing each step concatenated or compare only first step:
# We'll compute metrics for the first prediction step and also overall flatten.
if preds.ndim == 1:
    p_first = preds
else:
    p_first = preds[:, 0]   # first-step predictions
t_first = trues[:, 0] if trues.ndim > 1 else trues

def metrics(pred, true):
    mae = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    abs_err = np.abs(true - pred)
    acc1 = 100.0 * (abs_err <= 1).mean()
    acc5 = 100.0 * (abs_err <= 5).mean()
    acc10 = 100.0 * (abs_err <= 10).mean()
    # relative within 10% on true>0
    mask = true > 0
    rel10 = 100.0 * ((np.abs((pred[mask] - true[mask]) / true[mask]) <= 0.10).mean()) if mask.sum() > 0 else float('nan')
    return mae, rmse, r2, acc1, acc5, acc10, rel10

mae, rmse, r2, acc1, acc5, acc10, rel10 = metrics(p_first, t_first)
print("First-step metrics:")
print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
print(f"Absolute tol accuracies: ±1={acc1:.2f}%, ±5={acc5:.2f}%, ±10={acc10:.2f}%")
print(f"Relative within 10% (true>0): {rel10:.2f}%")

# Plot first N points
N = min(500, len(t_first))
plt.figure(figsize=(12,4))
plt.plot(t_first[:N], label="True (step 1)")
plt.plot(p_first[:N], label="Pred (step 1)")
plt.legend()
plt.title("Hybrid: True vs Pred (first step) — first {} points".format(N))
plt.savefig("checkpoints_hybrid/hybrid_firststep_plot.png", dpi=150)
plt.show()
print("Saved plot to checkpoints_hybrid/hybrid_firststep_plot.png")
