# ensemble_weighted.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

files = {
    "gru":"checkpoints_gru/test_preds.csv",
    "lstm":"checkpoints_lstm/test_preds.csv",
    "hyb":"checkpoints_hybrid/test_preds.csv"
}
t = np.loadtxt("checkpoints_gru/test_trues.csv", delimiter=",")

preds = {k: np.loadtxt(v, delimiter=",") for k,v in files.items()}
# align shapes
K = min(preds['gru'].shape[1], preds['lstm'].shape[1], preds['hyb'].shape[1], t.shape[1])
for k in preds: preds[k] = preds[k][:,:K]
t = t[:,:K]

# compute per-model RMSE (flattened)
rmses = {k: mean_squared_error(t.flatten(), preds[k].flatten())**0.5 for k in preds}
inv = {k: 1.0/r for k,r in rmses.items()}
total = sum(inv.values())
weights = {k: inv[k]/total for k in inv}
print("Model weights:", weights)

ens = sum(preds[k]*weights[k] for k in preds)
mae = mean_absolute_error(t.flatten(), ens.flatten())
rmse = mean_squared_error(t.flatten(), ens.flatten())**0.5
r2 = r2_score(t.flatten(), ens.flatten())
print("Weighted ensemble MAE:", mae, "RMSE:", rmse, "R2:", r2)
