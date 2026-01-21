# ensemble_avg.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

p_gru = np.loadtxt("checkpoints_gru/test_preds.csv", delimiter=",")
p_lstm = np.loadtxt("checkpoints_lstm/test_preds.csv", delimiter=",")
p_hyb = np.loadtxt("checkpoints_hybrid/test_preds.csv", delimiter=",")
t = np.loadtxt("checkpoints_gru/test_trues.csv", delimiter=",")

# Ensure same shapes and align K
K = min(p_gru.shape[1], p_lstm.shape[1], p_hyb.shape[1], t.shape[1])
p_gru = p_gru[:, :K]; p_lstm = p_lstm[:, :K]; p_hyb = p_hyb[:, :K]; t = t[:, :K]

ens = (p_gru + p_lstm + p_hyb) / 3.0

mae = mean_absolute_error(t.flatten(), ens.flatten())
rmse = mean_squared_error(t.flatten(), ens.flatten())**0.5
r2 = r2_score(t.flatten(), ens.flatten())
print("Ensemble avg MAE:", mae, "RMSE:", rmse, "R2:", r2)

# Save ensemble preds
np.savetxt("checkpoints/ensemble_avg_preds.csv", ens, delimiter=",")
print("Saved checkpoints/ensemble_avg_preds.csv")
