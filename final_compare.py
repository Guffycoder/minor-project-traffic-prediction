# final_compare.py
import numpy as np
import os, math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Paths ===
gru_preds = "checkpoints_gru/test_preds.csv"
gru_trues = "checkpoints_gru/test_trues.csv"

lstm_preds = "checkpoints_lstm/test_preds.csv"
lstm_trues = "checkpoints_lstm/test_trues.csv"

# fallback if earlier saved in "checkpoints"
if not os.path.exists(gru_preds):
    gru_preds = "checkpoints/test_preds.csv"
    gru_trues = "checkpoints/test_trues.csv"

if not os.path.exists(lstm_preds):
    lstm_preds = "checkpoints/test_preds.csv"
    lstm_trues = "checkpoints/test_trues.csv"

# Load arrays
def load_arr(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.loadtxt(path, delimiter=",")

pred_gru = load_arr(gru_preds).flatten()
pred_lstm = load_arr(lstm_preds).flatten()

if os.path.exists(gru_trues):
    true = load_arr(gru_trues).flatten()
elif os.path.exists(lstm_trues):
    true = load_arr(lstm_trues).flatten()
else:
    raise FileNotFoundError("No ground-truth file found.")

# Align lengths
n = min(len(true), len(pred_gru), len(pred_lstm))
true = true[:n]
pred_gru = pred_gru[:n]
pred_lstm = pred_lstm[:n]

# ===== METRICS =====
def compute_metrics(pred, true):
    mae = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    abs_err = np.abs(pred - true)
    acc1 = 100 * np.mean(abs_err <= 1)
    acc5 = 100 * np.mean(abs_err <= 5)
    acc10 = 100 * np.mean(abs_err <= 10)

    return mae, rmse, r2, acc1, acc5, acc10

m_gru = compute_metrics(pred_gru, true)
m_lstm = compute_metrics(pred_lstm, true)

mae_g, rmse_g, r2_g, acc1_g, acc5_g, acc10_g = m_gru
mae_l, rmse_l, r2_l, acc1_l, acc5_l, acc10_l = m_lstm

# ===== PRINT RESULTS =====
print("\n===== GRU METRICS =====")
print(f"MAE  = {mae_g:.3f}")
print(f"RMSE = {rmse_g:.3f}")
print(f"R²   = {r2_g:.3f}")
print(f"Acc ±1  = {acc1_g:.2f}%")
print(f"Acc ±5  = {acc5_g:.2f}%")
print(f"Acc ±10 = {acc10_g:.2f}%")

print("\n===== LSTM METRICS =====")
print(f"MAE  = {mae_l:.3f}")
print(f"RMSE = {rmse_l:.3f}")
print(f"R²   = {r2_l:.3f}")
print(f"Acc ±1  = {acc1_l:.2f}%")
print(f"Acc ±5  = {acc5_l:.2f}%")
print(f"Acc ±10 = {acc10_l:.2f}%")

# ===== PLOT =====
N = min(500, n)
plt.figure(figsize=(14,5))
plt.plot(true[:N], label="True", linewidth=2)
plt.plot(pred_gru[:N], label="GRU Pred", linestyle='--')
plt.plot(pred_lstm[:N], label="LSTM Pred", linestyle='-.')
plt.legend()
plt.title(f"First {N} Test Predictions: True vs GRU vs LSTM")
plt.xlabel("Sample index")
plt.ylabel("Total Traffic")
plt.grid(alpha=0.3)

# Add metrics box
box = (
    f"GRU:  MAE={mae_g:.3f}, RMSE={rmse_g:.3f}, R²={r2_g:.3f}\n"
    f"      Acc ±1={acc1_g:.2f}%, ±5={acc5_g:.2f}%, ±10={acc10_g:.2f}%\n\n"
    f"LSTM: MAE={mae_l:.3f}, RMSE={rmse_l:.3f}, R²={r2_l:.3f}\n"
    f"      Acc ±1={acc1_l:.2f}%, ±5={acc5_l:.2f}%, ±10={acc10_l:.2f}%"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
plt.gca().text(0.02, 0.98, box, transform=plt.gca().transAxes,
               fontsize=9, verticalalignment='top', bbox=props)

# Save output
os.makedirs("checkpoints", exist_ok=True)
plt.savefig("checkpoints/final_compare.png", dpi=140)
plt.show()
