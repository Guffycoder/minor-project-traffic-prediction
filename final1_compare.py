# final_compare.py
import numpy as np, os, math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths - change if needed
gru_preds = "checkpoints_gru/test_preds.csv"
gru_trues = "checkpoints_gru/test_trues.csv"
lstm_preds = "checkpoints_lstm/test_preds.csv"
lstm_trues = "checkpoints_lstm/test_trues.csv"

if not os.path.exists(gru_preds):
    gru_preds = "checkpoints/test_preds.csv"; gru_trues = "checkpoints/test_trues.csv"
if not os.path.exists(lstm_preds):
    lstm_preds = "checkpoints/test_preds.csv"; lstm_trues = "checkpoints/test_trues.csv"

def load(p):
    if not os.path.exists(p): raise FileNotFoundError(p)
    return np.loadtxt(p, delimiter=",")
pg = load(gru_preds).reshape(-1)
pl = load(lstm_preds).reshape(-1)
if os.path.exists(gru_trues):
    t = load(gru_trues).reshape(-1)
elif os.path.exists(lstm_trues):
    t = load(lstm_trues).reshape(-1)
else:
    raise FileNotFoundError("No trues file found")

n = min(len(t), len(pg), len(pl))
t = t[:n]; pg = pg[:n]; pl = pl[:n]

def metrics(pred,true):
    mae = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    abs_err = np.abs(true - pred)
    acc1 = 100*np.mean(abs_err<=1); acc5 = 100*np.mean(abs_err<=5); acc10 = 100*np.mean(abs_err<=10)
    return mae, rmse, r2, acc1, acc5, acc10

mg = metrics(pg, t)
ml = metrics(pl, t)

print("GRU MAE RMSE R2 ±1 ±5 ±10:", mg)
print("LSTM MAE RMSE R2 ±1 ±5 ±10:", ml)

# Plot first N
N = min(500, n)
plt.figure(figsize=(14,5))
plt.plot(t[:N], label="True", linewidth=2)
plt.plot(pg[:N], label="GRU Pred", linestyle='--')
plt.plot(pl[:N], label="LSTM Pred", linestyle='-.')
plt.legend()
box = (
    f"GRU: MAE={mg[0]:.3f}, RMSE={mg[1]:.3f}, R2={mg[2]:.3f}\n"
    f"     ±1={mg[3]:.2f}%, ±5={mg[4]:.2f}%, ±10={mg[5]:.2f}%\n\n"
    f"LSTM:MAE={ml[0]:.3f}, RMSE={ml[1]:.3f}, R2={ml[2]:.3f}\n"
    f"     ±1={ml[3]:.2f}%, ±5={ml[4]:.2f}%, ±10={ml[5]:.2f}%"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
plt.gca().text(0.02, 0.98, box, transform=plt.gca().transAxes,
               fontsize=9, verticalalignment='top', bbox=props)
os.makedirs("checkpoints", exist_ok=True)
plt.savefig("checkpoints/final_compare.png", dpi=140)
plt.show()
