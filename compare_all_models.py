# compare_all_models.py
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
os.makedirs("checkpoints", exist_ok=True)

def load_csv(path):
    a = np.loadtxt(path, delimiter=",")
    if a.ndim == 1:
        a = a.reshape(-1,1)
    return a

models = {
    "GRU": ("checkpoints_gru/test_preds.csv", "checkpoints_gru/test_trues.csv"),
    "LSTM":("checkpoints_lstm/test_preds.csv","checkpoints_lstm/test_trues.csv"),
    "MULTI":("checkpoints_multi/test_preds_reg.csv","checkpoints_multi/test_trues_reg.csv"),
    "HYBRID":("checkpoints_hybrid/test_preds.csv","checkpoints_hybrid/test_trues.csv"),
}

data = {}
for name,(pp,pt) in models.items():
    try:
        p = load_csv(pp); t = load_csv(pt)
        data[name] = (p,t)
    except Exception as e:
        print("skip", name, e)

K = min(p.shape[1] for p,t in data.values())
N = min(p.shape[0] for p,t in data.values())
Nplot = min(500, N)

# MAE per horizon
plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((3,2),(0,0), colspan=2)
for name,(p,t) in data.items():
    maes = [mean_absolute_error(t[:,k], p[:,k]) for k in range(K)]
    ax1.plot(range(K), maes, marker='o', label=name)
ax1.set_title("MAE by horizon"); ax1.set_xlabel("Horizon"); ax1.set_ylabel("MAE"); ax1.legend(); ax1.grid(alpha=0.3)

# time-series (first 500), horizon 0
ax2 = plt.subplot2grid((3,2),(1,0), colspan=1)
ax3 = plt.subplot2grid((3,2),(1,1), colspan=1)
for ax in (ax2,):
    ax.plot(data[list(data.keys())[0]][1][:Nplot,0], label="True (step0)", linewidth=2)
for name,(p,t) in data.items():
    ax2.plot(p[:Nplot,0], label=f"{name} pred", alpha=0.9)
ax2.set_title("First 500: True vs preds (step0)"); ax2.legend(); ax2.grid(alpha=0.2)

# small metrics table (horizon 0) + hist
rows = []
for name,(p,t) in data.items():
    pred0 = p[:,0].flatten(); true0 = t[:,0].flatten()
    mae = mean_absolute_error(true0, pred0)
    rmse = (mean_squared_error(true0, pred0))**0.5
    r2 = r2_score(true0, pred0)
    rows.append((name, mae, rmse, r2))
text = "\n".join([f"{r[0]}: MAE={r[1]:.3f}, RMSE={r[2]:.3f}, R2={r[3]:.3f}" for r in rows])
ax3.axis('off')
ax3.text(0,0.5, text, fontsize=10, family='monospace')
plt.tight_layout()
out = "checkpoints/compare_all_models.png"
plt.savefig(out, dpi=150)
print("Saved", out)
