# multi_step_prediction_plots.py
import os, numpy as np, matplotlib.pyplot as plt
from math import ceil

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
        print("Loaded", name, p.shape, t.shape)
    except Exception as e:
        print("Skip", name, e)

# use min horizon across loaded models
K = min(p.shape[1] for p,t in data.values())
N = min(p.shape[0] for p,t in data.values())
Nplot = min(500, N)

for k in range(K):
    plt.figure(figsize=(14,4))
    plt.title(f"Step {k} — first {Nplot} points")
    plt.plot(range(Nplot), data[list(data.keys())[0]][1][:Nplot,k], label="True (step {})".format(k), linewidth=2)
    for name,(p,t) in data.items():
        plt.plot(range(Nplot), p[:Nplot,k], label=f"{name} Pred (step {k})", linewidth=1)
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Total traffic")
    plt.grid(alpha=0.2)
    out = f"checkpoints/pred_step_{k}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("Saved", out)
