# distribution_plots.py
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
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

# choose horizon 0
k = 0
plt.figure(figsize=(10,4))
for name,(p,t) in data.items():
    pred = p[:,k].flatten(); true = t[:,k].flatten()
    plt.hist(true, bins=40, alpha=0.3, label=f"{name} True", density=True)
    plt.hist(pred, bins=40, alpha=0.3, label=f"{name} Pred", density=True)
plt.legend(); plt.title("Distribution: true vs pred (horizon 0)"); plt.xlabel("Traffic"); plt.tight_layout()
plt.savefig("checkpoints/dist_step0.png", dpi=150)
print("Saved checkpoints/dist_step0.png")

# error histograms (pred-true)
plt.figure(figsize=(10,4))
for name,(p,t) in data.items():
    err = (p[:,k].flatten() - t[:,k].flatten())
    plt.hist(err, bins=60, alpha=0.4, label=name)
plt.legend(); plt.title("Error distribution (pred - true) horizon 0"); plt.tight_layout()
plt.savefig("checkpoints/error_hist_step0.png", dpi=150)
print("Saved checkpoints/error_hist_step0.png")
