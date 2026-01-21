# analyze_hybrid.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

p = np.loadtxt("checkpoints_hybrid/test_preds.csv", delimiter=",")
t = np.loadtxt("checkpoints_hybrid/test_trues.csv", delimiter=",")

print("Shapes:", p.shape, t.shape)
K = p.shape[1]
maes = []
rmses = []
for k in range(K):
    m = mean_absolute_error(t[:,k], p[:,k])
    r = mean_squared_error(t[:,k], p[:,k])
    r = r**0.5
    maes.append(m); rmses.append(r)
    print(f"Horizon {k}: MAE={m:.4f}, RMSE={r:.4f}")

print("Overall flattened MAE:", mean_absolute_error(t.flatten(), p.flatten()))
print("Overall flattened RMSE:", mean_squared_error(t.flatten(), p.flatten())**0.5)

# residual histogram for horizon 0
res0 = p[:,0] - t[:,0]
plt.figure(figsize=(8,4))
plt.hist(res0, bins=80)
plt.title("Residual histogram (horizon 0)")
plt.tight_layout()
plt.savefig("checkpoints_hybrid/resid_h0.png")
print("Saved resid_h0.png")

# per-horizon MAE plot
plt.figure(figsize=(8,4))
plt.plot(range(K), maes, marker='o')
plt.xlabel("Horizon")
plt.ylabel("MAE")
plt.title("Hybrid per-horizon MAE")
plt.tight_layout()
plt.savefig("checkpoints_hybrid/mae_by_horizon.png")
print("Saved mae_by_horizon.png")
