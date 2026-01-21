# inspect_preds.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math, os

P = "checkpoints/test_preds.csv"
T = "checkpoints/test_trues.csv"

assert os.path.exists(P), f"{P} missing"
assert os.path.exists(T), f"{T} missing"

preds = np.loadtxt(P, delimiter=",")
trues = np.loadtxt(T, delimiter=",")

print("Shapes:", preds.shape, trues.shape)
print("Preds range:", preds.min(), preds.max(), "Trues range:", trues.min(), trues.max())
print("Preds mean/std:", preds.mean(), preds.std(), "Trues mean/std:", trues.mean(), trues.std())

mae = mean_absolute_error(trues, preds)
rmse = math.sqrt(mean_squared_error(trues, preds))
print(f"MAE = {mae:.4f}, RMSE = {rmse:.4f}")

# baseline: persistence (predict last observed value)
# We need the test inputs to compute persistence baseline. We'll try a simple baseline: predict median of training Y if available.
# If you have processed npz nearby, load to compute persistence properly:
npz_path = "D:/minor final/sequences_Tin12_pred1.npz"
if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X_test = data["X_test"]
    Y_test = data["Y_test"].flatten()
    # persistence baseline: predict last value of each input sequence (denormalizing if needed)
    # X_test contains normalized inputs. We will check whether trues match raw sums; assume Y_test is raw.
    last_vals = X_test[:, -1, :]  # normalized last values
    # try to denormalize using means/std if available
    if "means" in data and "stds" in data:
        means = data["means"].astype(float)
        stds = data["stds"].astype(float)
        # If multiple cells, sum them for total baseline
        # For single cell:
        baseline = (last_vals * stds) + means
        baseline_sum = baseline.sum(axis=1)
    else:
        baseline_sum = last_vals.sum(axis=1)
    base_mae = mean_absolute_error(Y_test.flatten(), baseline_sum)
    base_rmse = math.sqrt(mean_squared_error(Y_test.flatten(), baseline_sum))
    print("Persistence baseline (last timestep) -- MAE:", base_mae, "RMSE:", base_rmse)
else:
    print("Processed .npz not found at", npz_path, "- skipping persistence baseline")

# Show first 200 points plot
n = min(500, len(preds))
plt.figure(figsize=(12,4))
plt.plot(trues[:n], label="True", linewidth=1)
plt.plot(preds[:n], label="Pred", linewidth=1)
plt.legend()
plt.title("First {} test predictions vs true".format(n))
plt.show()

# Error histogram
errs = preds - trues
plt.figure(figsize=(6,3))
plt.hist(errs, bins=50)
plt.title("Prediction error histogram (pred - true)")
plt.show()
