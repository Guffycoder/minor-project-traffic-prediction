#!/usr/bin/env python3
"""
preprocess_multistep.py
Build multi-step regression + classification sequences from a single-site CSV.

Saves: sequences_Tin12_pred{K}.npz with fields:
  X_train (N_train, T_in, 1)
  Yreg_train (N_train, K)
  Ycls_train (N_train,)
  ... (val/test)
  means, stds, columns, label_thresholds
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="TrafficDataset.csv")
parser.add_argument("--date_col", type=str, default="DateTime")
parser.add_argument("--total_col", type=str, default="Total")
parser.add_argument("--t_in", type=int, default=12)
parser.add_argument("--pred_len", type=int, default=5)
parser.add_argument("--out", type=str, default="sequences_Tin12_pred5.npz")
parser.add_argument("--label_strategy", choices=["quantiles","fixed"], default="quantiles")
parser.add_argument("--fixed_low", type=float, default=20.0)
parser.add_argument("--fixed_high", type=float, default=80.0)
parser.add_argument("--freq", type=str, default=None)
parser.add_argument("--val_frac", type=float, default=0.18)
parser.add_argument("--test_frac", type=float, default=0.12)
args = parser.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} not found. Put your CSV in the folder or change --csv")

df = pd.read_csv(csv_path)
# try to infer combined datetime column if necessary
if args.date_col not in df.columns:
    if "Date" in df.columns and "Time" in df.columns:
        df[args.date_col] = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    else:
        raise KeyError(f"Date column {args.date_col} not found and couldn't infer from Date/Time.")

# parse datetime and sort
df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
df = df.sort_values(by=args.date_col).set_index(args.date_col)

# resample if requested
series = df[args.total_col].astype(float)
if args.freq:
    series = series.resample(args.freq).sum().fillna(0.0)
else:
    series = series.fillna(method="ffill").fillna(0.0)

values = series.values
T_in = args.t_in
K = args.pred_len
if len(values) < T_in + K:
    raise ValueError("Time series too short for requested T_in+pred_len.")

X_list = []
Yreg_list = []
for i in range(T_in, len(values) - K + 1):
    xin = values[i - T_in:i]         # shape (T_in,)
    yout = values[i:i + K]           # shape (K,)
    X_list.append(xin.reshape(T_in, 1))
    Yreg_list.append(yout)

X = np.array(X_list)    # (N, T_in, 1)
Yreg = np.array(Yreg_list)  # (N, K)

# label thresholds
if args.label_strategy == "quantiles":
    low_thr, high_thr = np.percentile(Yreg.flatten(), [33, 66])
else:
    low_thr, high_thr = args.fixed_low, args.fixed_high

def label_from_val(v):
    if v <= low_thr:
        return 0
    if v <= high_thr:
        return 1
    return 2

Ycls = np.array([label_from_val(y[-1]) for y in Yreg], dtype=np.int64)

# splits (time-based)
N = len(X)
test_count = int(np.floor(args.test_frac * N))
val_count = int(np.floor(args.val_frac * N))
train_count = N - val_count - test_count

X_train = X[:train_count]; Yreg_train = Yreg[:train_count]; Ycls_train = Ycls[:train_count]
X_val = X[train_count: train_count + val_count]; Yreg_val = Yreg[train_count: train_count + val_count]; Ycls_val = Ycls[train_count: train_count + val_count]
X_test = X[train_count + val_count:]; Yreg_test = Yreg[train_count + val_count:]; Ycls_test = Ycls[train_count + val_count:]

# normalize inputs (z-score on X_train)
means = X_train.mean(axis=(0,1))   # (C,) usually (1,)
stds = X_train.std(axis=(0,1))
stds[stds == 0.0] = 1.0

def normalize_arr(x):
    return (x - means.reshape(1,1,-1)) / stds.reshape(1,1,-1)

X_train_n = normalize_arr(X_train)
X_val_n = normalize_arr(X_val)
X_test_n = normalize_arr(X_test)

out_path = Path(args.out)
np.savez_compressed(str(out_path),
                    X_train=X_train_n, Yreg_train=Yreg_train, Ycls_train=Ycls_train,
                    X_val=X_val_n, Yreg_val=Yreg_val, Ycls_val=Ycls_val,
                    X_test=X_test_n, Yreg_test=Yreg_test, Ycls_test=Ycls_test,
                    means=means, stds=stds, columns=np.array([args.total_col]),
                    label_thresholds=np.array([low_thr, high_thr]))
print("Saved:", out_path)
print("Shapes: X_train", X_train_n.shape, "Yreg_train", Yreg_train.shape, "Ycls_train", Ycls_train.shape)
print("Label thresholds (low, high):", low_thr, high_thr)
