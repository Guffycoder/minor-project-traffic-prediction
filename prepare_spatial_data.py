# prepare_spatial_data.py
"""
Robust prepare: supports two CSV layouts:

1) Long-format multi-location:
   columns: DateTime (or Date+Time), LocationID, Total
   -> pivot to (T, num_locations) and create (N, C, T_in, H, W) npz (H=1,W=M or grid)

2) Single-site:
   columns: Date, Time, Total (or DateTime and Total)
   -> creates 1x1 spatial grid arrays so ST3D pipeline can still run

Usage:
  python prepare_spatial_data.py --csv TrafficDataset.csv --t_in 12 --pred_len 5 --grid_mode flat --out sequences_ST3D.npz
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="TrafficDataset.csv",
                    help="CSV file location")
parser.add_argument("--date_col", type=str, default="Date",
                    help="Date column name (or DateTime if present)")
parser.add_argument("--time_col", type=str, default="Time",
                    help="Time column name (used if DateTime not present)")
parser.add_argument("--loc_col", type=str, default="LocationID",
                    help="Location column (if multi-location long-format). If not present script will treat CSV as single-site")
parser.add_argument("--val_col", type=str, default="Total", help="Value/total column")
parser.add_argument("--t_in", type=int, default=12)
parser.add_argument("--pred_len", type=int, default=5)
parser.add_argument("--grid_mode", type=str, choices=["flat","square"], default="flat")
parser.add_argument("--out", type=str, default="sequences_ST3D.npz")
parser.add_argument("--freq", type=str, default=None, help="Optional resample freq like '10T' or '600S'")
args = parser.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} not found. Put your CSV in project folder or pass --csv")

df = pd.read_csv(csv_path)
cols = [c.strip() for c in df.columns.tolist()]

# Helper: build DateTime column from Date+Time or use existing DateTime
if "DateTime" in df.columns:
    df["DateTime"] = pd.to_datetime(df["DateTime"])
else:
    # if explicit Date & Time columns present
    if args.date_col in df.columns and args.time_col in df.columns:
        # combine Date + Time
        df["DateTime"] = pd.to_datetime(df[args.date_col].astype(str).str.strip() + " " + df[args.time_col].astype(str).str.strip(),
                                        errors='coerce')
        # If parsing failed, try swapping
        if df["DateTime"].isna().any():
            df["DateTime"] = pd.to_datetime(df[args.time_col].astype(str).str.strip() + " " + df[args.date_col].astype(str).str.strip(),
                                            errors='coerce')
        if df["DateTime"].isna().any():
            # Last-ditch parse with dateutil (blunter)
            df["DateTime"] = pd.to_datetime(df[args.date_col].astype(str).str.strip() + " " + df[args.time_col].astype(str).str.strip(),
                                            infer_datetime_format=True, errors='coerce')
    else:
        # No DateTime and no Date+Time -> cannot proceed
        raise KeyError("CSV must contain either 'DateTime' or both Date and Time columns (names configurable). Found columns: " + ", ".join(df.columns))

df = df.sort_values("DateTime").reset_index(drop=True)

# Decide if multi-location (long format) or single-site
if args.loc_col in df.columns:
    # LONG FORMAT (multi-location)
    print("Detected multi-location dataset (long format) with LocationID column:", args.loc_col)
    # pivot: rows=time, cols=location
    pivot = df.pivot_table(index="DateTime", columns=args.loc_col, values=args.val_col, aggfunc='sum')
    pivot = pivot.fillna(0.0)

    # optional resample
    if args.freq:
        pivot = pivot.resample(args.freq).sum().fillna(0.0)
    else:
        # try to ensure regular freq by infer_freq & asfreq fallback
        try:
            freq = pd.infer_freq(pivot.index)
            if freq is None:
                freq = pivot.index.to_series().diff().mode()[0]
            pivot = pivot.asfreq(freq).fillna(method="ffill").fillna(0.0)
        except Exception:
            pivot = pivot.fillna(0.0)

    locations = pivot.columns.tolist()
    values = pivot.values  # (T_total, M)
    M = values.shape[1]

    # grid shape
    if args.grid_mode == "flat":
        H, W = 1, M
    else:
        s = int(np.ceil(np.sqrt(M)))
        H, W = s, s
        if H*W > M:
            pad = np.zeros((values.shape[0], H*W - M))
            values = np.concatenate([values, pad], axis=1)

    # reshape to (T, H, W)
    values_hw = values.reshape(values.shape[0], H, W)
else:
    # SINGLE-SITE case
    print("Detected single-site dataset (no LocationID). Using Total column:", args.val_col)
    if args.val_col not in df.columns:
        # try case-insensitive match
        found = None
        for c in df.columns:
            if c.strip().lower() == args.val_col.lower():
                found = c; break
        if found is None:
            raise KeyError(f"Value column '{args.val_col}' not found. Columns: {list(df.columns)}")
        args.val_col = found

    series = df.set_index("DateTime")[args.val_col].astype(float)
    if args.freq:
        series = series.resample(args.freq).sum().fillna(0.0)
    else:
        try:
            freq = pd.infer_freq(series.index)
            if freq is None:
                freq = series.index.to_series().diff().mode()[0]
            series = series.asfreq(freq).fillna(method="ffill").fillna(0.0)
        except Exception:
            series = series.fillna(method="ffill").fillna(0.0)

    values = series.values.reshape(-1, 1)  # (T_total, 1)
    locations = ["single_site"]
    H, W = 1, 1
    values_hw = values.reshape(values.shape[0], H, W)

# Now build sliding windows (T_in, K)
T_in = args.t_in; K = args.pred_len
if values_hw.shape[0] < T_in + K:
    raise ValueError(f"Time series too short for T_in={T_in}, pred_len={K}. length={values_hw.shape[0]}")

X_list = []; Y_list = []
for i in range(T_in, values_hw.shape[0] - K + 1):
    xin = values_hw[i-T_in:i]   # (T_in, H, W)
    yout = values_hw[i:i+K]     # (K, H, W)
    X_list.append(xin)
    Y_list.append(yout)

X = np.array(X_list)  # (N, T_in, H, W)
Y = np.array(Y_list)  # (N, K, H, W)

# reorder to (N, C=1, T_in, H, W)
X = np.expand_dims(X, axis=1)  # (N,1,T_in,H,W)
Y = np.expand_dims(Y, axis=1)  # (N,1,K,H,W)

# time-based split
N = X.shape[0]
test_frac = 0.12; val_frac = 0.18
test_n = int(N*test_frac); val_n = int(N*val_frac); train_n = N - val_n - test_n

X_train = X[:train_n]; Y_train = Y[:train_n]
X_val = X[train_n:train_n+val_n]; Y_val = Y[train_n:train_n+val_n]
X_test = X[train_n+val_n:]; Y_test = Y[train_n+val_n:]

# normalization
means = X_train.mean(axis=(0,2,3,4))
stds = X_train.std(axis=(0,2,3,4)); stds[stds==0]=1.0
def normalize(a):
    return (a - means.reshape(1,-1,1,1,1)) / stds.reshape(1,-1,1,1,1)

X_train_n = normalize(X_train); X_val_n = normalize(X_val); X_test_n = normalize(X_test)

np.savez_compressed(args.out,
                    X_train=X_train_n, Y_train=Y_train,
                    X_val=X_val_n, Y_val=Y_val,
                    X_test=X_test_n, Y_test=Y_test,
                    means=means, stds=stds, locations=np.array(locations), H=H, W=W)
print("Saved", args.out, "shapes:", X_train_n.shape, Y_train.shape, "H,W:", H, W)
