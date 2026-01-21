#!/usr/bin/env python3
"""
train_multitask_improved.py
Improved multitask trainer with:
 - regression target normalization (z-score) + denorm at eval/save
 - class weighting for classification
 - gradient clipping, scheduler, better prints
 - saves denormalized preds to CSV
Run example:
python train_multitask_improved.py --data sequences_Tin12_pred1.npz --save_dir checkpoints_multi --epochs 40 --model GRU --hidden 128
"""

import os, time, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sequences_Tin12_pred1.npz")
    parser.add_argument("--save_dir", type=str, default="checkpoints_multi")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for classification loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["GRU", "LSTM"], default="GRU")
    parser.add_argument("--clip", type=float, default=1.0, help="grad norm clip")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 is safe for Windows)")
    return parser.parse_args()
# -------------------------
# Helpers for shapes/IO
# -------------------------
def get_or_fail(npz, key):
    if key not in npz.files:
        raise KeyError(f"Missing {key} in {args.data}")
    return npz[key]

def ensure_X_flat(X):
    if X.ndim == 3:
        return X.astype(np.float32)
    if X.ndim == 5:
        N,C,T,H,W = X.shape
        return X.reshape(N, T, C*H*W).astype(np.float32)
    if X.ndim == 2:
        return X[:, None, :].astype(np.float32)
    raise ValueError("Unsupported X ndim: %d" % X.ndim)

def ensure_Yreg(Y):
    if Y.ndim == 1:
        return Y.reshape(-1,1).astype(np.float32)
    if Y.ndim == 2:
        return Y.astype(np.float32)
    # fallback (shouldn't normally happen)
    N = Y.shape[0]
    return Y.reshape(N, -1).astype(np.float32)

class MT_Dataset(Dataset):
    def __init__(self, X, Yreg_n, Ycls):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Yr = torch.tensor(Yreg_n, dtype=torch.float32)
        self.Yc = torch.tensor(Ycls, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Yr[idx], self.Yc[idx]

class MultiTaskModel(nn.Module):
    def __init__(self, model_type, input_size, hidden, num_layers, dropout, pred_len):
        super().__init__()
        self.model_type = model_type
        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers>1 else 0.0)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers>1 else 0.0)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, max(hidden//2,8)), nn.ReLU(),
            nn.Linear(max(hidden//2,8), pred_len)
        )
        self.clf_head = nn.Sequential(
            nn.Linear(hidden, max(hidden//2,8)), nn.ReLU(),
            nn.Linear(max(hidden//2,8), 3)
        )
    def forward(self, x):
        if self.model_type == "LSTM":
            out, (h_n, c_n) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)
        last = out[:, -1, :]
        return self.reg_head(last), self.clf_head(last)

def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds_reg = []; trues_reg = []
    preds_cls = []; trues_cls = []
    with torch.no_grad():
        for xb, yr, yc in loader:
            xb = xb.to(device).float(); yr = yr.to(device).float(); yc = yc.to(device)
            out_r, out_c = model(xb)
            preds_reg.append(out_r.cpu().numpy())
            trues_reg.append(yr.cpu().numpy())
            preds_cls.append(out_c.argmax(dim=1).cpu().numpy())
            trues_cls.append(yc.cpu().numpy())
    preds_reg = np.concatenate(preds_reg, axis=0)
    trues_reg = np.concatenate(trues_reg, axis=0)
    preds_cls = np.concatenate(preds_cls, axis=0)
    trues_cls = np.concatenate(trues_cls, axis=0)

    # denormalize reg predictions and truths
    preds_reg_den = preds_reg * y_std + y_mean
    trues_reg_den = trues_reg * y_std + y_mean

    mae = mean_absolute_error(trues_reg_den.flatten(), preds_reg_den.flatten())
    rmse = math.sqrt(mean_squared_error(trues_reg_den.flatten(), preds_reg_den.flatten()))
    r2 = r2_score(trues_reg_den.flatten(), preds_reg_den.flatten())
    cls_acc = 100.0 * (preds_cls == trues_cls).mean()
    return mae, rmse, r2, cls_acc, preds_reg_den, trues_reg_den, preds_cls, trues_cls

def main():
    global args
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    npz = np.load(args.data, allow_pickle=True)
    print("Loaded npz keys:", npz.files)

    X_train = get_or_fail(npz, "X_train")
    X_val   = get_or_fail(npz, "X_val")
    X_test  = get_or_fail(npz, "X_test")

    # pick regression arrays
    if "Yreg_train" in npz.files:
        Yreg_train = npz["Yreg_train"]
        Yreg_val   = npz["Yreg_val"]
        Yreg_test  = npz["Yreg_test"]
    else:
        Yreg_train = get_or_fail(npz, "Y_train")
        Yreg_val   = get_or_fail(npz, "Y_val")
        Yreg_test  = get_or_fail(npz, "Y_test")

    # classification labels (prefer explicit)
    if "Ycls_train" in npz.files:
        Ycls_train = npz["Ycls_train"]
        Ycls_val   = npz["Ycls_val"]
        Ycls_test  = npz["Ycls_test"]
        print("Using explicit classification labels from npz.")
    else:
        # derive 3-class using 33/66 quantiles from last step
        def derive_cls(Y):
            if Y.ndim == 1:
                last = Y.reshape(-1)
            elif Y.ndim == 2:
                # if shape (N, pred_len)
                last = Y[:, -1]
            else:
                N = Y.shape[0]; K = Y.shape[1]
                collapsed = Y.reshape(N, K, -1).sum(axis=2)
                last = collapsed[:, -1]
            low, high = np.percentile(last, [33, 66])
            cls = np.zeros(len(last), dtype=np.int64)
            cls[(last > low) & (last <= high)] = 1
            cls[last > high] = 2
            return cls
        Ycls_train = derive_cls(Yreg_train)
        Ycls_val   = derive_cls(Yreg_val)
        Ycls_test  = derive_cls(Yreg_test)
        print("Derived classification labels using quantiles.")

    # -------------------------
    # Normalize/reshape helpers
    # -------------------------
    X_train = ensure_X_flat(X_train)
    X_val   = ensure_X_flat(X_val)
    X_test  = ensure_X_flat(X_test)
    Yreg_train = ensure_Yreg(Yreg_train)
    Yreg_val   = ensure_Yreg(Yreg_val)
    Yreg_test  = ensure_Yreg(Yreg_test)

    Ycls_train = np.array(Ycls_train, dtype=np.int64)
    Ycls_val   = np.array(Ycls_val, dtype=np.int64)
    Ycls_test  = np.array(Ycls_test, dtype=np.int64)

    print("After reshape: X_train", X_train.shape, "Yreg_train", Yreg_train.shape, "Ycls_train", Ycls_train.shape)

    # -------- normalize regression targets (fit on train) ----------
    y_mean = float(Yreg_train.mean())
    y_std  = float(Yreg_train.std())
    if y_std == 0.0: y_std = 1.0
    print(f"Yreg normalization: mean={y_mean:.4f} std={y_std:.4f}")

    Yr_train_n = (Yreg_train - y_mean) / y_std
    Yr_val_n   = (Yreg_val   - y_mean) / y_std
    Yr_test_n  = (Yreg_test  - y_mean) / y_std

    # -------------------------
    # Dataset & model
    # -------------------------
    train_loader = DataLoader(MT_Dataset(X_train, Yr_train_n, Ycls_train),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(torch.cuda.is_available()))
    val_loader   = DataLoader(MT_Dataset(X_val, Yr_val_n, Ycls_val),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=(torch.cuda.is_available()))
    test_loader  = DataLoader(MT_Dataset(X_test, Yr_test_n, Ycls_test),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=(torch.cuda.is_available()))

    input_size = X_train.shape[2]
    pred_len = Yreg_train.shape[1]

    model = MultiTaskModel(args.model, input_size, args.hidden, args.num_layers, args.dropout, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # class weights for crossentropy (help with imbalance)
    counter = Counter(Ycls_train.tolist())
    total = len(Ycls_train)
    class_counts = [counter.get(i,0) for i in range(3)]
    print("Class counts (train):", class_counts)
    # compute inverse-frequency weights (normalized)
    cls_weights = np.array([ (total / (1 + c)) for c in class_counts ], dtype=np.float32)
    cls_weights = torch.tensor(cls_weights / cls_weights.mean(), dtype=torch.float32).to(device)
    print("Class weights (normalized):", cls_weights.cpu().numpy())

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=cls_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    # -------------------------
    # Training loop
    # -------------------------
    best_val_metric = float("inf")
    epochs_no_improve = 0
    best_ckpt = os.path.join(args.save_dir, "best_multitask_improved.pth")
    print("Starting training...")
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        t0 = time.time()
        for xb, yr, yc in train_loader:
            xb = xb.to(device).float(); yr = yr.to(device).float(); yc = yc.to(device)
            opt.zero_grad()
            out_r, out_c = model(xb)
            loss_r = mse_loss(out_r, yr)
            loss_c = ce_loss(out_c, yc)
            loss = loss_r + args.alpha * loss_c
            loss.backward()
            # gradient clipping
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            losses.append(float(loss.item()))
        # validation
        val_mae, val_rmse, val_r2, val_cls_acc, _, _, _, _ = evaluate(model, val_loader, device, y_mean, y_std)
        # use val_rmse^2 for scheduler & early-stop heuristic
        val_metric = val_rmse**2
        scheduler.step(val_metric)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | TrainLoss: {np.mean(losses):.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R2: {val_r2:.4f} | Val ClfAcc: {val_cls_acc:.2f}% | Time: {elapsed:.1f}s")
        # early stopping-ish saving
        if val_metric < best_val_metric - 1e-6:
            best_val_metric = val_metric
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "opt_state": opt.state_dict(), "epoch": epoch, "val_metric": val_metric}, best_ckpt)
            print("  Saved new best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break
    # load best and evaluate on test
    if os.path.exists(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
    else:
        print("Warning: no checkpoint found, using current weights.")

    test_mae, test_rmse, test_r2, test_cls_acc, preds_reg_den, trues_reg_den, preds_cls, trues_cls = evaluate(model, test_loader, device, y_mean, y_std)
    print("\nTest results:")
    print(f" Regression -> MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
    print(f" Classification -> Accuracy: {test_cls_acc:.2f}%")

    # Save denormalized regression preds & classification outputs (with headers)
    preds_reg_den = np.asarray(preds_reg_den)
    trues_reg_den = np.asarray(trues_reg_den)
    preds_cls = np.asarray(preds_cls, dtype=np.int32)
    trues_cls = np.asarray(trues_cls, dtype=np.int32)

    # reshape/regress to 2D for saving (N, pred_len)
    preds_reg_save = preds_reg_den.reshape(preds_reg_den.shape[0], -1)
    trues_reg_save = trues_reg_den.reshape(trues_reg_den.shape[0], -1)

    np.savetxt(os.path.join(args.save_dir, "test_preds_reg_denorm.csv"), preds_reg_save, delimiter=",", header=",".join([f"t{i}" for i in range(preds_reg_save.shape[1])]), comments="")
    np.savetxt(os.path.join(args.save_dir, "test_trues_reg_denorm.csv"), trues_reg_save, delimiter=",", header=",".join([f"t{i}" for i in range(trues_reg_save.shape[1])]), comments="")
    np.savetxt(os.path.join(args.save_dir, "test_preds_clf.csv"), preds_cls, fmt="%d", delimiter=",", header="pred_cls", comments="")
    np.savetxt(os.path.join(args.save_dir, "test_trues_clf.csv"), trues_cls, fmt="%d", delimiter=",", header="true_cls", comments="")
    print("Saved test outputs to", args.save_dir)

if __name__ == "__main__":
    main()
