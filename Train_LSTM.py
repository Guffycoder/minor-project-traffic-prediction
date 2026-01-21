# Train_LSTM.py
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import time
import argparse
import pandas as pd

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        if self.Y.ndim == 2 and self.Y.shape[1] == 1:
            self.Y = self.Y[:, 0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class RNNRegressor(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0,
                               bidirectional=False)
        else:
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0,
                              bidirectional=False)
        mid = max(hidden_size // 2, 8)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Linear(mid, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.squeeze(1)

def evaluate(model, loader, device, abs_tol):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            out = model(xb)
            preds.append(out.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1)
    trues = np.concatenate(trues, axis=0).reshape(-1)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    acc_abs = float(np.mean(np.abs(trues - preds) <= abs_tol))
    return mae, rmse, r2, acc_abs, preds, trues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="D:/minor final/sequences_Tin12_pred1.npz",
                        help="Path to processed .npz file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rnn_type", type=str, choices=["GRU","LSTM"], default="LSTM")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 safe on Windows)")
    parser.add_argument("--abs_tol", type=float, default=1.0, help="Absolute tolerance for accuracy")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("RNN type:", args.rnn_type)

    # load data
    data_path = Path(args.data)
    assert data_path.exists(), f"Data file not found: {data_path}"
    npz = np.load(data_path, allow_pickle=True)
    X_train = npz["X_train"]
    Y_train = npz["Y_train"]
    X_val = npz["X_val"]
    Y_val = npz["Y_val"]
    X_test = npz["X_test"]
    Y_test = npz["Y_test"]
    print("Shapes:", X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

    input_size = X_train.shape[2]

    train_ds = SeqDataset(X_train, Y_train)
    val_ds = SeqDataset(X_val, Y_val)
    test_ds = SeqDataset(X_test, Y_test)

    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    # model
    model = RNNRegressor(args.rnn_type, input_size=input_size,
                         hidden_size=args.hidden_size, num_layers=args.num_layers,
                         dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_path = os.path.join(args.save_dir, "best_model.pth")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float().squeeze()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if len(epoch_losses)>0 else 0.0
        val_mae, val_rmse, val_r2, val_acc, _, _ = evaluate(model, val_loader, device, args.abs_tol)
        val_loss = val_rmse**2
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | TrainLoss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R2: {val_r2:.4f} | Time: {elapsed/60:.2f}m")

        # early stopping & checkpoint
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, best_path)
            print(f"  Saved new best model to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    # load best and evaluate on test
    if not os.path.exists(best_path):
        print("Warning: no checkpoint found at", best_path, "- using current model weights.")
    else:
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_mae, test_rmse, test_r2, test_acc, preds, trues = evaluate(model, test_loader, device, args.abs_tol)
    print(f"\nTest results -- MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, AbsAcc(±{args.abs_tol}): {test_acc*100:.2f}%")

    # Save test predictions to CSV
    out_dir = args.save_dir
    np.savetxt(os.path.join(out_dir, "test_preds.csv"), preds, delimiter=",", header="pred", comments="")
    np.savetxt(os.path.join(out_dir, "test_trues.csv"), trues, delimiter=",", header="true", comments="")
    pd.DataFrame({"true": trues, "pred": preds, "abs_err": np.abs(trues - preds)}).to_csv(os.path.join(out_dir, "diagnostics.csv"), index=False)
    print("Saved test predictions and diagnostics to", out_dir)

if __name__ == "__main__":
    main()
