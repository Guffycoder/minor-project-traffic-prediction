# train_rnn.py
import argparse, os, time, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="sequences_Tin12_pred1.npz")
parser.add_argument("--rnn", type=str, choices=["GRU","LSTM"], default="GRU")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--save_dir", type=str, default="checkpoints_rnn")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(args.data, allow_pickle=True)
X_train = data["X_train"]; Y_train = data["Yreg_train"] if "Yreg_train" in data else data["Y_train"]
X_val = data["X_val"]; Y_val = data["Yreg_val"] if "Yreg_val" in data else data["Y_val"]
X_test = data["X_test"]; Y_test = data["Yreg_test"] if "Yreg_test" in data else data["Y_test"]

# X assumed shape (N, T, C) or (N, T, 1) from preprocess_multistep
if X_train.ndim == 3:
    # (N, T, C) -> keep
    pass
else:
    # maybe (N,1,T,1,1) or (N,C,T,H,W) -> flatten spatial dims to features
    N, C, T, H, W = X_train.shape
    X_train = X_train.reshape(N, T, C*H*W)
    N2, C2, T2, H2, W2 = X_val.shape
    X_val = X_val.reshape(X_val.shape[0], T, C*H*W)
    X_test = X_test.reshape(X_test.shape[0], T, C*H*W)

# Dataset
class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(SeqDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(SeqDataset(X_val, Y_val), batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(SeqDataset(X_test, Y_test), batch_size=args.batch_size, shuffle=False)

input_size = X_train.shape[2]
pred_len = Y_train.shape[1] if Y_train.ndim==2 else 1

class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden, num_layers, dropout, pred_len):
        super().__init__()
        if rnn_type=="LSTM":
            self.rnn = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        else:
            self.rnn = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, pred_len))
    def forward(self, x):
        out,_ = self.rnn(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y

model = RNNModel(args.rnn, input_size, args.hidden, args.num_layers, args.dropout, pred_len).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
crit = nn.MSELoss()


os.makedirs(args.save_dir, exist_ok=True)
best_val = 1e18; early = 0; patience=8

def evaluate(loader):
    model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            out = model(xb)
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0); trues = np.concatenate(trues, axis=0)
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    rmse = math.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    return mae, rmse, preds, trues

for epoch in range(1, args.epochs+1):
    model.train(); losses=[]
    t0=time.time()
    for xb,yb in train_loader:
        xb = xb.to(device).float(); yb = yb.to(device).float()
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    val_mae, val_rmse, _, _ = evaluate(val_loader)
    vloss = val_rmse**2
    print(f"Epoch {epoch} train_loss={np.mean(losses):.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} time={(time.time()-t0):.1f}s")
    if vloss < best_val - 1e-6:
        best_val = vloss; early=0
        torch.save(model.state_dict(), os.path.join(args.save_dir,"best_rnn.pth"))
    else:
        early += 1
        if early>=patience:
            print("Early stopping"); break

# test
model.load_state_dict(torch.load(os.path.join(args.save_dir,"best_rnn.pth"), map_location=device))
test_mae, test_rmse, preds, trues = evaluate(test_loader)
print("Test MAE:", test_mae, "RMSE:", test_rmse)
np.savetxt(os.path.join(args.save_dir,"test_preds.csv"), preds.reshape(preds.shape[0], -1), delimiter=",")
np.savetxt(os.path.join(args.save_dir,"test_trues.csv"), trues.reshape(trues.shape[0], -1), delimiter=",")
