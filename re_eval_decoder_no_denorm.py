# re_eval_decoder_no_denorm.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ---------- PATHS ----------
feat_npz = os.path.join("checkpoints_hybrid", "sequences_ST3D_from_single_feats.npz")
decoder_weights = os.path.join("checkpoints_hybrid", "best_decoder.pth")
out_dir = "checkpoints_hybrid"
os.makedirs(out_dir, exist_ok=True)

# ---------- AUTO-LOCATE feature NPZ ----------
if not os.path.exists(feat_npz):
    # try fallback names
    for f in os.listdir("checkpoints_hybrid"):
        if f.endswith("_feats.npz"):
            feat_npz = os.path.join("checkpoints_hybrid", f)
            break

if not os.path.exists(feat_npz):
    print("Could not find *_feats.npz in checkpoints_hybrid. Files present:")
    print(os.listdir("checkpoints_hybrid"))
    sys.exit(1)

print("Using feature file:", feat_npz)

# ---------- LOAD FEATURE NPZ ----------
d = np.load(feat_npz, allow_pickle=True)
print("NPZ keys:", d.files)

# safely extract feats_test
if "feats_test" in d:
    feats_test = d["feats_test"]
else:
    print("ERROR: feats_test not found in NPZ!! Keys =", d.files)
    sys.exit(1)

# extract Y_test
Y_test = None
if "Y_test" in d:
    Y_test = d["Y_test"]
else:
    # fallback: load from original ST3D NPZ
    src = np.load("sequences_ST3D_from_single.npz", allow_pickle=True)
    if "Y_test" in src:
        Y_test = src["Y_test"]

print("feats_test shape:", np.asarray(feats_test).shape)
if Y_test is not None:
    print("Y_test raw shape:", np.asarray(Y_test).shape)

# ---------- CLEAN Y_test ----------
if Y_test is not None:
    Y = np.squeeze(np.asarray(Y_test))
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)
    elif Y.ndim > 2:
        # collapse any leftover dims
        axes = tuple(range(2, Y.ndim))
        Y = Y.sum(axis=axes)
else:
    Y = None

# ---------- Define decoder architecture (must match training) ----------
class DecoderMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ---------- Determine dimensions ----------
feats_test = np.asarray(feats_test)
in_dim = feats_test.shape[1]
K = 1
if Y is not None:
    if Y.ndim == 1:
        K = 1
    else:
        K = Y.shape[1]

print("Decoder input dim =", in_dim, "| output dim (K) =", K)

# ---------- Load decoder weights ----------
hidden_candidates = [128, 256, 64, 512]
decoder = None
loaded = False

for h in hidden_candidates:
    try:
        model = DecoderMLP(in_dim, h, K)
        model.load_state_dict(torch.load(decoder_weights, map_location="cpu"))
        decoder = model
        loaded = True
        print("Loaded decoder weights with hidden size =", h)
        break
    except Exception as e:
        pass

if not loaded:
    print("Could NOT load decoder weights. Wrong architecture?")
    sys.exit(1)

decoder.eval()

# ---------- RUN INFERENCE ----------
with torch.no_grad():
    X = torch.from_numpy(feats_test).float()
    preds = decoder(X).cpu().numpy()

print("Predictions computed.")
print("Preds stats:", preds.shape, preds.min(), preds.max(), preds.mean())
if Y is not None:
    print("Trues stats:", Y.min(), Y.max(), Y.mean())

# ---------- SAVE ----------
np.savetxt(os.path.join(out_dir, "test_preds_no_denorm.csv"),
           preds.reshape(preds.shape[0], -1), delimiter=",")

if Y is not None:
    np.savetxt(os.path.join(out_dir, "test_trues_no_denorm.csv"),
               Y.reshape(Y.shape[0], -1), delimiter=",")

print("Saved:")
print(" - test_preds_no_denorm.csv")
if Y is not None:
    print(" - test_trues_no_denorm.csv")
