#!/usr/bin/env python3
"""
train_hybrid_temporal.py (final, fully debugged)

Features:
 - Temporal-only encoder (Conv1D or LSTM) in Keras.
 - Option to train encoder to predict scalar (mean) or full multi-step targets.
 - Robust encoder checkpointing (saves base encoder weights reliably).
 - Extracts features via feature-extractor model and saves a features .npz.
 - Trains a PyTorch MLP decoder on features -> multi-step outputs.
 - Saves raw decoder outputs (test_preds_raw.csv) and trues; applies denorm only if safe.
 - Heuristics prevent accidental double-denormalization.
Usage:
 python train_hybrid_temporal.py --st3d_npz "d:/minor final/sequences_ST3D_from_single.npz"
"""
import os
import argparse
import numpy as np
import math
import time
import logging

# TF/Keras
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--st3d_npz", type=str, required=True,
                    help="Input NPZ containing X_train and Y_train (or Yreg_train).")
parser.add_argument("--save_dir", type=str, default="checkpoints_hybrid")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)        # decoder lr
parser.add_argument("--encoder_lr", type=float, default=5e-4)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--feat_dim", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_lstm", action="store_true")
parser.add_argument("--normalize_inputs", action="store_true")
parser.add_argument("--encoder_target", choices=["scalar", "multi"], default="scalar")
args = parser.parse_args()

# -------------------------
# Setup
# -------------------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)
tf.random.set_seed(args.seed)
os.makedirs(args.save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("Using PyTorch device: %s", device)

# -------------------------
# Load NPZ
# -------------------------
npz = np.load(args.st3d_npz, allow_pickle=True)
logging.info("Loaded NPZ keys: %s", list(npz.files))

if "X_train" not in npz:
    raise KeyError("X_train not found in NPZ file.")
# support both names for regression target
if "Y_train" in npz:
    rawY_train = npz["Y_train"]
    rawY_val = npz.get("Y_val", None)
    rawY_test = npz.get("Y_test", None)
elif "Yreg_train" in npz:
    rawY_train = npz["Yreg_train"]
    rawY_val = npz.get("Yreg_val", None)
    rawY_test = npz.get("Yreg_test", None)
else:
    raise KeyError("Regression targets not found (expect Y_train or Yreg_train).")

X_train = npz["X_train"]
X_val = npz.get("X_val", None)
X_test = npz.get("X_test", None)
means = npz.get("means", None)
stds  = npz.get("stds", None)
if means is not None: means = np.asarray(means).astype(float)
if stds is not None:  stds  = np.asarray(stds).astype(float)

# -------------------------
# Ensure regression shape (N, K)
# -------------------------
def ensure_reg_shape(Y):
    if Y is None:
        return None
    Y = np.asarray(Y)
    Y = np.squeeze(Y)
    if Y.ndim == 0:
        Y = Y.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim > 2:
        axes = tuple(range(2, Y.ndim))
        Y = Y.sum(axis=axes)
    return Y

Y_train = ensure_reg_shape(rawY_train)
Y_val   = ensure_reg_shape(rawY_val) if rawY_val is not None else None
Y_test  = ensure_reg_shape(rawY_test) if rawY_test is not None else None
if Y_train is None:
    raise ValueError("Regression targets are required in NPZ.")
K = Y_train.shape[1]
logging.info("Y shapes after ensure: train=%s val=%s test=%s (K=%d)",
             Y_train.shape, None if Y_val is None else Y_val.shape, None if Y_test is None else Y_test.shape, K)
logging.info("means: %s stds: %s", None if means is None else means.shape, None if stds is None else stds.shape)

# -------------------------
# Prepare temporal X -> (N, T, C)
# -------------------------
def prep_temporal_X(X):
    X = np.asarray(X)
    if X.ndim == 5 and X.shape[3] == 1 and X.shape[4] == 1:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # (N,C,T)
    if X.ndim == 3:
        N, a, b = X.shape
        if a <= 8 and b > a:
            Xp = np.transpose(X, (0,2,1))  # (N,T,C)
        else:
            Xp = X
        return Xp.astype(np.float32)
    else:
        raise ValueError("Unsupported X shape for temporal encoder. Got: %s" % (X.shape,))

Xtr_tem = prep_temporal_X(X_train)
Xval_tem = prep_temporal_X(X_val) if X_val is not None else None
Xtest_tem = prep_temporal_X(X_test) if X_test is not None else None
logging.info("Prepared temporal X shapes (N,T,C): tr=%s val=%s test=%s",
             Xtr_tem.shape, None if Xval_tem is None else Xval_tem.shape, None if Xtest_tem is None else Xtest_tem.shape)

# -------------------------
# Optional input normalization
# -------------------------
if args.normalize_inputs:
    if means is not None and stds is not None:
        m = np.asarray(means).reshape(-1)
        s = np.asarray(stds).reshape(-1)
        if m.size == 1:
            Xtr_tem = (Xtr_tem - float(m[0])) / float(s[0])
            if Xval_tem is not None: Xval_tem = (Xval_tem - float(m[0])) / float(s[0])
            if Xtest_tem is not None: Xtest_tem = (Xtest_tem - float(m[0])) / float(s[0])
        else:
            if m.size != Xtr_tem.shape[2]:
                logging.warning("means size mismatch; using scalar mean/std")
                Xtr_tem = (Xtr_tem - float(m.mean())) / float(s.mean())
                if Xval_tem is not None: Xval_tem = (Xval_tem - float(m.mean())) / float(s.mean())
                if Xtest_tem is not None: Xtest_tem = (Xtest_tem - float(m.mean())) / float(s.mean())
            else:
                Xtr_tem = (Xtr_tem - m.reshape(1,1,-1)) / s.reshape(1,1,-1)
                if Xval_tem is not None: Xval_tem = (Xval_tem - m.reshape(1,1,-1)) / s.reshape(1,1,-1)
                if Xtest_tem is not None: Xtest_tem = (Xtest_tem - m.reshape(1,1,-1)) / s.reshape(1,1,-1)
    else:
        ch_mean = Xtr_tem.mean(axis=(0,1)); ch_std = Xtr_tem.std(axis=(0,1)); ch_std[ch_std==0]=1.0
        Xtr_tem = (Xtr_tem - ch_mean.reshape(1,1,-1)) / ch_std.reshape(1,1,-1)
        if Xval_tem is not None: Xval_tem = (Xval_tem - ch_mean.reshape(1,1,-1)) / ch_std.reshape(1,1,-1)
        if Xtest_tem is not None: Xtest_tem = (Xtest_tem - ch_mean.reshape(1,1,-1)) / ch_std.reshape(1,1,-1)
    logging.info("Applied input normalization to temporal X")

# -------------------------
# Build encoder base (features + head_out)
# -------------------------
def build_conv1d_base(T, C, feat_dim=128, conv_filters=(64,32), kernel_size=3, dropout=0.2, out_units=1):
    inp = tf.keras.Input(shape=(T, C), name="temporal_input")
    x = inp
    for f in conv_filters:
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(pool_size=2, padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(max(feat_dim//2, 32), activation="relu")(x)
    if dropout and dropout>0:
        x = layers.Dropout(dropout)(x)
    feat = layers.Dense(feat_dim, activation=None, name="features")(x)
    head = layers.Dense(out_units, activation=None, name="head_out")(feat)
    return Model(inputs=inp, outputs=[feat, head], name="Conv1DBase")

def build_lstm_base(T, C, feat_dim=128, lstm_units=(128,), dropout=0.2, out_units=1):
    inp = tf.keras.Input(shape=(T, C), name="temporal_input")
    x = inp
    for i,u in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units)-1)
        x = layers.LSTM(u, return_sequences=return_sequences)(x)
        if dropout and dropout>0:
            x = layers.Dropout(dropout)(x)
    x = layers.Dense(max(feat_dim//2, 32), activation="relu")(x)
    feat = layers.Dense(feat_dim, activation=None, name="features")(x)
    head = layers.Dense(out_units, activation=None, name="head_out")(feat)
    return Model(inputs=inp, outputs=[feat, head], name="LSTMBase")

T_len = Xtr_tem.shape[1]; C_ch = Xtr_tem.shape[2]
out_units = 1 if args.encoder_target == "scalar" else K
base_encoder = build_lstm_base(T_len, C_ch, feat_dim=args.feat_dim, lstm_units=(128,), dropout=args.dropout, out_units=out_units) if args.use_lstm \
               else build_conv1d_base(T_len, C_ch, feat_dim=args.feat_dim, conv_filters=(64,32), kernel_size=3, dropout=args.dropout, out_units=out_units)
base_encoder.summary()

# -------------------------
# Prepare encoder head targets
# -------------------------
if args.encoder_target == "scalar":
    scalar_train = np.mean(Y_train, axis=1).astype(np.float32)
    scalar_val = np.mean(Y_val, axis=1).astype(np.float32) if Y_val is not None else None
    y_mean = float(np.mean(scalar_train)); y_std = float(np.std(scalar_train)); y_std = 1.0 if y_std==0 else y_std
    train_head_target = ((scalar_train - y_mean) / y_std).reshape(-1,1).astype(np.float32)
    val_head_target = ((scalar_val - y_mean) / y_std).reshape(-1,1).astype(np.float32) if scalar_val is not None else None
    logging.info("Encoder targets: scalar normalized by mean=%.6f std=%.6f", y_mean, y_std)
else:
    train_Y = Y_train.astype(np.float32)
    val_Y = Y_val.astype(np.float32) if Y_val is not None else None
    Y_mean = train_Y.mean(axis=0); Y_std = train_Y.std(axis=0); Y_std[Y_std==0]=1.0
    train_head_target = ((train_Y - Y_mean.reshape(1,-1)) / Y_std.reshape(1,-1)).astype(np.float32)
    val_head_target = ((val_Y - Y_mean.reshape(1,-1)) / Y_std.reshape(1,-1)).astype(np.float32) if val_Y is not None else None
    y_mean = y_std = None
    logging.info("Encoder targets: multi-step normalized (per-output)")

# -------------------------
# Robust encoder checkpointing (must end with .weights.h5)
# -------------------------
enc_ckpt_path = os.path.join(args.save_dir, "best_temporal_base_encoder.weights.h5")
class SaveBaseEncoderOnImprovingVal(callbacks.Callback):
    def __init__(self, base_model, ckpt_path):
        super().__init__()
        self.base_model = base_model
        self.ckpt_path = ckpt_path
        self.best = np.inf
    def on_epoch_end(self, epoch, logs=None):
        val_loss = None
        if logs is not None:
            val_loss = logs.get("val_loss", None)
        if val_loss is None:
            return
        if val_loss < self.best - 1e-8:
            self.best = val_loss
            try:
                # Use default save_weights naming (.weights.h5)
                self.base_model.save_weights(self.ckpt_path)
            except Exception:
                # fallback to TF format if needed
                try:
                    self.base_model.save_weights(self.ckpt_path, save_format='tf')
                except Exception as e:
                    logging.warning("Failed to save encoder weights: %s", e)
            logging.info("Saved base encoder weights to %s (val_loss=%.6f)", self.ckpt_path, val_loss)

enc_callbacks = [
    SaveBaseEncoderOnImprovingVal(base_encoder, enc_ckpt_path),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=False, verbose=1)
]

# -------------------------
# Create train_model (head_out) and encoder_feat_model (features)
# -------------------------
head_output = base_encoder.get_layer("head_out").output
train_model = Model(inputs=base_encoder.input, outputs=head_output, name="encoder_train_model")
feat_output = base_encoder.get_layer("features").output
encoder_feat_model = Model(inputs=base_encoder.input, outputs=feat_output, name="encoder_feat_model")

train_model.compile(optimizer=optimizers.Adam(learning_rate=args.encoder_lr),
                    loss="mse", metrics=["mse","mae"])

# -------------------------
# Fit encoder
# -------------------------
logging.info("Starting encoder training (target=%s)...", args.encoder_target)
if val_head_target is not None:
    train_model.fit(Xtr_tem, train_head_target,
                    validation_data=(Xval_tem, val_head_target),
                    epochs=min(10, args.epochs),
                    batch_size=args.batch_size,
                    callbacks=enc_callbacks,
                    verbose=2)
else:
    train_model.fit(Xtr_tem, train_head_target,
                    epochs=min(5, args.epochs),
                    batch_size=args.batch_size,
                    callbacks=enc_callbacks,
                    verbose=2)

# Load encoder weights into feature extractor (robust)
if os.path.exists(enc_ckpt_path):
    try:
        encoder_feat_model.load_weights(enc_ckpt_path)
        logging.info("Loaded encoder feature-extractor weights from %s", enc_ckpt_path)
    except Exception as e:
        logging.warning("Direct load into encoder_feat_model failed: %s", e)
        try:
            base_encoder.load_weights(enc_ckpt_path)
            encoder_feat_model.set_weights(base_encoder.get_weights())
            logging.info("Loaded weights via base_encoder fallback.")
        except Exception as e2:
            logging.warning("Fallback loading failed: %s. Continuing with current weights.", e2)
else:
    logging.info("No encoder checkpoint found; using current weights.")

temporal_encoder = encoder_feat_model

# -------------------------
# Encode features
# -------------------------
def encode_features(model, X_tem):
    if X_tem is None:
        return None
    preds = model.predict(X_tem, batch_size=args.batch_size, verbose=0)
    return np.asarray(preds)

feats_train = encode_features(temporal_encoder, Xtr_tem)
feats_val = encode_features(temporal_encoder, Xval_tem) if Xval_tem is not None else None
feats_test = encode_features(temporal_encoder, Xtest_tem) if Xtest_tem is not None else None
logging.info("Encoded feature shapes: train=%s val=%s test=%s", feats_train.shape,
             None if feats_val is None else feats_val.shape, None if feats_test is None else feats_test.shape)

# Save features and encoder target normalizers if available
feats_npz = os.path.join(args.save_dir, os.path.basename(args.st3d_npz).replace(".npz", "_feats_temporal.npz"))
np.savez_compressed(feats_npz, feats_train=feats_train, feats_val=feats_val, feats_test=feats_test,
                    Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, means=means, stds=stds,
                    y_mean=(y_mean if 'y_mean' in locals() else None),
                    y_std=(y_std if 'y_std' in locals() else None))
logging.info("Saved features npz to %s", feats_npz)

# -------------------------
# PyTorch decoder (MLP)
# -------------------------
in_dim = feats_train.shape[1]
class DecoderMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        mid = max(hidden // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

decoder = DecoderMLP(in_dim=in_dim, hidden=args.hidden, out_dim=K, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=1e-5)
criterion = nn.MSELoss()

train_ds = TensorDataset(torch.from_numpy(feats_train).float(), torch.from_numpy(Y_train).float())
val_ds = TensorDataset(torch.from_numpy(feats_val).float(), torch.from_numpy(Y_val).float()) if feats_val is not None and Y_val is not None else None
test_ds = TensorDataset(torch.from_numpy(feats_test).float(), torch.from_numpy(Y_test).float()) if feats_test is not None and Y_test is not None else None

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds is not None else None

# -------------------------
# Training loop for decoder
# -------------------------
best_val = float('inf'); best_path = os.path.join(args.save_dir, "best_decoder.pth")
def evaluate_loader(model, loader):
    model.eval()
    preds = []; trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy()); trues.append(yb.cpu().numpy())
    if not preds:
        return None, None, None, None
    preds = np.concatenate(preds, axis=0); trues = np.concatenate(trues, axis=0)
    mae = float(np.mean(np.abs(trues - preds))); rmse = float(math.sqrt(np.mean((trues - preds)**2)))
    return mae, rmse, preds, trues

logging.info("Training decoder (PyTorch MLP) ...")
for epoch in range(1, args.epochs + 1):
    decoder.train(); losses = []; t0 = time.time()
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad()
        out = decoder(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    if val_loader is not None:
        val_mae, val_rmse, _, _ = evaluate_loader(decoder, val_loader)
    else:
        val_mae = val_rmse = None
    elapsed = time.time() - t0
    logging.info("Epoch %03d | TrainLoss: %.6f | Val MAE: %s RMSE: %s | Time: %.2fs",
                 epoch, float(np.mean(losses)), "N/A" if val_mae is None else f"{val_mae:.6f}",
                 "N/A" if val_rmse is None else f"{val_rmse:.6f}", elapsed)
    if val_rmse is not None and val_rmse < best_val - 1e-8:
        best_val = val_rmse; torch.save(decoder.state_dict(), best_path)
        logging.info("Saved best decoder to %s (val_rmse=%.6f)", best_path, val_rmse)

if not os.path.exists(best_path):
    torch.save(decoder.state_dict(), best_path); logging.info("Saved decoder final weights to %s", best_path)

decoder.load_state_dict(torch.load(best_path, map_location=device))
if test_loader is not None:
    test_mae, test_rmse, preds_test, trues_test = evaluate_loader(decoder, test_loader)
    logging.info("Hybrid test results -- MAE: %.6f, RMSE: %.6f", test_mae, test_rmse)
else:
    preds_test = trues_test = None; logging.info("No test set provided; skipping final test evaluation.")

# -------------------------
# Safe saving: raw preds + optional denorm (heuristic)
# -------------------------
def compute_metrics(p, t):
    mae = float(np.mean(np.abs(t - p))); rmse = float(math.sqrt(np.mean((t - p)**2)))
    return mae, rmse

preds = np.asarray(preds_test) if preds_test is not None else None
trues = np.asarray(trues_test) if trues_test is not None else None

if preds is None or trues is None:
    logging.info("No predictions/trues to save.")
else:
    raw_path = os.path.join(args.save_dir, "test_preds_raw.csv")
    np.savetxt(raw_path, preds.reshape(preds.shape[0], -1), delimiter=",")
    np.savetxt(os.path.join(args.save_dir, "test_trues.csv"), trues.reshape(trues.shape[0], -1), delimiter=",")
    raw_mae, raw_rmse = compute_metrics(preds, trues)
    logging.info("Saved raw preds to %s", raw_path)
    logging.info("Raw preds -> TEST MAE: %.6f, RMSE: %.6f", raw_mae, raw_rmse)

    # Attempt safe denorm using features npz y_mean/y_std or original NPZ means/stds
    denorm_done = False
    try:
        # Prefer feats npz saved y_mean/y_std
        if os.path.exists(feats_npz):
            f = np.load(feats_npz, allow_pickle=True)
            fy_mean = f.get("y_mean", None); fy_std = f.get("y_std", None)
            if fy_mean is not None and fy_std is not None:
                preds_mean = float(np.mean(preds))
                # Heuristic: only denorm if preds appear to be small (i.e., normalized)
                if abs(preds_mean) < 100.0:
                    denorm_preds = preds * float(fy_std) + float(fy_mean)
                    denorm_path = os.path.join(args.save_dir, "test_preds_denorm_via_featsnpz.csv")
                    np.savetxt(denorm_path, denorm_preds.reshape(denorm_preds.shape[0], -1), delimiter=",")
                    denorm_mae, denorm_rmse = compute_metrics(denorm_preds, trues)
                    logging.info("Denorm (feats npz) -> TEST MAE: %.6f, RMSE: %.6f (saved to %s)",
                                 denorm_mae, denorm_rmse, denorm_path)
                    denorm_done = True
        # Fallback: use original NPZ means/stds if plausible
        if not denorm_done:
            src = np.load(args.st3d_npz, allow_pickle=True)
            cand_m = src.get('means', src.get('data_mean', None))
            cand_s = src.get('stds', src.get('data_std', None))
            if cand_m is not None and cand_s is not None:
                m = float(np.asarray(cand_m).reshape(-1).mean())
                s = float(np.asarray(cand_s).reshape(-1).mean())
                preds_mean = float(np.mean(preds))
                if abs(preds_mean) < 100.0 and s > 0:
                    denorm_preds = preds * s + m
                    denorm_path = os.path.join(args.save_dir, "test_preds_denorm_via_orignpz.csv")
                    np.savetxt(denorm_path, denorm_preds.reshape(denorm_preds.shape[0], -1), delimiter=",")
                    denorm_mae, denorm_rmse = compute_metrics(denorm_preds, trues)
                    logging.info("Denorm (orig npz) -> TEST MAE: %.6f, RMSE: %.6f (saved to %s)",
                                 denorm_mae, denorm_rmse, denorm_path)
                    denorm_done = True
    except Exception as e:
        logging.warning("Denorm attempt failed: %s", e)

    if not denorm_done:
        logging.info("Skipped automatic denormalization (no safe normalizers found or preds already raw).")

logging.info("All done.")
