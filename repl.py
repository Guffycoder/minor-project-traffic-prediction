import numpy as np, tensorflow as tf
npz = np.load("sequences_ST3D_from_single.npz", allow_pickle=True)
X_train = npz["X_train"]   # expected (N, C, T, H, W)
Y_train = npz["Y_train"]   # expected (N, C, K, H, W) or (N, K) depending on preprocess

print("X_train:", X_train.shape, X_train.dtype)
print("Y_train:", Y_train.shape, Y_train.dtype)

from ST3DNet import ST3DNet
c_conf = (X_train.shape[2], X_train.shape[1], X_train.shape[3], X_train.shape[4])  # (len_closeness, nb_flow, H, W)
model = ST3DNet(c_conf=c_conf, t_conf=None, external_dim=None, nb_residual_unit=2, out_channels=8)
model.summary()
print("Model inputs:", [i.shape for i in model.inputs])
print("Model outputs:", [o.shape for o in model.outputs])

# show a few data samples
print("X[0] shape, min,max:", X_train[0].shape, X_train[0].min(), X_train[0].max())
print("Y[0] shape:", Y_train[0].shape)
