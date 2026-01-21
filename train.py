import numpy as np

data = np.load("sequences_Tin12_pred1.npz", allow_pickle=True)
print(list(data.keys()))

X_train, Y_train = data["X_train"], data["Y_train"]
X_val, Y_val = data["X_val"], data["Y_val"]
X_test, Y_test = data["X_test"], data["Y_test"]

print("Train shapes:", X_train.shape, Y_train.shape)
