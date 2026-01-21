import numpy as np

for p in [
    "checkpoints_gru/test_trues.csv",
    "checkpoints_lstm/test_trues.csv",
    "checkpoints_multi/test_trues_reg.csv",
    "checkpoints_hybrid/test_trues.csv",
]:
    try:
        arr = np.loadtxt(p, delimiter=",")
        print(p, arr.shape)
    except:
        pass
