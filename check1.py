import numpy as np

paths = {
    "GRU": "checkpoints_gru/test_preds.csv",
    "LSTM": "checkpoints_lstm/test_preds.csv",
    "MULTITASK": "checkpoints_multi/test_preds_reg.csv",
    "ST3D": "checkpoints_st3d/best_st3d.weights.h5",
    "HYBRID": "checkpoints_hybrid/test_preds.csv",
    "HYBRID_NPZ": "checkpoints_hybrid/hybrid_predictions.npz"
}

for name, p in paths.items():
    try:
        if p.endswith(".csv"):
            arr = np.loadtxt(p, delimiter=",")
        elif p.endswith(".npz"):
            d = np.load(p)
            print(name, "npz keys:", list(d.keys()))
            arr = d[list(d.keys())[0]]
        else:
            continue
        print(name, arr.shape)
    except:
        pass
