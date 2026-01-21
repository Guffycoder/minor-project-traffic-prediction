import numpy as np
p = np.loadtxt('checkpoints_hybrid/test_preds.csv', delimiter=',')
t = np.loadtxt('checkpoints_hybrid/test_trues.csv', delimiter=',')
print('preds shape, min/max/mean:', p.shape, p.min(), p.max(), p.mean())
print('trues shape, min/max/mean:', t.shape, t.min(), t.max(), t.mean())
# print first 8 rows for quick look
print('first 8 preds/trues:')
for i in range(8):
    print(i, p[i,:], t[i,:])