import numpy as np
from pathlib import Path
RAW = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse/eth_data_train.npy")
NUM = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse/eth_num_train.npy")
z = np.load(RAW, mmap_mode='r')
print("RAW shape", z.shape, "dtype", z.dtype)
n = np.load(NUM)
print("NUM shape", n.shape, n.dtype, "minmax", n.min(), n.max())
# sample 24102 row from metadata alignment - use eth_train sample_id index
import numpy as np
train = np.load(r"<WORKSPACE_ROOT>\REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES\results\cache\train_dev\eth_train.npz")
row = int(np.where(train["sample_id"]==24102)[0][0])
print("train row", row, "sample_id", train["sample_id"][row])
print("RAW[row] shape", z[row].shape)
print("NUM[row]", n[row] if n.ndim==1 else n[row])
# print channels
arr = np.array(z[row])
print("arr sample ego past-ish", arr[0,:8,-2:])  # guess last2 are xy
print("arr[0,0]", arr[0,0])
print("arr[1,0]", arr[1,0] if arr.shape[0]>1 else None)
