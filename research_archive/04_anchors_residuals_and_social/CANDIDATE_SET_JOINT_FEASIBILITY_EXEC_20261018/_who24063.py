import numpy as np
from pathlib import Path

# metadata for 24063
meta_paths=[
 r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018",
]
# find metadata
import os
for root,dirs,files in os.walk(r"<WORKSPACE_ROOT>"):
  if "metadata.npz" in files and "F16" in root.replace("\\","/"):
    print("meta", root)
    break

# eth sample 24063
eth=np.load(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\eth_data_train.npy", allow_pickle=True)
# often object array of dicts or structured
print("eth type", type(eth), getattr(eth,"shape",None), eth.dtype)
# try common ASTRA format
if eth.dtype==object:
  s=eth[24063]
  print("sample keys/type", type(s), s if not hasattr(s,"keys") else list(s.keys())[:20] if isinstance(s,dict) else s.shape)
else:
  print(eth.shape)
