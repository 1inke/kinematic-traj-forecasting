import numpy as np
from pathlib import Path
meta=np.load(r"<WORKSPACE_ROOT>\F16_OBSERVABLE_CUE_DIAGNOSIS_20261003\metadata.npz", allow_pickle=True)
i=int(np.where(meta["sample_id"]==24063)[0][0])
sid=int(meta["sample_id"][i]); seq=str(meta["sequence"][i]); ped=int(meta["pedestrian_id"][i]); sf=int(meta["start_frame"][i])
t0=sf+7; tend=sf+19
print(dict(sample_id=sid, sequence=seq, ped=ped, start_frame=sf, t0_frame=t0, pred_end_frame=tend,
           t0_s=t0*0.4, pred_end_s=tend*0.4, video_t0_frame=t0*10, video_end_frame=tend*10))
