from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np

im=Image.open(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003\frame_0161.jpg").convert("RGB")
# crop around each
pts={"24102_ped102":(409,192),"24063_ped51":(434,290)}
for name,(u,v) in pts.items():
    box=(max(0,u-60), max(0,v-80), min(720,u+60), min(576,v+80))
    c=im.crop(box)
    out=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")/f"crop_{name}.png"
    c.save(out); print(name, box, out)
# also analyze mean color in small patch at foot/body
arr=np.array(im)
for name,(u,v) in pts.items():
    # body patch above feet
    y1,y2=int(v-40),int(v-5); x1,x2=int(u-15),int(u+15)
    patch=arr[y1:y2,x1:x2]
    print(name, "mean_rgb", patch.mean(axis=(0,1)), "pos", u,v)
