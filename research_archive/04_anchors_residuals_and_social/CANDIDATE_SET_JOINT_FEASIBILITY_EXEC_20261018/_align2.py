import numpy as np
from pathlib import Path
# Match world ped51 to vsp by comparing world displacement direction with all splines at scaled frames
# world ped51 from txt
rows=[]
for line in Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt").read_text(encoding="utf-8",errors="ignore").splitlines():
    a=line.split()
    if len(a)<16: continue
    if int(float(a[1]))!=51: continue
    rows.append((float(a[0]), float(a[13]), float(a[15])))
print("txt ped51 n", len(rows), "f", rows[0][0], rows[-1][0], "xy0", rows[0][1:], "xy_end_around161", [r for r in rows if abs(r[0]-161)<1e-6])
# eth_data row 24063 world already known
# Find which spline best matches displacement at vf 1490->1610 in pixel space correlated with world
# Actually check: world at 149 and 161
w149=[r for r in rows if abs(r[0]-149)<1e-6][0]
w161=[r for r in rows if abs(r[0]-161)<1e-6][0]
print("world 149", w149, "161", w161, "d", (w161[1]-w149[1], w161[2]-w149[2]))
