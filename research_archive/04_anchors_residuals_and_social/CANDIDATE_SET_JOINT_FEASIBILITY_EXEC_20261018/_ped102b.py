from pathlib import Path
import numpy as np

# classic train format: frame ped x y
p = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\eth\students003_train.txt")
print("head:")
print("\n".join(p.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]))
rows=[]
for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
    a=line.split()
    if len(a)<4: continue
    try:
        fr,ped,x,y=float(a[0]),int(float(a[1])),float(a[2]),float(a[3])
    except: continue
    if ped==102: rows.append((fr,x,y))
print("n",len(rows),"range",rows[0] if rows else None, rows[-1] if rows else None)
for want in [144,154,164,174,184,194,204,214]:
    print(want, [r for r in rows if abs(r[0]-want)<1e-6])
# H
for hp in Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy").rglob("H.txt"):
    if "univ" in str(hp).lower() or "student" in str(hp).lower():
        print("H", hp)
        print(hp.read_text()[:300])
        break
else:
    hs=list(Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy").rglob("H.txt"))
    print("all H", hs[:10])
    for hp in hs[:3]:
        print(hp, hp.read_text()[:200])
# imgs folder
imgd=Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003")
print("imgs isdir", imgd.is_dir(), "listdir", list(imgd.iterdir())[:5] if imgd.exists() else None)
