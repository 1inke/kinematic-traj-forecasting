from pathlib import Path
p = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt")
rows = []
with p.open(encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.split()
        if len(parts) < 4:
            continue
        frame, ped = int(float(parts[0])), int(float(parts[1]))
        x, y = float(parts[2]), float(parts[3])
        if ped == 102:
            rows.append((frame, x, y))
print("n_rows ped102", len(rows))
if rows:
    print("frame range", rows[0][0], rows[-1][0])
for want in [144, 154, 164, 174, 184, 194, 204, 214, 224, 234]:
    hit = [r for r in rows if r[0] == want]
    print("frame", want, hit)
fs = sorted({r[0] for r in rows})
print("frames 120-260", [f for f in fs if 120 <= f <= 260])
print("all ped102 first20", rows[:20])
for hp in [
    Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\H.txt"),
    Path(r"<LOCAL_DESKTOP>\DRSE\eth_dataset\labels\univ\H.txt"),
]:
    print("H exists", hp, hp.exists())
    if hp.exists():
        print(hp.read_text(encoding="utf-8", errors="ignore")[:400])
univ = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ")
print("univ files", sorted(x.name for x in univ.iterdir()) if univ.exists() else None)
