from pathlib import Path
p=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\_univ_rar\students003.vsp")
raw=p.read_bytes()
print("size", len(raw), "head ascii", raw[:200])
# try text
try:
  t=p.read_text(encoding="utf-8", errors="replace")
  print("text head\n", t[:500])
  print("count 102", t.count("\n102\t"), t.count(" 102 "), t.count("\t102\t"))
except Exception as e:
  print(e)
# search for ped id patterns as binary int
import struct, re
# as text lines
lines=t.splitlines()
print("nlines", len(lines), "line0", lines[0] if lines else None)
for i,l in enumerate(lines[:30]):
  print(i, repr(l[:120]))
