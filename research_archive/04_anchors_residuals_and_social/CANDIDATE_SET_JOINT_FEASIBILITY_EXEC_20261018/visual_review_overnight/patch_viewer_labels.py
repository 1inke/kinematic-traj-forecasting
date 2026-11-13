from pathlib import Path
import re
p = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight\viewer.js")
t = p.read_text(encoding="utf-8")
pat1 = r'const scene=\(C\.scenes&&C\.scenes\[i\]\)\|\|"\?";\s*const flip=C\.display_y_sign\?C\.display_y_sign\[i\]:1;\s*const yLab=flip<0\?[^;]+;\s*const sc=axes\(svg,xd,yd,"[^"]+",yLab\);'
rep1 = (
    'const scene=(C.scenes&&C.scenes[i])||"?";\n'
    '  const dcode=(C.display_code&&C.display_code[i])||0;\n'
    '  const xLab=dcode===1?("display x (m) · "+scene+" = world y → right"):("world x (m) · "+scene+" → right");\n'
    '  const yLab=dcode===1?("display y (m) · "+scene+" = -world x → up"):("world y (m) · "+scene+" → up");\n'
    '  const sc=axes(svg,xd,yd,xLab,yLab);'
)
t2, n = re.subn(pat1, rep1, t, count=1)
print("replace1", n)
if n != 1:
    idx = t.find("display_y_sign")
    print(repr(t[idx-100:idx+300]))
    raise SystemExit(1)
pat2 = r'add\(svg,"text",\{x:sc\.L\+5,y:sc\.T\+15\},\$\("focus"\)\.checked\?"[^"]*":\(flip<0\?scene\+"[^"]*":scene\+"[^"]*"\)\);'
rep2 = 'add(svg,"text",{x:sc.L+5,y:sc.T+15},$("focus").checked?"local focus":(dcode===1?scene+" display(y,-x) · 1:1m":scene+" world(x,y) · 1:1m"));'
t3, n2 = re.subn(pat2, rep2, t2, count=1)
print("replace2", n2)
if n2 != 1:
    idx = t2.find("flip<0")
    print(repr(t2[idx-80:idx+200]))
    raise SystemExit(2)
p.write_text(t3, encoding="utf-8")
print("OK")
