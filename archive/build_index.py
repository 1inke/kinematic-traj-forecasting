"""Index pre-protocol directories without moving or reinterpreting them."""
import csv,re,subprocess,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    folders=subprocess.run(['git','-c','core.quotepath=false','ls-tree','-d','--name-only','HEAD'],cwd=ROOT,check=True,text=True,capture_output=True,encoding='utf-8').stdout.splitlines();rows=[]
    for folder in sorted(folders):
        path=ROOT/folder
        if not path.is_dir() or folder.startswith('.'):continue
        sources=[p for p in [path/'RESULT_ANALYSIS.md',path/'ANALYSIS.md',path/'README.md'] if p.exists()]
        source=sources[0] if sources else None;snippet='NOT_REVIEWED_IN_THIS_MIGRATION'
        if source:
            lines=source.read_text(encoding='utf-8-sig',errors='replace').splitlines()
            snippet=next((line[:200] for line in lines[:80] if re.search(r'(?:verdict|判定|结论|状态)\s*[:：]',line,re.I)),snippet)
        dates=re.findall(r'20\d{6}',folder)
        rows.append(dict(directory=folder,date_from_name=dates[-1] if dates else '',direction=folder.split('_')[0],conclusion_reference=snippet,report=source.relative_to(ROOT).as_posix() if source else '',report_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source else '',pre_protocol=True,physical_location='ORIGINAL_PATH_UNCHANGED'))
    with (ROOT/'archive/INDEX.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Indexed {len(rows)} historical directories; moved 0.')

if __name__=='__main__':main()
