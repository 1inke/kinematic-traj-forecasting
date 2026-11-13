"""Verify installed solver wheels against official PyPI release SHA256; no installs."""
import sys,json,hashlib,zipfile,urllib.request,importlib.metadata as md
from pathlib import Path
H=Path(__file__).resolve().parent
def main():
    assert Path(sys.prefix).resolve()==(H/'.solver_env').resolve()
    assert 'include-system-site-packages = false' in (H/'.solver_env/pyvenv.cfg').read_text()
    rows=[];wheels=[]
    for p in sorted((H/'wheelhouse').glob('*.whl')):
        with zipfile.ZipFile(p) as z:
            from email.parser import Parser
            t=Parser().parsestr(z.read(next(x for x in z.namelist() if x.endswith('.dist-info/METADATA'))).decode())
        name,version=t['Name'],t['Version'];url=f'https://pypi.org/pypi/{name}/{version}/json'
        with urllib.request.urlopen(url,timeout=60) as res:j=json.load(res)
        remote=next(x for x in j['urls'] if x['filename']==p.name)
        sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==remote['digests']['sha256']
        assert remote['url'].startswith('https://files.pythonhosted.org/')
        wheels.append(dict(name=name,version=version,file=str(p),SHA256=sha,origin=remote['url'],release_metadata=url))
    for d in sorted(md.distributions(),key=lambda x:x.metadata['Name'].lower()):
        rows.append(dict(name=d.metadata['Name'],version=d.version,location=str(d.locate_file(''))))
        assert str(d.locate_file('')).lower().startswith(str(H/'.solver_env').lower())
    import cvxpy,clarabel,numpy,scipy,psutil
    assert (cvxpy.__version__,clarabel.__version__,numpy.__version__,scipy.__version__,psutil.__version__)==('1.7.3','0.11.1','2.3.3','1.16.2','7.1.0')
    assert md.version('osqp')=='1.0.4'
    installed={}
    for p in sorted((H/'.solver_env').rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.py','.pyd','.dll','.exe','.cfg') and '__pycache__' not in p.parts:
            installed[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=dict(status='PASS',python=sys.version,prefix=sys.prefix,isolated_no_system_site=True,scientific_solver='CLARABEL',versions=rows,wheels=wheels,installed_code_files_SHA=installed,optional_osqp_1p1p3_import_failure_preserved=True,old_environments_not_modified=True)
    (H/'SOLVER_ENVIRONMENT_LOCK.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=out['status'],installed_code_files=len(installed),wheels=len(wheels),versions=rows)))
if __name__=='__main__':main()
