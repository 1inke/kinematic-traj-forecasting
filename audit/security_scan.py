"""Scan new maintained code/configs only; print locations, never secret values.

Not a substitute for the hosting provider's secret scanning of Git history.
"""
import argparse,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ABSOLUTE=re.compile(r'[A-Za-z]:[\\/]+Users[\\/]|/home/')
SECRET=re.compile(r'(?i)(?:password|passwd|api_key|access_token|smtp_password)\s*[:=]\s*[\"\']([^\"\']{8,})[\"\']')
EXT={'.py','.json','.yaml','.yml','.toml','.md','.cfg','.ini','.txt'}

def scan():
    files=[]
    for base in ['src','experiments','config','audit','tests','.github']:
        directory=ROOT/base
        if directory.exists():files.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in EXT and '__pycache__' not in p.parts and p.name!='paths.yaml')
    errors=[]
    for p in files:
        for lineno,line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
            if p.is_relative_to(ROOT/'src') or p.is_relative_to(ROOT/'experiments'):
                if ABSOLUTE.search(line):errors.append((p.relative_to(ROOT).as_posix(),lineno,'hardcoded user path'))
            if SECRET.search(line):errors.append((p.relative_to(ROOT).as_posix(),lineno,'possible embedded credential'))
    # Existing environment/file exclusions must remain effective; never open secrets.
    return errors,len(files)

if __name__=='__main__':
    errors,n=scan();print(f'Scanned {n} maintained text files; findings={len(errors)}')
    for path,line,kind in errors:print(f'{path}:{line}: {kind}')
    raise SystemExit(bool(errors))
