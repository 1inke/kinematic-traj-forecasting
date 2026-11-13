"""New-run output contracts, separate from incomplete legacy checkpoints."""
import platform,subprocess,sys
from pathlib import Path
from .io import sha256,write_json,verify_manifest,read_json

def environment_record(root,protocol,data_manifest,sources):
    root=Path(root);result={}
    for p in sources:
        p=Path(p);result[p.relative_to(root).as_posix()]=sha256(p)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain'],cwd=root,text=True,capture_output=True,check=True).stdout.splitlines()
    # Machine-specific pip freeze is kept local/ignored, not used as scientific data.
    return dict(git_commit=commit,git_worktree_dirty=bool(dirty),python=platform.python_version(),platform=platform.platform(),protocol_sha256=sha256(protocol),data_manifest_sha256=sha256(data_manifest),source_sha256=result)

def verify_completed(root,manifest,expected_contract):
    value=read_json(manifest)
    if value.get('contract')!=expected_contract:raise ValueError('Completed output source/ID/protocol contract mismatch')
    if not value.get('files'):raise ValueError('Empty completion manifest')
    verify_manifest(value['files'],root)
    return value
