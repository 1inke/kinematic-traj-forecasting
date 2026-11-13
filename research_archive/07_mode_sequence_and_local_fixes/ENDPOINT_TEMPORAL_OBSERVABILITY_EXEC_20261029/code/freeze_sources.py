"""Extend an inherited source map WITHOUT replacing any inherited digest."""
import argparse
from pathlib import Path
from io_contract import sha, json_read, json_new, code_digest


def freeze(parent, project_root, extras, out):
    root = Path(project_root).resolve()
    old = json_read(parent)
    if not isinstance(old.get('files'), dict):
        raise ValueError('parent must explicitly contain files->SHA; no guessed schema')
    files = {}
    for name, digest in old['files'].items():
        p = (Path(name) if Path(name).is_absolute() else root/name).resolve()
        if sha(p) != digest:
            raise ValueError('inherited source conflict: '+str(p))
        if str(p) in files and files[str(p)] != digest:
            raise ValueError('duplicate path with conflicting inherited digest')
        files[str(p)] = digest
    for p in list(Path(__file__).resolve().parent.glob('*.py'))+[Path(p).resolve() for p in extras]:
        value = sha(p)
        key = str(p.resolve())
        if key in files and files[key] != value:
            raise ValueError('refuse inherited hash replacement: '+key)
        files[key] = value
    json_new(out, dict(files=files, inherited_manifest_sha=sha(parent),
                      inherited_file_count=len(old['files']), code_digest=code_digest(),
                      status='SOURCE_LOCK_ONLY_NOT_PREFLIGHT_PASS'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--project-root', required=True)
    p.add_argument('--extra', nargs='+', required=True, help='resource adapter and frozen design/config files')
    p.add_argument('--out', required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if args.run:
        freeze(args.parent, args.project_root, args.extra, args.out)
    else:
        print('PLAN_ONLY: no source lock created')
