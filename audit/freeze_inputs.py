"""Capture checked input digests, NOT author protocol approval."""
from f16.paths import resolve_paths
from f16.io import sha256,write_json

EXPECTED={
'cache':'f8e6769c7e28b0d5ad4c108989246d6e724c078190102ae90443e62600d5a3ea',
'labels':'01c8226f6bc3e7103c61f52294ac109ad2d585be2500a25d1601e08efe487efc',
'metadata':'9e0e2f581d05487e4905fc2fe34c963458cace0254daf5289e12aaea9ac49591',
'splits':'349659306a435a3a70e65d75d08d618a2aa8bc76fca38142c3433ab5678e12e7',
'encoder':'f6c232e4b05e4e8db2a01c7fd0880beecd835b59352b81ba08132c8b610e87f0'}

def main():
    p=resolve_paths();files={}
    for name,digest in EXPECTED.items():
        if sha256(p[name])!=digest:raise ValueError('Historical input digest mismatch: '+name)
        files[p[name].relative_to(p['root']).as_posix()]=digest
    write_json(p['root']/'config/data_manifest.v1.json',dict(state='INPUT_DIGESTS_VERIFIED_NOT_AUTHOR_FREEZE',files=files))
    print('Five historical input digests verified; no protocol author-freeze implied.')

if __name__=='__main__':main()
