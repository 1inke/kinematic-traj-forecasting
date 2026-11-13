"""Bind immutable completed P0 dependencies and count real tiny strata."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from io_contract import load_npz,json_read,json_new,sha,verify_manifest,checked_seal,code_digest


def main():
    verify_manifest(ROOT/'source_manifest.json')
    ext=verify_manifest(ROOT/'p0_resume_source_extension.json')
    reused=json_read(ROOT/'P0_TIME_train_labels_resume_20261029/PARTIAL_REUSE.json')['files']
    frozen=ext['partial_shards_frozen']
    if len(frozen)!=29 or len(reused)!=29:raise AssertionError('partial source registry count')
    for item in reused:
        # Normalize absolute-vs-project-relative paths only; never replace a SHA.
        key=next((k for k in frozen if Path(k).resolve()==Path(item['path']).resolve()),None)
        if key is None or item['sha']!=frozen[key] or sha(item['path'])!=frozen[key]:raise AssertionError('partial source provenance')
    review=json_read(ROOT/'p0_completion_review/REVIEW.json')
    files={str(ROOT/name):digest for name,digest in review['files'].items()}
    for name,digest in files.items():
        if sha(name)!=digest:raise AssertionError('P0 saved output SHA')
    registry=json_read(ROOT/'P0_STAGE_REGISTRY.json');strata={}
    for b in ('END','TIME'):
        strata[b]={}
        for pop in ('train','val'):
            root=ROOT/registry[b][pop]['labels'];checked_seal(root)
            target=np.concatenate([load_npz(p,('target',))['target'] for p in sorted(root.glob('[0-9]*.npz'))])
            active=np.any(target!=0,axis=(1,2))
            strata[b][pop]=dict(windows=len(target),active_windows=int(active.sum()),inactive_windows=int((~active).sum()),
                active_candidates=int(np.any(target!=0,axis=2).sum()),candidate_denominator=int(target.shape[0]*20))
            for kind in ('masks','labels'):
                directory=ROOT/registry[b][pop][kind];checked_seal(directory)
                for p in directory.iterdir():
                    if p.is_file():files[str(p)]=sha(p)
    for p in (ROOT/'prepared').iterdir():
        if p.is_file():files[str(p)]=sha(p)
    for name in ('source_manifest.json','SOURCE_EQUIVALENCE.json','p0_resume_source_extension.json','P0_STAGE_REGISTRY.json','P0_RESULT.json',
        'P0_WORKER_SUCCESS.json','p0_os_exit.json','p0_completion_review/REVIEW.json','learning_preflight.py','prepare_learning.py'):
        files[str(ROOT/name)]=sha(ROOT/name)
    json_new(ROOT/'TINY_STRATUM_ADMISSION.json',dict(status='MEASURED_BEFORE_ANY_TINY_FIT',strata=strata,
        registered_policy='64 active windows plus 64 inactive/other-stratum windows; no val/held substitution',
        inherited_partial_shards_verified=29,held_GT_opened=False))
    files[str(ROOT/'TINY_STRATUM_ADMISSION.json')]=sha(ROOT/'TINY_STRATUM_ADMISSION.json')
    json_new(ROOT/'learning_source_extension.json',dict(status='EXPLICIT_P0_COMPLETED_DEPENDENCY_AND_LEARNING_PREFLIGHT_SOURCE_LOCK',
        original_manifest_sha=sha(ROOT/'source_manifest.json'),inherited_original_source_count=4451,code_digest=code_digest(),files=files,
        no_inherited_digest_replaced=True,partial_provenance_pass=True))
    print('Learning dependency lock and all29 original partial provenance PASS',strata,flush=True)


if __name__=='__main__':main()
