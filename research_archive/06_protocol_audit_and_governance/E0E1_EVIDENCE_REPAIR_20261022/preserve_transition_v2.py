"""Keep failed v1 source/evidence; extend lock for explicit implementation fix."""
import shutil
from common import *

def main():
    verify_sources();fail=HERE/'failures/preflight1_batch_kernel';fail.mkdir(parents=True,exist_ok=True)
    for name in ['preflight.py','common.py','protocol.json','source_manifest_before.json','PREFLIGHT_FAILURE_BATCH_DIAGNOSIS.json']:
        destination=fail/name
        if not destination.exists():shutil.copy2(HERE/name,destination)
    files={p.relative_to(ROOT).as_posix():sha256(p) for p in [HERE/'stable_forward_v2.py',HERE/'preflight_v2.py',HERE/'run_frozen_v2.py',HERE/'preserve_transition_v2.py']}
    value=dict(status='IMPLEMENTATION_FIX_NOT_SCIENCE_CHANGE',reason='128 vs32 float32 GEMM max4.2915344e-6 > registered2e-6 engineering check; use constant256 padded compute, keep tolerance and all scientific definitions',scientific_protocol_sha_unchanged=sha256(HERE/'protocol.json'),old_source_manifest_preserved=sha256(HERE/'source_manifest_before.json'),all_original_locked_files_unchanged=True,files=files,source_change='new implementation files only; original failing source retained',old_inner_counts_must_still_reconcile_exact=True)
    write_json(HERE/'IMPLEMENTATION_TRANSITION_V2.json',value)
    print('Preserved failed source and added explicit constant-compute implementation lock.')

if __name__=='__main__':main()
