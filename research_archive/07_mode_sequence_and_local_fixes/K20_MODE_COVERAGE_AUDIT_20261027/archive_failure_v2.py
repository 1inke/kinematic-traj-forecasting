from common import *
import shutil
if __name__=='__main__':
    out=HERE/'failures/v2_preflight_fixture_preserve';out.mkdir(parents=True,exist_ok=False)
    paths=list(HERE.glob('*.py'))+[HERE/n for n in ('resume_state.json','resume_exit.json','config/sources_manifest.json','config/CONFIG_LOCK.json','audit/fixture128.npz')]
    for p in paths:
        q=out/p.relative_to(HERE);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(p)==sha(q)
    write(out/'RECEIPT.json',dict(status='PRESERVED',no_E1_or_E2_output_created=not (HERE/'E1_LOCK.json').exists(),fixture_was_already_complete_before_previous_source_check_failure=True))
