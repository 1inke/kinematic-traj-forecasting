from common import *
import shutil
if __name__=='__main__':
    dest=HERE/'failures/v1_runtime_in_source_manifest';dest.mkdir(parents=True,exist_ok=False)
    paths=list(HERE.glob('*.py'))+[HERE/n for n in ('FAILURE.json','worker_state.json','exit_status.json','progress.json','config/sources_manifest.json','config/CONFIG_LOCK.json','config/analysis_config.json')]
    records={}
    for p in paths:
        target=dest/p.relative_to(HERE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha(p)==sha(target);records[p.relative_to(HERE).as_posix()]=sha(p)
    write(dest/'ARCHIVE_RECEIPT.json',dict(status='PRESERVED_EXACT',files=records,E1_not_run=not (HERE/'E1_LOCK.json').exists(),no_training=True))
    print('PRESERVED FAILURE V1',flush=True)
