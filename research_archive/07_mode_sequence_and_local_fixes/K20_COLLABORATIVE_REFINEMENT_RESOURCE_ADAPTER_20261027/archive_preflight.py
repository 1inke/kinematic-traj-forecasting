from core import *
import shutil
if __name__=='__main__':
    for name in ('PREFLIGHT_GPU.json','PREFLIGHT_GPU_RESOURCE.json','PREFLIGHT_CPU.json','PREFLIGHT_CPU_RESOURCE.json'):
        p=HERE/name
        if not p.exists():continue
        f=HERE/'preflight_history'/source();f.mkdir(parents=True,exist_ok=True);target=f/name
        if target.exists():assert sha256(target)==sha256(p)
        else:shutil.copyfile(p,target)
