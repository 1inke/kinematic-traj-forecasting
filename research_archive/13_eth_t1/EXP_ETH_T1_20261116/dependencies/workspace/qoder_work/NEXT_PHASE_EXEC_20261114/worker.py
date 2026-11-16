"""One finite on-demand worker: E1, seal, preflight, E2--E4, saved-data report."""
import os
import subprocess
import sys
import time
import traceback
from common import *
from analysis_phase import e1a,choose_budget,seal_contract,final_report,ARMS,prediction_file


def atomic(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)


def run(label,script,args):
    # Each run has a fresh process, so initialization and global dropout/augmentation RNG are isolated.
    import resource_guard as rg
    estimate=read_json(HERE/'PREFLIGHT_E1.json')['estimated_peak_bytes']
    admission=rg.decide(rg.snapshot(),estimate,'startup')
    write_json(HERE/'admission'/f'{label}.json',dict(at=now(),**admission))
    if admission['status']!='PASS':raise MemoryError('PRE_POPEN_CURRENT_RESOURCE_ADMISSION')
    log=HERE/'logs';log.mkdir(exist_ok=True)
    command=[sys.executable,'-B',str(HERE/script),*args]
    with (log/f'{label}.stdout.log').open('xb') as stdout,(log/f'{label}.stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        atomic(HERE/'CHILD_STATE.json',dict(at=now(),label=label,pid=child.pid,command=command,state='RUNNING'))
        exitcode=child.wait()
    write_json(HERE/'receipts'/f'{label}.json',dict(at=now(),command=command,pid=child.pid,exit_code=exitcode,
        stdout=sha(log/f'{label}.stdout.log'),stderr=sha(log/f'{label}.stderr.log')))
    if exitcode!=0:raise RuntimeError(f'CHILD_FAILURE {label} exit={exitcode}; see logs')


def spec(name,seed,epochs,arm='B0',order=None,split='fit'):
    return dict(name=name,arm=arm,seed=seed,order=seed if order is None else order,epochs=epochs,split=split)


def train(s):run(s['name']+'_train','execution.py',['train',json.dumps(s)])
def evaluate(s):run(s['name']+'_eval','execution.py',['eval',json.dumps(s)])


def main():
    started=time.perf_counter()
    # Exclusive marker: never restart or rerun a partially completed finite study automatically.
    fd=os.open(HERE/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    write_json(HERE/'WORKER_START.json',dict(at=now(),pid=os.getpid(),args=sys.argv,source=sha(HERE/'SOURCE_MANIFEST.json')))
    try:
        manifest=read_json(HERE/'SOURCE_MANIFEST.json');verify(manifest['files'])
        if read_json(HERE/'PREFLIGHT_E1.json')['status']!='PASS':raise ValueError('E1_PREFLIGHT_REQUIRED')
        run('numpy_parts','test_numpy_parts.py',[])
        run('make_calib_split','make_calib_split.py',['--fit',str(HERE/'C0/fit.npz'),'--out',str(HERE/'E1/calib_split.npz')])
        jobs=[spec(f'E1A_s{s}',s,8) for s in range(43,47)]
        jobs+=[spec('E1A_order_only',42,8,order=43),spec('E1A_init_only',43,8,order=42),spec('E1A_rerun',42,8)]
        for s in jobs:train(s)
        write_json(HERE/'E1/A_ALL_FINAL_LOCK.json',dict(at=now(),files=seal([HERE/'runs'/s['name']/'final.pt' for s in jobs])))
        for s in jobs:evaluate(s)
        run('e1a_analysis','analysis_phase.py',['e1a'])
        bjobs=[spec(f'E1B_E{E}_s{s}',s,E,split='fit85') for E in (8,16,32,64) for s in (42,43)]
        for s in bjobs:train(s);evaluate(s)
        E=choose_budget()
        run('calibration_figure','plot_calibration.py',[])
        refs=['OLD42']+[f'E1A_s{s}' for s in range(43,47)]
        if E>8:
            cjobs=[spec(f'E1C_E{E}_s{s}',s,E) for s in range(42,47)]
            for s in cjobs:train(s)
            write_json(HERE/'E1/C_ALL_FINAL_LOCK.json',dict(at=now(),files=seal([HERE/'runs'/s['name']/'final.pt' for s in cjobs])))
            for s in cjobs:evaluate(s)
            refs=[s['name'] for s in cjobs]
        seal_contract(E,refs)
        atomic(HERE/'CURRENT.json',dict(at=now(),stage='E1_COMPLETE_CONTRACT_SEALED',E_star=E))
        # Method preflight happens only after E1 contract has been frozen.
        if not (HERE/'PREFLIGHT_METHODS.json').exists():
            run('method_preflight','preflight_phase.py',['methods'])
        elif read_json(HERE/'PREFLIGHT_METHODS.json')['status']!='PASS':raise ValueError('METHOD_PREFLIGHT_FAILURE')
        for seed,b in zip((42,43,44),refs[:3]):
            for split in ('fit','dev'):
                run(f'E2zero_s{seed}_{split}','respline_saved.py',['--data',str(HERE/'C0'/f'{split}.npz'),'--pred',str(prediction_file(b,split)),
                    '--outdir',str(HERE/'E2_zero'/f's{seed}_{split}')])
                run(f'E2zero_analysis_s{seed}_{split}','diagnostics_phase.py',['zero',str(seed),split,'--reference',str(prediction_file(b,split))])
        method_jobs=[spec(f'{a}_s{s}',s,E,arm=a) for a in ARMS for s in (42,43,44)]
        for s in method_jobs:train(s)
        write_json(HERE/'METHOD_ALL_FINAL_LOCK.json',dict(at=now(),E_star=E,files=seal([HERE/'runs'/s['name']/'final.pt' for s in method_jobs]),
            E2E_V2=sha(HERE/'E2E_V2_CONTRACT.json')))
        for s in method_jobs:evaluate(s)
        for s in method_jobs:
            for split in ('fit','dev'):run(s['name']+'_geometry_'+split,'diagnostics_phase.py',['geometry',s['name'],split])
        run('cost_B0','cost_methods.py',[refs[0],'B0'])
        for arm in ARMS:run('cost_'+arm,'cost_methods.py',[f'{arm}_s42',arm])
        atomic(HERE/'CURRENT.json',dict(at=now(),stage='FINAL_SAVED_DATA_ANALYSIS',E_star=E))
        run('final_analysis','analysis_phase.py',['final',str(E),json.dumps(refs)])
        verify(manifest['files']);verify(read_json(HERE/'E2E_V2_SEAL.json')['files'])
        write_json(HERE/'FINAL_OUTPUT_SEAL.json',dict(at=now(),files=seal([p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.pt','.csv','.md') and p.name not in ('CURRENT.json','CHILD_STATE.json') and 'design_original' not in p.parts])))
        done=dict(status='SUCCESS',at=now(),E_star=E,new_training_runs=len(jobs)+len(bjobs)+(5 if E>8 else 0)+len(method_jobs),
                  seconds=time.perf_counter()-started,no_test=True,no_baseline_upgrade=True,no_factor_combination=True)
        write_json(HERE/'RESULT.json',done);write_json(HERE/'WORKER_EXIT.json',dict(**done,exit_code=0))
        atomic(HERE/'CURRENT.json',done)
    except BaseException as err:
        failure=dict(status='FAILURE',at=now(),error=repr(err),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
        write_json(HERE/'FAILURE.json',failure);write_json(HERE/'WORKER_EXIT.json',dict(**failure,exit_code=1));atomic(HERE/'CURRENT.json',failure)
        raise


if __name__=='__main__':main()
