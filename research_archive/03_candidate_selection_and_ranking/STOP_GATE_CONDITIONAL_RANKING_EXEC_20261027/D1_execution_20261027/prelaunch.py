from core import *
import shutil,ast
def main():
    g=guard('CPU');verify_prepared();h=verify_manifest(HERE);a=read_json(HERE/'PREFLIGHT_GPU.json');b=read_json(HERE/'PREFLIGHT_CPU.json');assert a['status']==b['status']=='PASS' and a['source']==b['source']==h
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    # Nine fixed runs plus three opt readings/run, 27 final populations and
    # independent complete-link/pooled/ranking/source/bootstrap IO; no hard cap.
    train=sum(a['benchmark'][arm]['mean_seconds']*1024*3 for arm in ARMS)
    savecost=(a['saved128_GPU_seconds']+b['actual_CPU_save_audit_seconds'])/128/3
    final_windows=3*3*(4096+3825+1856);population=savecost*final_windows
    readings=3*3*4096*a['saved128_GPU_seconds']/128/3
    planning=1.3*(train+population+readings)+240
    newbytes=2*1024**3;free=shutil.disk_usage(HERE).free;assert free>=1.3*newbytes
    assert verify_manifest(HERE)==h
    write_json(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',source=h,source_count=len(read_json(HERE/'source_manifest.json')['files']),planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],training_seconds=train,save_independent_audit_seconds=population,opt_intermediate_reading_allowance=readings,actual128measured_not_production_guarantee=True,disk_free=free,conservative_new_bytes=newbytes,all_original_gates_unmodified=True,DA1_same_assistant_not_external=dict(verdict='PASS_LIMITED',limits=['objective organization not pure STOP coefficient','full S and native both required','old cost unresolved','F16seen/repeated hold/3 correlated seeds'])))
    write_json(HERE/'PRODUCTION_SOURCE_LOCK.json',dict(source=h,count=len(read_json(HERE/'source_manifest.json')['files']),at=now()));resource(HERE/'PRELAUNCH_RESOURCE.json',g);print('PASS PRELAUNCH seconds',planning,flush=True)
if __name__=='__main__':main()
