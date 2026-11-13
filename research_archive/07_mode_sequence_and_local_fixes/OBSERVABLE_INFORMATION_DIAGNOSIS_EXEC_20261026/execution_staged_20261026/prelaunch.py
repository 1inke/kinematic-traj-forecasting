"""Last current-source review and immutable lock, never changes science."""
from base import *
import ast,shutil

def main():
    if (HERE/'PRODUCTION_SOURCE_LOCK.json').exists():raise FileExistsError('Already production locked')
    s=verify();a=read_json(HERE/'PREFLIGHT_MLP.json');b=read_json(HERE/'PREFLIGHT_CPU.json');assert a['status']==b['status']=='PASS' and a['source']==b['source']==s
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    original=read_json(HERE.parent/'steps12_execution_20261026/protocol.json');current=read_json(HERE/'protocol.json')
    for k,v in original.items():assert current[k]==v,'Scientific protocol changed '+k
    prepared=read_json(HERE/'prepared/done.json');completed(HERE/'prepared',prepared['contract'])
    free=shutil.disk_usage(HERE).free;assert free>=1.3*read_json(HERE/'RESOURCE_STAGE_PLAN.json')['disk_planning_bytes']
    perms=read_json(HERE/'prepared/PERMUTATION_MAP.json');assert perms['blocked_seeds']==[42,43,44]
    # Estimate serial actual cost: measured train, fixed trees, all401 save/review,
    # 6tiny+3MLP startup overhead and CPU subprocess IO/source reserves.
    train=3*512*(a['step_seconds']['I']+a['step_seconds']['S'])+3*1024*a['step_seconds']['M_full']
    trees=3*sum(v['seconds']*4096/256 for v in b['tree_subset_timings'].values())
    windows=24*128+9*(4096+3825+1856);saved=windows/128*b['CPU128_independent_save_review_seconds']*3
    startup=9*20+60*2;sourceIO=180;planning=train+trees+saved+startup+sourceIO
    eta=dict(planning_seconds=planning,range_seconds=[planning*.7,planning*1.5],components=dict(train=train,trees=trees,saved_geometry_cluster_motion_and_IO=saved,serial_process_startups=startup,final_sourceIO=sourceIO),not_guarantee=True)
    report=dict(status='PASS',source=s,at=now(),scientific_protocol_original_fields_EXACT=True,actual_MLP_CPU_preflight=True,T_perm_all3_BLOCKED_original_rule=True,other9admissible=True,DA2='Tiny memorization is not generalization; BCE reduction is not decision success; keep all fixed outcomes.',DA3='Trees not equal capacity; geometry includes history; permutation unavailable, no observation increment claim; historical hold/inner exploratory.',review='same assistant inline adversarial review, not external',no_formal_fit_before_lock=True,no_delete=True,disk_free=free,ETA=eta,resource_1p3_stage_actual_preflight=True)
    write_json(HERE/'PRELAUNCH_REVIEW.json',report);write_json(HERE/'PRODUCTION_SOURCE_LOCK.json',dict(digest=s,files_count=len(read_json(HERE/'source_manifest.json')['files']),locked_at=now(),no_further_science_edits=True));print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise
