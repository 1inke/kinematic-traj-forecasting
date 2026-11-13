"""Final report-only aggregation and SHA audit. No Torch/model/optimizer imports."""
import csv,json,hashlib,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;G0=BASE/'G0'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def source_check():
    m=read(BASE/'source_manifest.json');ext=read(BASE/'source_extension_prelaunch.json')
    for x in (m,ext):
        expected=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in x['files'].items()},sort_keys=True,separators=(',',':')).encode()).hexdigest();assert expected==x['digest']
        for v in x['files'].values():assert sha(v['path'])==v['sha256'],v['path']
    assert ext['base_digest']==m['digest']
    return dict(base_count=len(m['files']),base_digest=m['digest'],extension_count=len(ext['files']),extension_digest=ext['digest'])

def main():
    before=source_check();r=read(G0/'RESULT.json');d=read(G0/'DECISION.json');audit=read(G0/'REVIEW.json');packets=read(G0/'PACKETS.json')
    assert len(packets)==48 and audit['status']=='PASS_INDEPENDENT_SAVED_G0'
    assert read(BASE/'worker_state.json')['status']==read(BASE/'exit_status.json')['status']=='SUCCESS'
    for name,h in r['output_SHA'].items():assert sha(G0/name)==h,name
    output_before={name:sha(G0/name) for name in r['output_SHA']}
    table=[];summary=[];observations=[]
    for s in (42,43,44):
        rows=[p for p in packets if p['seed']==s];assert len(rows)==16
        valid=[p for p in rows if p['valid_double_group']];u=[p['arms']['U'] for p in valid];ps=[p['arms']['P'] for p in valid]
        conflict=[any(v>x['solver_tolerance'] for v in x['raw_Gd0']) for x in u]
        harms={a:float(np.mean([p['arms'][a]['h'] for p in valid])) for a in ('U','N','P')}
        for p in rows:
            for arm,x in p['arms'].items():
                table.append(dict(seed=s,batch=p['batch'],arm=arm,valid=p['valid_double_group'],batch_ID_SHA=p['batch_ID_SHA'],shared_count=x['counts'][0],rescued_count=x['counts'][1],R_shared_before=x['R_before'][0],R_rescued_before=x['R_before'][1],R_shared_after=x['R_after'][0],R_rescued_after=x['R_after'][1],h=x['h'],task_before=x['task_before'],task_after=x['task_after'],d0_norm=x['d0_norm'],dP_norm=x['dP_norm'],norm_ratio=x['norm_ratio'],active=str(x['active']),raw_Gd0=str(x['raw_Gd0']),raw_GdP=str(x['raw_GdP']),raw_positive_actual=x['cast']['raw_positive_actual'],cast_norm=x['cast']['cast_error_norm'],J_extra_ties=x['J_extra_ties'],J_component_ties=x['J_component_ties']))
        summary.append(dict(seed=s,valid=len(valid),conflict_count=sum(conflict),conflict_fraction=sum(conflict)/len(valid),mean_h=harms,P_reduction_vs_U=1-harms['P']/harms['U'],P_reduction_vs_N=1-harms['P']/harms['N'],P_changed_batches=sum(bool(x['active']) for x in ps),ordinary_harm_batches=sum(x['h']>0 for x in u),ordinary_harm_without_linear_conflict=sum(x['h']>0 and not flag for x,flag in zip(u,conflict)),P_harm_batches=sum(x['h']>0 for x in ps),P_norm_ratio_min=min(x['norm_ratio'] for x in ps),mean_task_delta={a:float(np.mean([p['arms'][a]['task_after']-p['arms'][a]['task_before'] for p in valid])) for a in ('U','N','P')}))
        for p,flag in zip(valid,conflict):
            if p['arms']['U']['h']>0 and not flag:
                observations.append(dict(seed=s,batch=p['batch'],U=p['arms']['U'],P=p['arms']['P'],reason='actual finite-step group harm despite nonpositive within-certificate first-order proposal'))
                break
    global_h={a:float(np.mean([v['mean_h'][a] for v in summary])) for a in ('U','N','P')}
    assert all(abs(global_h[a]-d['equal_seed_mean_h'][a])<2e-12 for a in global_h)
    assert sum(v['conflict_fraction']>=.2 for v in summary)==0
    resource=read(G0/'RESOURCE.json');assert all(x['status']=='PASS' for x in resource)
    after=source_check();assert before==after
    assert output_before=={name:sha(G0/name) for name in r['output_SHA']}
    result=dict(status='PASS_FINAL_SAVED_REPORT_SHA',at=datetime.now(timezone(timedelta(hours=8))).isoformat(),source_before=before,source_after=after,completed_output_files=len(output_before),completed_output_SHA=output_before,packets=48,arm_records=144,summary_by_seed=summary,all_seed_mean_h=global_h,P_reduction_vs_U=1-global_h['P']/global_h['U'],P_reduction_vs_N=1-global_h['P']/global_h['N'],G0_gates=d['gates'],decision='STOP_G0_NO_FULL_TRAINING',G1='NOT_TRIGGERED',G2_full_18_models='NOT_APPLICABLE_G0_STOP',formal_training=0,no_new_model_forward=True,no_optimizer_or_gradient_replay=True,saved_audit=audit,finite_step_counterexamples=observations,resource=dict(all_pass=True,observations=len(resource),retained_peak_private_bytes=max(o['snapshot']['peak_private'] for o in resource),retained_peak_RSS_bytes=max(o['snapshot']['peak_rss'] for o in resource),minimum_available_physical_bytes=min(o['snapshot']['physical'] for o in resource),minimum_available_commit_bytes=min(o['snapshot']['commit'] for o in resource)),FD_preflight_scope_note='Actual preflight used four fixed RNG directions per objective (12 total) rather than one common bank of4 across all3 objectives. Original raw tests retained; no post-result model replay; this execution-detail deviation cannot be claimed as common-bank compliance and does not alter the two independently failed G0 gates.')
    (HERE/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (HERE/'ALL48_PACKET_ARM_SUMMARY.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps({k:result[k] for k in ('status','source_after','completed_output_files','summary_by_seed','all_seed_mean_h','P_reduction_vs_U','P_reduction_vs_N','resource')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
