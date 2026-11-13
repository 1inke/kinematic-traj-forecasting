"""CPU saved-metric/trace/source review only: no torch, model, optimizer, or new CI."""
import json,hashlib,math,time,ctypes,os
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
SEEDS=(42,43,44);ARMS=('ordinary','projected');GROUPS=('overall','stable','transition','change')
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    assert min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE: physical/commit reserve'
def read(p):return json.loads((H/p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def verify(files):
    for p,h in files.items():guard();assert sha(p)==h,('SHA_CHANGED',p)
def save(p,z):
    p=H/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def mean(x):return math.fsum(x)/len(x)
def close(a,b):assert abs(a-b)<=2e-12,(a,b)
def main():
    t=time.perf_counter();guard();files=read('source_manifest.json')['files'];verify(files)
    jobs=read('JOBS.json');rows=read('PACKET_SUMMARY.json');decision=read('DECISION.json');trace=read('TRACE.json');audit=read('ARITHMETIC_AUDIT.json');result=read('RESULT.json')
    assert len(jobs)==219 and len(rows)==24 and len(trace)==48
    assert audit['status']=='PASS' and audit['jobs']==219 and audit['max_error']==0
    assert len(audit['rows'])==219 and all(r['fixed16_EXACT'] and r['position_bound'] for r in audit['rows'])
    assert result['status']=='PASS_D0_FINITE_UPDATE_DIAGNOSIS' and result['NoUpdate_EXACT_all3seeds']
    outputs={r['file']:r['SHA'] for r in jobs.values()};verify(outputs)
    # All registered group/step/split saved metrics consumed; no different paths or group denominators are paired.
    summary=[];gates=[];maximum=0.
    for s in SEEDS:
        per=[r for r in rows if r['seed']==s];assert sorted(r['packet'] for r in per)==list(range(8))
        for arm in ARMS:
            for step in (1,4):
                for split in ('source','probe'):
                    for group in GROUPS:
                        deltas={k:[] for k in ('K4_L','ADE','FDE','J','coverage','effective_distinct','effective_good')};lost=[];rescued=[];n=[];motions=[];baselines=[];temporal=[]
                        for p in range(8):
                            name=f's{s}_p{p}_{arm}_step{step}_{split}'
                            base=f's{s}_step0_probe' if split=='probe' else f's{s}_p{p}_step0_source'
                            m=jobs[name]['metrics'][group];b=jobs[base]['metrics'][group]
                            assert m['n']==b['n'] and m['n']>0
                            n.append(m['n']);baselines.append(b)
                            for k in deltas:deltas[k].append(m[k]-b[k])
                            lost.append(m['lost_step0good']);rescued.append(m['rescued_from_step0'])
                            close(m['good']-b['good'],m['rescued_from_step0']-m['lost_step0good'])
                            for key in m['motion']:
                                motions.append(dict(packet=p,metric=key,harm_percent=(m['motion'][key]/b['motion'][key]-1)*100))
                            temporal.append([m['single_Jclosest_early_middle_late'][i]-b['single_Jclosest_early_middle_late'][i] for i in range(3)])
                            if step==4 and group=='overall':
                                rr=per[p][arm]
                                for value,stored in ((m['K4_L']-b['K4_L'],rr['source_delta_L']) if split=='source' else (m['J']-b['J'],rr['probe_delta_J']),):
                                    maximum=max(maximum,abs(value-stored));close(value,stored)
                                if split=='probe':
                                    close(m['lost_step0good'],rr['probe_lost_step0good']);close(m['rescued_from_step0'],rr['probe_rescued_from_step0']);close(m['coverage']-b['coverage'],rr['probe_delta_coverage'])
                        baseline={k:mean([b[k] for b in baselines]) for k in ('K4_L','ADE','FDE','J','coverage','good','step0good','effective_distinct','effective_good')}
                        summary.append(dict(seed=s,arm=arm,step=step,split=split,group=group,window_n_min=min(n),window_n_max=max(n),aggregation='equal8packet_means; source group sizes can vary; probe same IDs reused',baseline=baseline,delta={k:mean(v) for k,v in deltas.items()},mean_lost=mean(lost),mean_rescued=mean(rescued),mean_net_good=mean(rescued)-mean(lost),delta_coverage_pp=100*mean(deltas['coverage']),source_K4_loss_gain_percent=-100*mean(deltas['K4_L'])/baseline['K4_L'],motion_relative_to_step0_NOT_native=motions,mean_single_Jclosest_early_middle_late_delta=[mean([v[i] for v in temporal]) for i in range(3)]))
        a={k:mean([r['ordinary'][k] for r in per]) for k in per[0]['ordinary']};b={k:mean([r['projected'][k] for r in per]) for k in per[0]['projected']}
        count=sum(r['ordinary']['source_delta_L']<0 and r['ordinary']['probe_delta_J']>0 for r in per)
        ga=a['source_delta_L']<0 and a['probe_delta_J']>0 and count>=6
        gb=b['source_delta_L']<0 and -b['source_delta_L']>=.8*(-a['source_delta_L']) and b['probe_delta_J']<=.5*a['probe_delta_J'] and b['probe_lost_step0good']<=a['probe_lost_step0good']
        original=next(x for x in decision['perseed'] if x['seed']==s)
        assert ga==original['ordinary_interference_gate'] and gb==original['projection_gate']
        gates.append(dict(seed=s,ordinary=a,projected=b,ordinary_direction_packets=count,ordinary_gate=ga,projection_raw_gate=gb,source_improvement_retained_percent=100*b['source_delta_L']/a['source_delta_L'],probe_harm_reduced_percent=None if a['probe_delta_J']<=0 else 100*(1-b['probe_delta_J']/a['probe_delta_J']),all_gate=ga and gb))
    passed=all(r['all_gate'] for r in gates);assert passed==decision['passed'] and not passed
    traces=[]
    for s in SEEDS:
        for arm in ARMS:
            rr=[r for r in trace if r['seed']==s and r['arm']==arm];assert sorted(r['packet'] for r in rr)==list(range(8))
            ss=[x for r in rr for x in r['steps']];assert len(ss)==32
            for r in rr:assert [x['step'] for x in r['steps']]==[1,2,3,4]
            for x in ss:
                dot=x['dot_matrix'];assert x['negative_pair_count']==sum(dot[i][j]<0 for i in range(3) for j in range(i+1,3))
                for g in range(3):close(x['group_norms'][g]**2,dot[g][g])
                assert x['parameter_displacement_norm']>0 and math.isfinite(x['combined_gradient_norm'])
                if arm=='ordinary':assert x['projections']==0
            traces.append(dict(seed=s,arm=arm,steps=32,steps_with_negative_pair=sum(x['negative_pair_count']>0 for x in ss),negative_pairs=sum(x['negative_pair_count'] for x in ss),pair_opportunities=96,projections=sum(x['projections'] for x in ss),steps_with_projection=sum(x['projections']>0 for x in ss),mean_gradient_norm=mean([x['combined_gradient_norm'] for x in ss]),mean_parameter_displacement_norm=mean([x['parameter_displacement_norm'] for x in ss]),winner_changes_observed=sum(x['winner_changes_from_previous_step'] or 0 for x in ss),lambda_face_changes_observed=sum(x['lambda_face_changes_from_previous_step'] or 0 for x in ss),lambda_active_fraction=mean([x['lambda_active_count']/1024 for x in ss])))
    verify(outputs);verify(files)
    out=dict(status='PASS_SAVED_SUMMARY_REVIEW',source_files=len(files),output_files=len(outputs),all_group_rows=len(summary),all_saved_group_entries=sum(len(r['metrics']) for r in jobs.values()),trace_steps=192,packet_summary_max_error=maximum,gates=gates,groups=summary,trace=traces,scope='saved scalar/group/trace arithmetic+all1578source219output SHA initial/final; not all models/gradients or clustering/motion replay again; production independent audit already covered all219 geometry jobs',newCI=False,neural_forward=False,optimizer=False,elapsed_seconds=time.perf_counter()-t)
    save('completion_review/REVIEW.json',out)
    print(json.dumps(dict(status=out['status'],groups=len(summary),traces=traces,gates=gates,maxerror=maximum,elapsed_seconds=out['elapsed_seconds']),ensure_ascii=False))
if __name__=='__main__':main()
