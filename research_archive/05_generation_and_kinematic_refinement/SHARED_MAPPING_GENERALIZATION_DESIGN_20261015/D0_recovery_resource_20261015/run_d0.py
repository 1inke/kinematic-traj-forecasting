import core as c
import numpy as np,torch,time,gc

def gate(rows):
    report=[]
    for s in c.SEEDS:
        allr=[r for r in rows if r['seed']==s];assert len(allr)==8
        dA=np.array([r['ordinary']['source_delta_L'] for r in allr]);dB=np.array([r['projected']['source_delta_L'] for r in allr]);jA=np.array([r['ordinary']['probe_delta_J'] for r in allr]);jB=np.array([r['projected']['probe_delta_J'] for r in allr]);lA=np.array([r['ordinary']['probe_lost_step0good'] for r in allr]);lB=np.array([r['projected']['probe_lost_step0good'] for r in allr])
        ordinary=bool(dA.mean()<0 and jA.mean()>0 and np.sum((dA<0)&(jA>0))>=6)
        projection=bool(dB.mean()<0 and -dB.mean()>=.8*(-dA.mean()) and jB.mean()<=.5*jA.mean() and lB.mean()<=lA.mean())
        report.append(dict(seed=s,ordinary_source_delta_L=float(dA.mean()),projected_source_delta_L=float(dB.mean()),ordinary_probe_delta_J=float(jA.mean()),projected_probe_delta_J=float(jB.mean()),ordinary_joint_direction_packets=int(((dA<0)&(jA>0)).sum()),ordinary_lost_mean=float(lA.mean()),projected_lost_mean=float(lB.mean()),ordinary_interference_gate=ordinary,projection_gate=projection))
    passed=all(r['ordinary_interference_gate'] and r['projection_gate'] for r in report)
    return dict(status='D0_TRIGGER_U1_PREFLIGHT' if passed else 'D0_STOP_NO_U1',passed=passed,perseed=report,U1_production_authorized_by_this_result=False,meaning='If passed, U1 still needs its own actual preflight and<=30min budget; D0 is local previouslyseen FIT, not generalization')

def summarize_trace(trace):
    previous=None;out=[]
    for r in trace:
        r=dict(r);win=np.asarray(r.pop('winner'));lam=np.asarray(r.pop('lambda_active'));r['winner_changes_from_previous_step']=None if previous is None else int(np.sum(win!=previous[0]));r['lambda_face_changes_from_previous_step']=None if previous is None else int(np.sum(lam!=previous[1]));r['winner_counts']=np.bincount(win,minlength=4).tolist();r['lambda_active_count']=int(lam.sum());previous=(win,lam);out.append(r)
    return out

def main():
    t=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
    files=c.read(c.H/'source_manifest.json')['files'];assert c.digest(files)==pre['source_digest'];c.stage('D0_INITIAL_SOURCE_SHA');c.verify(files)
    data,sp=c.load_data();packs,probe,sel=c.selection(data,sp);saved=c.npz(c.H/'selection.npz');assert np.array_equal(packs,saved['source_packets']) and np.array_equal(probe,saved['probe_index'])
    alljobs={};rows=[];trace=[]
    def save(name,z,baseline,meta):
        path=c.H/'saved'/f'{name}.npz';path.parent.mkdir(exist_ok=True);assert not path.exists(),('PRESERVE_EXISTING_PARTIAL_OUTPUT',name)
        np.savez_compressed(path,**z);back=c.npz(path);assert all(np.array_equal(v,back[k]) for k,v in z.items())
        met=c.metrics(back,baseline);record=dict(name=name,file=str(path),SHA=c.sha(path),metadata=meta,metrics=met)
        alljobs[name]=record;c.write('saved/'+name+'_done.json',record);return met
    for s in c.SEEDS:
        c.stage('D0_BASELINE_NOUPDATE',seed=s);m=c.build(s);probe0=c.predict(m,data,probe);probe_copy=c.predict(m,data,probe);assert all(np.array_equal(v,probe_copy[k]) for k,v in probe0.items());del probe_copy
        source_all=c.predict(m,data,packs.ravel());source_copy=c.predict(m,data,packs.ravel());assert all(np.array_equal(v,source_copy[k]) for k,v in source_all.items());del source_copy,m
        p0=save(f's{s}_step0_probe',probe0,probe0['paths'],dict(seed=s,packet=None,arm='NoUpdate',step=0,split='probe'))
        for pack,ix in enumerate(packs):
            source0={k:v[pack*256:(pack+1)*256] for k,v in source_all.items()};src0=save(f's{s}_p{pack}_step0_source',source0,source0['paths'],dict(seed=s,packet=pack,arm='NoUpdate',step=0,split='source'))
            packet=dict(seed=s,packet=pack)
            for arm in c.ARMS:
                c.stage('D0_FINITE_UPDATES',seed=s,packet=pack,arm=arm,completed_packets=len(rows));m=c.build(s);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);history=[];last={}
                for step in range(4):
                    history.append(c.apply_step(m,o,data,ix,arm,step))
                    if step+1 in (1,4):
                        for label,indices,base in [('source',ix,source0),('probe',probe,probe0)]:
                            z=c.predict(m,data,indices);met=save(f's{s}_p{pack}_{arm}_step{step+1}_{label}',z,base['paths'],dict(seed=s,packet=pack,arm=arm,step=step+1,split=label));last[label]=met;del z
                packet[arm]=dict(source_delta_L=last['source']['overall']['K4_L']-src0['overall']['K4_L'],probe_delta_J=last['probe']['overall']['J']-p0['overall']['J'],probe_lost_step0good=last['probe']['overall']['lost_step0good'],probe_rescued_from_step0=last['probe']['overall']['rescued_from_step0'],probe_delta_coverage=last['probe']['overall']['coverage']-p0['overall']['coverage'])
                trace.append(dict(seed=s,packet=pack,arm=arm,steps=summarize_trace(history)));del m,o;gc.collect();torch.cuda.empty_cache()
            rows.append(packet);c.write('partial_packet_summary.json',rows)
        del source_all,probe0
    decision=gate(rows);c.write('TRACE.json',trace);c.write('PACKET_SUMMARY.json',rows);c.write('DECISION.json',decision)
    c.write('JOBS.json',alljobs);c.stage('D0_INDEPENDENT_SAVED_AUDIT',jobs=len(alljobs))
    from audit_saved import main as audit_main
    audit_main()
    c.stage('D0_FINAL_SOURCE_SHA');c.verify(files)
    result=dict(status='PASS_D0_FINITE_UPDATE_DIAGNOSIS',decision=decision,source_digest=c.digest(files),jobs=len(alljobs),source_packets=24,NoUpdate_EXACT_all3seeds=True,source_probe_track_disjoint=True,new_formal_count=0,old_optimizer_not_resumed=True,new_optimizer_used_for_temporary_intervention=True,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE)
    c.write('RESULT.json',result);c.write('RESOURCE_USAGE.json',dict(status='PASS',**c.RESOURCE));c.stage('SUCCESS',decision=decision['status']);print('D0_RESULT',decision,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.reject_failure('STOP_D0');raise
