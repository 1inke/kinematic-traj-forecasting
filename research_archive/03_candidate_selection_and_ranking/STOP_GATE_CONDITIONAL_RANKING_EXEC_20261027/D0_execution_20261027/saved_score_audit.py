import sys,time,csv,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from shared import *
from ranking import diagnose,summarize,random_hit
HERE=Path(__file__).resolve().parent
MODELS=('legacy_I','legacy_S','probe_M_full','probe_T_full','probe_T_geom','token_C_base','token_C_token')
def population(model,seed,split,g):
    family,arm=model.split('_',1);oldfolder=OLD/f'populations/seed{seed}/{split}'
    folder=oldfolder if family=='legacy' else (P12 if family=='probe' else P3)/f'populations/{arm}_seed{seed}/{split}'
    done=read_json(folder/'done.json');outputs=done['outputs'];chunks=done['chunks'];parts=[];sourcefiles={};auditcount=0
    for j,(name,z) in enumerate(aux(seed,split)):
        fn=arm+'_'+name[len('aux_'):] if family=='legacy' else chunks[j]
        path=folder/fn
        if not path.exists():raise FileNotFoundError('BLOCKED_SAVED_SCORES '+str(path))
        assert sha256(path)==outputs[fn]
        with np.load(path) as q:
            assert np.array_equal(q['sample_id'],z['sample_id'])
            if 'eligible' in q.files:
                for k in ('eligible','B','target'):assert np.array_equal(q[k],z[k]),k
            actual=q['action'] if family=='legacy' else q['chosen'];scores=q['scores']
            if scores.shape!=(len(z['sample_id']),401):raise ValueError('BLOCKED_SAVED_SCORES full401')
            dd=diagnose(scores,z['eligible'],z['B'],z['quality'],actual)
            for i,a in enumerate(actual):
                expected=z['native'][i].copy()
                if a:r,l=divmod(int(a)-1,20);expected[r]=z['kin'][i,l]
                assert np.array_equal(expected,q['paths'][i])
            dd.update(sample_id=z['sample_id'],track_key=z['track_key'],group=z['group']);parts.append(dd)
        sourcefiles[str(path.relative_to(ROOT))]=outputs[fn];auditcount+=len(z['sample_id']);g.check()
    d={k:np.concatenate([x[k] for x in parts]) for k in parts[0]};assert len(set(d['sample_id']))==len(d['sample_id'])
    out=HERE/f'populations/{model}_seed{seed}/{split}';out.mkdir(parents=True)
    save_npz(out/'PER_WINDOW_RANK_MARGIN.npz',**d)
    groups={}
    for i,k in enumerate(GROUPS):groups[k]=summarize(d,np.ones(len(d['group']),bool) if i==0 else d['group']==i-1)
    strata={}
    for good in (False,True):strata['native_success' if good else 'native_failure']=summarize(d,d['native_good']==good)
    write_json(out/'ALL_GROUPS.json',groups);write_json(out/'NATIVE_SUCCESS_FAILURE.json',strata)
    finish(out,dict(source=source(HERE),model=model,seed=seed,split=split,old_done_SHA=sha256(folder/'done.json')),n=len(d['sample_id']),all_original_paths_exact=True,independent_stable_action_sort_all_windows=True)
    write_json(HERE/'progress.json',dict(stage='D0_SAVED_SCORE_AUDIT',model=model,seed=seed,split=split,at=now()))
    return dict(model=model,seed=seed,split=split,n=auditcount,status='PASS_FULL401_SAVED',source_folder=str(folder.relative_to(ROOT)),old_done_SHA=sha256(folder/'done.json'),score_SHA=sourcefiles),groups
def weights():
    records=[]
    for seed in SEEDS:
        with np.load(OLD/f'prepared_v1/labels_seed{seed}.npz') as z:
            e=z['eligible'];t=z['target'];B=z['B'];ids=z['sample_id'];assert np.array_equal(t[:,0],~B.any(1))
        p=e&t;n=e&~t;has=n.any(1);wp=np.divide(.5,p.sum(1),out=np.zeros(len(e)),where=has);wn=np.divide(.5,n.sum(1),out=np.zeros(len(e)),where=has)
        sp=float(wp[t[:,0]].sum());sn=float(wn[~t[:,0]].sum());ep=float((.5*has-wp*t[:,0]).sum());en=float((.5*has-wn*~t[:,0]).sum())
        table=load_npz(P12/f'prepared/actions_seed{seed}.npz');assert np.array_equal(table['sample_id'],ids);m=table['actions']==0
        asp=float(table['weights'][m&table['labels']].sum());asn=float(table['weights'][m&~table['labels']].sum());assert abs(sp-asp)<1e-10 and abs(sn-asn)<1e-10
        records.append(dict(seed=seed,STOP_positive_windows=int(t[:,0].sum()),STOP_negative_windows=int((~t[:,0]).sum()),no_negative_windows=int((~has).sum()),STOP_positive_weight=sp,STOP_negative_weight=sn,nonSTOP_positive_weight=ep,nonSTOP_negative_weight=en,STOP_ratio=sp/sn,STOP_identity_optimal_sigmoid=sp/(sp+sn),nonSTOP_identity_optimal_sigmoid=ep/(ep+en),sampled_IPW_STOP_weights_exact=True,
            zero_score_STOP_gradient_emptyB=-.25,zero_score_STOP_gradient_nonemptyB_formula='+.25/Nnegative',not_actual_model_probability=True))
    return records
def preflight():
    # Finite tie/no-edit/emptyB/rescue and independent combinatorial product.
    e=np.zeros((5,401),bool);e[:,0]=True;e[1:,1:5]=True;b=np.zeros_like(e);b[1:4,2]=True
    s=np.zeros(e.shape);s[1,2]=1;s[2,2]=-1;s[2,1]=-2;s[2,3:5]=-2;s[3,1]=1;s[4,1]=1
    q=np.ones(e.shape+(3,));q[:,:,2]=2;q[1:4,2,2]=.9
    a=np.where(e,s,-np.inf).argmax(1);d=diagnose(s,e,b,q,a)
    assert d['event'].tolist()==['NO_ELIGIBLE_EDIT','PASS_GOOD','BLOCKED_GOOD','BAD_INTERNAL','BAD_NO_B_RELEASE']
    for m in (1,4,20,400):
        for bb in (0,1,m):
            for k in (1,5,10):
                other=1.-np.prod([(m-bb-j)/(m-j) for j in range(min(k,m))]) if m-bb>=min(k,m) else 1.
                assert abs(random_hit(m,bb,k)-other)<=2e-12
    # Analytic derivative of constant-score BCE versus centered finite differences.
    for n in (1,5,400):
        fun=lambda x:.5*np.logaddexp(0,x)/n
        assert abs((fun(1e-5)-fun(-1e-5))/2e-5-.25/n)<1e-10
    return dict(status='PASS',fixtures='noedit/tie/blockedgood/rankbad/rescue/random_exact/zero_gradient',no_Torch=True)
def main():
    assert not (HERE/'RESULT.json').exists();g=Guard(read_json(HERE/'RESOURCE_STAGE_PLAN.json')['estimate_bytes']);g.check('startup');start=time.perf_counter()
    m=build_source(HERE);pf=preflight();write_json(HERE/'PREFLIGHT.json',dict(**pf,source=m['digest']));write_json(HERE/'PRODUCTION_SOURCE_LOCK.json',dict(source=m['digest'],at=now(),source_count=len(m['files'])))
    matrices=[];allgroups=[]
    for model in MODELS:
        for seed in SEEDS:
            for split in SPLITS:
                record,gg=population(model,seed,split,g);matrices.append(record)
                for group,values in gg.items():allgroups.append(dict(model=model,seed=seed,split=split,group=group,**values))
    assert len(matrices)==63;write_json(HERE/'SOURCE_REUSE_MATRIX.json',matrices);write_json(HERE/'ALL_GROUPS.json',allgroups);write_json(HERE/'STOP_WEIGHT_AUDIT.json',weights())
    with (HERE/'ALL_GROUPS.csv').open('x',encoding='utf-8-sig',newline='') as f:
        keys=('model','seed','split','group','n','B_nonempty','rescue_opportunities','native_good','actual_good','actual_rescue','actual_lost','actual_executed','best_B','best_rescue','blocked_good','blocked_rescue','internal_bad');w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(allgroups)
    from statistics_saved import run
    run(g);assert verify_manifest(HERE)==m['digest']
    write_json(HERE/'RESOURCE.json',dict(last=g.check(),observations=g.observations));write_json(HERE/'RESULT.json',dict(status='PASS_D0_SAVED_SCORE_REVIEW_REQUIRED',source=m['digest'],populations=63,rows=sum(x['n'] for x in matrices),seconds=time.perf_counter()-start,no_Torch_model_forward_training=True))
    outputs={p.relative_to(HERE).as_posix():sha256(p) for p in HERE.rglob('*') if p.is_file() and (p.is_relative_to(HERE/'populations') or p.name in ('ALL_GROUPS.json','ALL_GROUPS.csv','STOP_WEIGHT_AUDIT.json','SOURCE_REUSE_MATRIX.json','BOOTSTRAP_DRAWS.npz','BOOTSTRAP.json','RESOURCE.json','RESULT.json'))}
    write_json(HERE/'PRESERVATION.json',dict(status='PASS',source=m['digest'],outputs=outputs));assert 'torch' not in sys.modules;print('PASS D0 ALL63',time.perf_counter()-start,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:write_json(HERE/('FAILURE_'+str(time.time_ns())+'.json'),dict(error=str(ex),traceback=traceback.format_exc(),preserve_all=True));raise
