"""Reconstruct all saved-template candidates with independent equations, without re-fitting."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,sys,time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'design_original/code'))
import kine_common as kc

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def same(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for j,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(j))
    elif isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)):
        np.testing.assert_allclose(a,b,atol=3e-11,rtol=3e-11,err_msg=path)
    else:assert a==b,(path,a,b)
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def frame(d):
    p=d['past'][:,0].astype(float);y=d['future'].astype(float);disp=p[:,-1]-p[:,-4];norm=np.hypot(disp[:,0],disp[:,1]);valid=norm>.03
    h=np.zeros_like(disp);h[:,0]=1;h[valid]=disp[valid]/norm[valid,None]
    rel=y-p[:,-1,None];fl=np.stack([rel[:,:,0]*h[:,0,None]+rel[:,:,1]*h[:,1,None],-rel[:,:,0]*h[:,1,None]+rel[:,:,1]*h[:,0,None]],axis=-1)
    return fl,norm/1.2,valid
def reconstruct(family,template,ve,lam):
    n=len(ve)
    if family=='P':return np.broadcast_to(template.reshape(1,20,3,2),(n,20,3,2))
    if family=='S':return template.reshape(1,20,3,2)*(.4*ve[:,None,None,None])
    rho=np.clip(np.exp(template[:,:3])-.05,0,None);phi=template[:,3:]/lam
    velocity=np.stack([rho*np.cos(phi),rho*np.sin(phi)],axis=-1)
    return np.cumsum(velocity[None]*(1.6*ve[:,None,None,None]),axis=2)
def errors(kp,future):
    n=len(future);a=np.empty(n);f=np.empty(n)
    for start in range(0,n,1024):
        points=kp[start:start+1024];cur=future[start:start+1024];segments=[]
        for j in range(3):
            left=np.zeros_like(points[:,:,0]) if j==0 else points[:,:,j-1]
            right=points[:,:,j]
            segments.extend([left+(right-left)*(step/4) for step in range(1,5)])
        path=np.stack(segments,axis=2);delta=path-cur[:,None];dist=np.hypot(delta[:,:,:,0],delta[:,:,:,1])
        a[start:start+len(cur)]=np.min(np.sum(dist,axis=2)/12,axis=1);f[start:start+len(cur)]=dist[:,:,-1].min(axis=1)
    return a,f
def groups(d,profile):
    labels=kc.maneuver_labels(d['past'],d['future']);bins=kc.speed_bins(d['past'],profile)
    result={'overall':np.ones(len(labels),bool),'decel_stop':np.isin(labels,['deceleration','stop'])}
    result.update({'maneuver:'+v:labels==v for v in kc.MANEUVERS})
    result.update({'speed:'+v:bins==j for j,v in enumerate(kc.SPEED_NAMES)})
    result.update({'change:'+v:d['group']==j for j,v in enumerate(kc.CHANGE_NAMES)})
    return result,labels,bins
def block(a,f,mask):
    n=int(sum(mask))
    if not n:return {'n':0}
    return dict(n=n,small=n<20,minADE=float(a[mask].mean()),minFDE=float(f[mask].mean()),FDE_p90=float(np.percentile(f[mask],90)),reach_FDE_le_0p4=float(np.count_nonzero(f[mask]<=.4)/n))
def main():
    started=time.monotonic();cfg=read(ROOT/'CONFIG.json');out=Path(cfg['output_dir']);done=read(ROOT/'DONE.json')
    assert done['status']=='SUCCESS'
    for p,h in read(ROOT/'SOURCE_MANIFEST.json')['files'].items():assert sha(p)==h,p
    for p,h in done['files'].items():assert sha(p)==h,p
    result=read(out/'RESULT.json');tpl=load(out/'TEMPLATES.npz');assert len(tpl)==150
    dest=out/'analysis_inputs';assert not dest.exists(),'Verification output already exists; inspect instead of overwrite';dest.mkdir()
    arms=[('P','P',.3,1.),('P_AUG','P',.3,1.),('S','S',.3,1.),('K','K',.3,1.),('S_v0=0.15','S',.15,1.),('S_v0=0.5','S',.5,1.),('K_v0=0.15','K',.15,1.),('K_v0=0.5','K',.5,1.),('K_lam=0.5','K',.3,.5),('K_lam=2','K',.3,2.)]
    populations=[];recomputed={};group_count=0
    for fold in cfg['folds']:
        train=kc.load_split(cfg['data_dir'],fold,'train');test=kc.load_split(cfg['data_dir'],fold,'test');fi,vi=kc.scene_tail_split(train['scene']);val=kc.subset(train,vi)
        profile=kc.fit_profile(train['past'][fi]);same(profile,result['folds'][fold]['profile'],fold+'/profile')
        recomputed[fold]={}
        for split,d in [('val',val),('test',test)]:
            print(f'VERIFY {fold}/{split}',flush=True)
            y,vr,valid=frame(d);g,labels,bins=groups(d,profile);saved=load(out/f'PER_WINDOW_{fold}_{split}.npz')
            np.testing.assert_array_equal(saved['ids'],d['ids']);np.testing.assert_array_equal(saved['labels'],labels);np.testing.assert_array_equal(saved['speed_bin'],bins);np.testing.assert_array_equal(saved['change'],d['group']);np.testing.assert_allclose(saved['v_rec'],vr,rtol=1e-12,atol=1e-12)
            arrays=dict(ids=d['ids'],labels=labels,speed_bin=bins,change=d['group'],v_rec=vr,heading_valid=valid,scene=d['scene']);reports={}
            for name,family,v0,lam in arms:
                aa=[];ff=[];reports[name]=[]
                for seed in cfg['seeds']:
                    templates=tpl[f'{fold}__{name}__{seed}'];assert templates.shape==(20,6) and np.isfinite(templates).all()
                    kp=reconstruct(family,templates,np.maximum(vr,v0),lam);a,f=errors(kp,y);aa.append(a);ff.append(f)
                    reports[name].append({key:block(a,f,mask) for key,mask in g.items()})
                arrays[name+'__minADE']=np.stack(aa);arrays[name+'__minFDE']=np.stack(ff)
                same(reports[name],result['folds'][fold]['splits'][split][name],fold+'/'+split+'/'+name);group_count+=len(g)*3
                if name in ('P','P_AUG','S','K'):
                    np.testing.assert_allclose(np.mean(aa,axis=0),saved[name+'_minADE'],atol=3e-11,rtol=3e-11);np.testing.assert_allclose(np.mean(ff,axis=0),saved[name+'_minFDE'],atol=3e-11,rtol=3e-11)
            name='REF_P_ORACLE_SCALE';reports[name]=[];aa=[];ff=[]
            for seed in cfg['seeds']:
                base=reconstruct('P',tpl[f'{fold}__P__{seed}'],vr,1.);a=np.full(len(y),np.inf);f=a.copy()
                for scale in np.geomspace(.25,4,17):
                    candidate_a,candidate_f=errors(base*scale,y);better=candidate_a<a;a[better]=candidate_a[better];f[better]=candidate_f[better]
                aa.append(a);ff.append(f);reports[name].append({key:block(a,f,mask) for key,mask in g.items()})
            same(reports[name],result['folds'][fold]['splits'][split][name],fold+'/'+split+'/'+name)
            arrays[name+'__minADE']=np.stack(aa);arrays[name+'__minFDE']=np.stack(ff)
            if fold=='eth' and split=='test':
                for spec in cfg['references']:
                    _,tag,path=spec.split(':',2);pred=load(path);lookup={v:i for i,v in enumerate(pred['ids'].tolist())};order=np.array([lookup[v] for v in d['ids'].tolist()]);delta=pred['paths'][order].astype(float)-d['future'][:,None].astype(float);dist=np.hypot(delta[:,:,:,0],delta[:,:,:,1]);a=dist.mean(-1).min(-1);f=dist[:,:,-1].min(-1);name='REF_'+tag
                    reports[name]=[{key:block(a,f,mask) for key,mask in g.items()}];same(reports[name],result['folds'][fold]['splits'][split][name],name);arrays[name+'__minADE']=a[None];arrays[name+'__minFDE']=f[None]
            np.savez_compressed(dest/f'{fold}_{split}_seed_metrics.npz',**arrays)
            recomputed[fold][split]=reports;populations.append(dict(fold=fold,split=split,n=len(y),arms=len(reports),all_groups_seed_metrics='PASS'))
    # Independent construction of all six prespecified checks.
    mean=lambda fold,split,arm,metric:float(np.mean([b['overall'][metric] for b in recomputed[fold][split][arm]]))
    foldmean=lambda split,arm,metric:float(np.mean([mean(fold,split,arm,metric) for fold in cfg['folds']]))
    gain=lambda a,b:(b-a)/b if b>0 else 0.
    ef=gain(mean('eth','test','K','minFDE'),mean('eth','test','P','minFDE'));ea=gain(mean('eth','test','K','minADE'),mean('eth','test','P','minADE'))
    per={f:dict(g_ADE=gain(mean(f,'test','K','minADE'),mean(f,'test','P','minADE')),g_FDE=gain(mean(f,'test','K','minFDE'),mean(f,'test','P','minFDE'))) for f in cfg['folds']}
    va=gain(foldmean('val','K','minADE'),foldmean('val','P','minADE'));vf=gain(foldmean('val','K','minFDE'),foldmean('val','P','minFDE'))
    pairs=[gain(k['overall']['minFDE'],p['overall']['minFDE']) for k,p in zip(recomputed['eth']['test']['K'],recomputed['eth']['test']['P'])]
    pooled={}
    for arm in ('P','P_AUG','S','K'):
        pooled[arm]={'decel_stop':{}}
        for metric in ('minADE','minFDE'):
            values=[]
            for i in range(3):
                blocks=[recomputed[f]['test'][arm][i]['decel_stop'] for f in cfg['folds']];total=sum(b['n'] for b in blocks);values.append(sum(b.get(metric,0)*b['n'] for b in blocks)/total)
            pooled[arm]['decel_stop'][metric]=values
    same(pooled,result['summary']['pooled_test'],'pooled')
    expected_folds={fold:{split:{arm:dict(minADE=[b['overall']['minADE'] for b in per_seed],minFDE=[b['overall']['minFDE'] for b in per_seed]) for arm,per_seed in recomputed[fold][split].items() if len(per_seed)==3} for split in ('val','test')} for fold in cfg['folds']}
    same(expected_folds,result['summary']['folds'],'summary/folds')
    dec={a:float(np.mean(pooled[a]['decel_stop']['minFDE'])) for a in pooled}
    checks=dict(C1_eth_speed_shift=ef>=.05 and ea>=.03,C2_not_worse_than_aug_fivefold=foldmean('test','K','minFDE')<=foldmean('test','P_AUG','minFDE'),C3_no_test_fold_worse=all(b['g_ADE']>=-.02 and b['g_FDE']>=-.02 for b in per.values()),C4_in_distribution_ok=va>=-.01 and vf>=-.01,C5_decel_ok=dec['K']<=dec['P'] and dec['K']<=dec['P_AUG'],C6_eth_all_seed_pairs_positive=all(v>0 for v in pairs))
    same(checks,result['decision']['checks'],'checks')
    numbers=dict(eth_test_g_FDE=ef,eth_test_g_ADE=ea,eth_seed_pair_g_FDE=pairs,per_fold_test=per,fivefold_val_g_ADE=va,fivefold_val_g_FDE=vf,fivefold_test_minFDE_main=foldmean('test','K','minFDE'),fivefold_test_minFDE_P_AUG=foldmean('test','P_AUG','minFDE'),decel_stop_minFDE=dict(main=dec['K'],P=dec['P'],P_AUG=dec['P_AUG']))
    same(numbers,result['decision']['numbers'],'decision/numbers')
    for x,y in [('K','S'),('S','P')]:
        ge=gain(mean('eth','test',x,'minFDE'),mean('eth','test',y,'minFDE'));gd=gain(dec[x],dec[y])
        label=x+'_BETTER' if ge>=.02 or gd>=.02 else y+'_BETTER' if ge<=-.02 or gd<=-.02 else 'SIMILAR'
        if (ge>=.02 and gd<=-.02) or (ge<=-.02 and gd>=.02):label='MIXED'
        same(dict(label=label,eth_test_g_FDE=ge,decel_stop_g_FDE=gd),result['decision']['attribution'][x+'_vs_'+y],'attribution')
    c=checks;verdict='SUPPORT' if c['C1_eth_speed_shift'] and c['C3_no_test_fold_worse'] and c['C4_in_distribution_ok'] and c['C6_eth_all_seed_pairs_positive'] and (c['C2_not_worse_than_aug_fivefold'] or c['C5_decel_ok']) else 'WEAK' if c['C3_no_test_fold_worse'] and c['C4_in_distribution_ok'] and ef>=.02 else 'NOT_SUPPORTED'
    assert verdict==result['decision']['verdict']
    assert 'torch' not in sys.modules
    for p,h in done['files'].items():assert sha(p)==h,p
    write(out/'DERIVED_INPUTS.json',dict(origin='Independent reconstruction from sealed TEMPLATES and source coordinates, no refitting.',seeds=cfg['seeds'],array_shape='arm__minADE/minFDE are [seed,N]; references B0/AUG have one row.',aggregation='Each metric per seed and group first; then seed average. Do not compute p90/reach from the seed-averaged errors.',raw_coordinates_included=False,source_done_sha256=sha(ROOT/'DONE.json'),verification_script_sha256=sha(__file__)))
    value=dict(status='PASS',at=datetime.now(timezone(timedelta(hours=8))).isoformat(),seconds=time.monotonic()-started,script_sha256=sha(__file__),main_verdict=verdict,checks=checks,populations=populations,template_arrays=150,all_sensitivity_and_oracle_reference_geometry='PASS',raw_input_alignment='PASS',all_original_scientific_output_hashes_preserved=True,no_network_training_or_inference=True,GPU_used=False,derived_files={str(p):sha(p) for p in dest.glob('*.npz')})
    write(ROOT/'INDEPENDENT_VERIFICATION.json',value);print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
