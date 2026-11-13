import copy
import json
import random
import numpy as np
import pytest
from f16.preprocessing import visible_local,fit_statistics
from f16.data import inference_input,split_checks,SIZES
from f16.eval import errors,evaluate,clusters,motion
from f16.statistics import paired_bootstrap,holm_adjust,verdict,mde_normal_approx
from f16.governance import require_stage,consume_once
from f16.resource import decide
from f16.scene_prior import Fragment,HistogramPrior
from f16.io import write_json
from audit.verify import audit_file,compare


@pytest.fixture
def bundle():
    rng=np.random.default_rng(4);native=rng.normal(size=(9,20,12,2)).cumsum(2)*.05
    return dict(paths=native+.002,native=native,gt=rng.normal(size=(9,12,2))*.04,groups=np.arange(9)%3,sample_id=np.arange(9),track=np.array(['a','a','b','b','c','c','d','e','e']))


def test_T1_hidden_perturbation():
    a=np.random.default_rng(5).normal(size=(7,20,2));b=a.copy();b[:,6:]=np.nan
    assert np.array_equal(visible_local(a,6).positions,visible_local(b,6).positions)


def test_T2_visible_prefix_only():
    a=np.random.default_rng(5).normal(size=(7,20,2));a[:,8:]=1e200
    assert np.array_equal(visible_local(a,8).positions,visible_local(a[:,:8],8).positions)
    assert np.array_equal(visible_local(np.zeros((2,8,2)),8).positions,np.zeros((2,8,2)))


def test_T3_GT_isolated_API():
    obs=np.ones((3,8,2));a=inference_input(obs);gt=np.full((3,12,2),np.nan)
    b=inference_input(obs);assert np.array_equal(a.observed,b.observed)
    with pytest.raises(TypeError):inference_input(obs,gt=gt)
    with pytest.raises(TypeError):inference_input(obs,change_group=np.ones(3))


def test_T4_FIT_statistics():
    x=np.arange(20).reshape(10,2);s=fit_statistics(x,np.arange(10),np.arange(12))
    assert np.array_equal(s['mean'],x.mean(0))
    with pytest.raises(ValueError):fit_statistics(x,np.arange(10),np.arange(9))
    with pytest.raises(ValueError):fit_statistics(x,np.zeros(10),np.arange(10))
    with pytest.raises(ValueError):fit_statistics(x[:0],[],[])


def test_T5_explicit_ID_and_track_contract():
    n=sum(SIZES.values());ids=np.arange(n);tracks=ids.astype(str);splits={};at=0
    for name,size in SIZES.items():splits[name]=np.arange(at,at+size);at+=size
    tracks[splits['outer'][0]]=tracks[0]
    result=split_checks(ids,tracks,splits);assert result['FIT__outer']['shared_tracks']==1
    splits['outer'][0]=0
    with pytest.raises(ValueError):split_checks(ids,tracks,splits)


def test_T6_native_zero_drift(bundle):
    z=bundle;m,_=evaluate(z['native'],z['gt'],z['groups'],z['native'],z['sample_id'],z['track'])
    assert m['overall']['lost']==m['overall']['rescue']==0
    assert m['overall']['correction_max']==0
    assert m['overall']['motion']==m['overall']['native_motion']


def test_T7_deterministic(bundle):
    x=np.arange(27).reshape(3,9)/30
    a,draws=paired_bootstrap(x,x*.8,bundle['track'],draws=100)
    b,other=paired_bootstrap(x,x*.8,bundle['track'],draws=100)
    assert a==b and np.array_equal(draws,other)


def test_complete_candidate_not_split_oracle():
    p=np.ones((1,20,12,2))*5;y=np.zeros((1,12,2))
    p[0,0]=0;p[0,0,-1]=[.5,0]
    p[0,1]=[.3,0];p[0,1,-1]=[.1,0]
    e=errors(p,y);assert not e['coverage'][0]
    assert e['ADE'][0].min()<.2 and e['FDE'][0].min()<.4


def test_Q_winner_not_min_error():
    p=np.ones((1,20,12,2))*5;y=np.zeros((1,12,2));p[0,0]=[.01,0];p[0,0,-1]=[.6,0];p[0,1]=[.16,0]
    e=errors(p,y);assert e['Q_winner'][0]==1
    assert e['Q_ADE'][0]>e['ADE'][0].min()


def test_twenty_copies_one_cluster():
    assert clusters(np.zeros((2,20,12,2)),np.ones((2,20),bool)).tolist()==[[1,1],[1,1]]


def test_motion_origin_and_counts():
    p=np.zeros((1,20,12,2));p[:,:,:,0]=np.arange(1,13)*.4
    m=motion(p);assert abs(m['speed_p95']-1)<1e-14 and m['accel_mean']<1e-14


def test_independent_audit_and_bad_output(tmp_path,bundle):
    p=tmp_path/'p.npz';r=tmp_path/'r.json';np.savez(p,**bundle)
    m,_=evaluate(bundle['paths'],bundle['gt'],bundle['groups'],bundle['native'],bundle['sample_id'],bundle['track']);write_json(r,m)
    report=audit_file(p,r);assert report['status']=='PASS' and report['n_fields']>=140
    m['change']['good']+=1;write_json(r,m)
    assert audit_file(p,r)['status']=='FAIL'
    del m['change']['Q_winner_ADE'];write_json(r,m)
    assert audit_file(p,r)['max_abs_diff'] is None


def test_auditor_not_importing_eval():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'audit/verify.py').read_text()
    assert 'from f16.eval' not in source and 'import f16.eval' not in source


def test_bootstrap_order_independent_loop():
    t=np.array([[0,1,1,0],[1,1,0,1],[1,0,1,1.]])
    b=np.zeros_like(t);tracks=np.array(['a','b','b','c']);s,boot=paired_bootstrap(t,b,tracks,draws=100,seed=2)
    rng=np.random.Generator(np.random.PCG64(2));out=[];unique=np.unique(tracks)
    for _ in range(100):
        selected=rng.integers(0,3,3);seeds=rng.integers(0,3,3)
        ix=np.concatenate([np.flatnonzero(tracks==unique[j]) for j in selected]);out.append(t[seeds[:,None],ix].mean(1).mean())
    assert np.allclose(boot,out,atol=1e-15,rtol=0)
    assert s['n_tracks']==3 and s['n_windows']==4


@pytest.mark.parametrize('ci,guards,status',[
    ((.01,.03),{'G1':True},'PASS'),((-.01,.03),{'G1':True},'NON-INFERIOR'),
    ((-.04,.03),{'G1':True},'INCONCLUSIVE'),((-.04,-.01),{'G1':True},'FAIL'),
    ((.01,.03),{'G1':False},'FAIL'),((.01,.03),{'G1':None},'INCONCLUSIVE')])
def test_statuses(ci,guards,status):assert verdict(ci,{**dict.fromkeys(['G1','G2','G3','G4','G5'],True),**guards})==status


def test_holm_not_renamed_CI():
    assert np.allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])
    assert verdict((.01,.04),dict.fromkeys(['G1','G2','G3','G4','G5'],True),multiplicity_required=True)=='INCONCLUSIVE'
    assert verdict((.01,.04),{'G1':True})=='INCONCLUSIVE'
    assert mde_normal_approx(.01)==pytest.approx(.02801585,abs=1e-8)


def test_stage_and_once_guards(tmp_path):
    with pytest.raises(PermissionError):require_stage('S3',{'state':'DRAFT'})
    p=dict(state='FROZEN',author_signed_freeze=True,freeze_blockers=[])
    with pytest.raises(PermissionError):require_stage('S4',p)
    args=dict(experiment='x',configuration_sha='a',split='outer',direction='A',stage='S3',protocol=p)
    consume_once(tmp_path/'uses.json',**args)
    with pytest.raises(PermissionError):consume_once(tmp_path/'uses.json',**args)
    args['split']='original_CAL'
    with pytest.raises(PermissionError):consume_once(tmp_path/'other.json',**args)


def test_memory_estimate_and_retained_peak():
    s=dict(physical=1300,commit=1300,rss=100,private=100,peak_rss=100,peak_private=100)
    assert decide(s,1000,'startup')['status']=='PASS'
    s['physical']=1200;assert decide(s,1000,'startup')['status']=='STOP_RESOURCE'
    assert decide(s,1000)['status']=='PASS'
    s['peak_private']=1301;assert decide(s,1000)['status']=='STOP_RESOURCE'


def test_prior_FIT_ownership_target_exclusion_and_causality():
    records=[Fragment(i,'target' if i==0 else str(i),'seq','scene',i,(0,0),0,(1.,1.),(.5,.5)) for i in range(10)]
    with pytest.raises(ValueError):HistogramPrior(records,fit_ids=range(9),kind='static',bins=16,extent=6,alpha=1)
    p=HistogramPrior(records,fit_ids=range(10),kind='static',bins=16,extent=6,alpha=1)
    a=p.query(track='target',sequence='seq',scene='scene',t0=3,cell=(0,0),heading=0);assert 0 not in a['source_ids']
    p=HistogramPrior(records,fit_ids=[],kind='causal',bins=16,extent=6,alpha=1)
    a=p.query(track='target',sequence='seq',scene='scene',t0=3,cell=(0,0),heading=0)
    assert max(a['source_end_indices'])<=3 and a['density'].sum()==pytest.approx(1)


def test_T8_hybrid_identity_and_true_forward_GT_isolation():
    torch=pytest.importorskip('torch');from f16.hybrid import SoftHybrid,set_loss
    class Head(torch.nn.Module):
        def forward(self,q,c):return q[...,:2,None].transpose(-1,-2).expand(-1,-1,12,-1)
    q=torch.randn(2,20,4);context=torch.randn(2,3);m=SoftHybrid(Head(),Head(),4,3)
    expected=Head()(q,context);a=m(q,context)
    assert torch.equal(a['paths'],expected) and torch.equal(a['alpha'],torch.full((2,20,1),.5))
    with pytest.raises(TypeError):m(q,context,gt=torch.full((2,12,2),float('nan')))
    loss,_=set_loss(a['paths'],torch.randn(2,12,2),a['alpha'],temperature=.2,coverage_weight=.1,entropy_weight=.01,margin=.1,softmin_definition='logmeanexp');loss.backward()
    assert torch.isfinite(m.gate.weight.grad).all()


def test_checkpoint_CPU_exact_optimizer_and_all_CPU_RNG(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');from f16.checkpoint import save_checkpoint,load_checkpoint
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    random.seed(7);np.random.seed(7);torch.manual_seed(7);gen=torch.Generator().manual_seed(7)
    m=torch.nn.Linear(2,1);opt=torch.optim.AdamW(m.parameters(),lr=.01)
    def step():
        x=torch.randn(4,2,generator=gen);y=torch.rand(4,1)+random.random()+np.random.random();opt.zero_grad();loss=(m(x)-y).square().mean();loss.backward();opt.step();return float(loss.detach())
    step();contract={'source':'s','IDs':'i','protocol':'p','stats':'t'};p=tmp_path/'state.pt'
    save_checkpoint(p,m,opt,gen,[1],contract);expected=step();state=copy.deepcopy(m.state_dict());optim=copy.deepcopy(opt.state_dict());rng=(random.random(),np.random.random(),torch.rand(1),torch.rand(1,generator=gen))
    with pytest.raises(ValueError):load_checkpoint(p,m,opt,gen,{**contract,'source':'bad'})
    assert load_checkpoint(p,m,opt,gen,contract)==[1];assert step()==expected
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
    assert all(torch.equal(opt.state_dict()['state'][k][n],v) for k,entry in optim['state'].items() for n,v in entry.items())
    assert random.random()==rng[0] and np.random.random()==rng[1] and torch.equal(torch.rand(1),rng[2]) and torch.equal(torch.rand(1,generator=gen),rng[3])
