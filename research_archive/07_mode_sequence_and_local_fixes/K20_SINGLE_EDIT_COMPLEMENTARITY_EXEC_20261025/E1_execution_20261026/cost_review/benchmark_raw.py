"""Frozen real observation -> F16 + KIN + all400 checks/features + editor.
No legacy main, training, GT labels or output writes outside this new cost folder.
"""
from pathlib import Path
import sys,ast,copy,importlib.util,os,time,traceback,json
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from env import *
from feature_builder import build
from editor import ActionEditor
import torch
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def main():
    guard=Guard(ESTIMATE);guard.check('startup');torch.set_num_threads(1);seed_all(42)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    original=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py'
    nodes=[n for n in ast.parse(original.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name in {'fold_data','fold_run','load_module','build_baseline'}];assert len(nodes)==4
    astra=Path(r'<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20')
    for p in (astra,astra/'scripts',ROOT/'FRONT_ENCODER_SERIES'):sys.path.insert(0,str(p))
    os.chdir(astra);import sitecustomize
    from experiment_config import load_locked_q20
    cfgfile=ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml'
    cfg=load_locked_q20(cfgfile);cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.device='cpu';cfg.device_list=['cpu']
    ns=dict(Path=Path,importlib=__import__('importlib'),sys=sys,torch=torch,os=os,copy=copy,DATA_ROOT=astra/'datasets/eth_ucy/processed_data_diverse',BASE_ROOT=ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(original),'exec'),ns)
    front=module('_single_edit_cost_front',ROOT/'FRONT_ENCODER_SERIES/front_encoder_models.py');trainer=module('_single_edit_cost_trainer',ROOT/'FRONT_ENCODER_SERIES/train_front_variant.py');adapter=module('_single_edit_cost_adapter',ROOT/'E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py')
    net=ns['build_baseline']('eth',cfg,front,adapter,torch.device('cpu'))
    sys.path.insert(0,str(ROOT/'KINEMATIC_DECODER_SPEED_CLAMP_20261018'));from kdec.models import KinematicClampDecoder
    kp=ROOT/'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed42.pt';ck=torch.load(kp,map_location='cpu',weights_only=False)
    kin=KinematicClampDecoder(ck['v_max'],ck['a_max']);kin.load_state_dict(ck['state_dict'],strict=True);kin.eval().requires_grad_(False)
    ep=HERE/'runs/S_seed42/checkpoint.pt';ec=torch.load(ep,map_location='cpu',weights_only=False)
    ed=ActionEditor(np.zeros(323),np.ones(323));ed.load_state_dict(ec['model'],strict=True);ed.eval().requires_grad_(False)
    rawfile,numfile=ns['fold_data']('eth','train');raw=np.load(rawfile,mmap_mode='r');num=np.load(numfile,mmap_mode='r')
    with np.load(HERE/'prepared_v1/identity.npz') as z:ids=z['sample_id'][:32].copy()
    encfile=ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
    with np.load(encfile) as z:
        lookup={int(x):i for i,x in enumerate(z['sample_id'])};rr=np.asarray([lookup[int(x)] for x in ids]);cachepast=z['past'][rr];cachetokens=z['tokens'][rr]
    with np.load(E0/'seed42/input_pools.npz') as z:assert np.array_equal(z['sample_id'][:32],ids);cachenative=z['native'][:32];cachekin=z['kin'][:32]
    files=[original,cfgfile,kp,ep,rawfile,numfile,encfile,Path(__file__),HERE/'feature_builder.py',E0/'space.py']+[ns['fold_run']('eth',s)/'best_selected.pth' for s in ('h3','v2')]
    before_files={str(p):sha256(p) for p in files};before=[state(m) for m in (net,kin,ed)]
    param_counts={n:sum(p.numel() for p in m.parameters()) for n,m in zip(('F16','KIN','editor'),(net,kin,ed))};unique={id(p):p for m in (net,kin,ed) for p in m.parameters()};total=sum(p.numel() for p in unique.values())
    captured=[];hook=net.decoder.register_forward_pre_hook(lambda owner,args:captured.append(args[0][:,0].detach()))
    scale=float(cfg.PREDICTION.TRAJ_SCALE)
    def pipeline(which,device):
        # In-memory read of the real raw observation slice and existing num sidecar is timed; no future slice is read.
        p=torch.from_numpy(np.array(raw[which,:9,:8,-2:],np.float32,copy=True)/scale).to(device)
        valid=torch.tensor([min(int(num[which]),9)],dtype=torch.int32,device=device)
        p=p.unsqueeze(0);local,_=trainer.normalize_to_focal_heading(p,p);center=local.mean(-2,keepdim=True);captured.clear()
        pred=net(local-center,None,None,mode='testing',num_valid=valid,graph_loc=local)[2]
        native=(pred[:,0]+center[:,0,None]-local[:,0,-1,None,None])[0].cpu().numpy().astype(np.float64)
        past=local[:,0];kk=kin(past)['paths'][0].cpu().numpy().astype(np.float64);tok=captured[0][0].cpu().numpy().copy();pp=past[0].cpu().numpy().copy()
        eligible,mm,rejects=e0space.eligible_actions(native,kk)
        features=build(pp,tok,native,kk,mm).astype(np.float32)
        scores=ed(torch.from_numpy(features).unsqueeze(0).to(device))[0].cpu().numpy()
        action=int(np.where(eligible,scores,-np.inf).argmax());out=e0space.materialize(native,kk,action)
        return dict(native=native,kin=kk,tokens=tok,past=pp,eligible=eligible,features=features,scores=scores,action=action,paths=out)
    checks=[];timing=[]
    with torch.inference_mode():
        for device in ('cpu','cuda'):
            net.to(device);kin.to(device);ed.to(device)
            for i in range(32):
                o=pipeline(int(ids[i]),device)
                err={k:float(np.max(abs(o[k]-v[i]))) for k,v in (('native',cachenative),('kin',cachekin),('tokens',cachetokens),('past',cachepast))}
                if max(err.values())>2e-5:raise ValueError('Raw pipeline identity '+json.dumps(dict(device=device,id=int(ids[i]),errors=err)))
                checks.append(dict(device=device,sample_id=int(ids[i]),errors=err))
            guard.check();times=[]
            for i in range(550):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();o=pipeline(int(ids[i%32]),device)
                if device=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                if i>=50:times.append(elapsed)
                if i%50==0:guard.check()
            timing.append(dict(device=device,batch=1,CPU_threads=1,warm=50,measure=500,mean_seconds=float(np.mean(times)),p50_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),scope='real raw memory-map observation slice+existing neighbor valid sidecar+normalization+fullF16+KIN+400physicalchecks+324features+CPU/GPUtransfers+editor+chosenK20',disk_cold_start_excluded=True))
    hook.remove()
    assert all(all(torch.equal(v, m.state_dict()[k].detach().cpu()) for k,v in b.items()) for b,m in zip(before,(net,kin,ed)))
    assert all(not p.requires_grad and p.grad is None for m in (net,kin,ed) for p in m.parameters())
    for p,h in before_files.items():assert sha256(p)==h
    imported={}
    for m in list(sys.modules.values()):
        fn=getattr(m,'__file__',None)
        if fn:
            p=Path(fn).resolve()
            if p.is_file() and p.suffix=='.py' and (p.is_relative_to(ROOT) or p.is_relative_to(astra)):imported[str(p)]=sha256(p)
    report=dict(status='PASS_ACTUAL_FROZEN_RAW_PIPELINE_COST',timing=timing,parameter_counts=param_counts,total_unique_parameters=total,parameter_budget=305000,budget_pass=total<=305000,allweights_buffers_exact=True,no_backward_optimizer=True,no_GT_forward=True,identity32_each_device=checks,source_files_before_after=before_files,actual_imported_sources=imported,resource=guard.check(),limitations=['only fixed seed42 S representative cost, no model selection','CPU one thread and GPU batch1; not cache-only, not cold disk latency','existing neighbor num sidecar may encode future-availability curation; no future trajectory passed to pipeline','component identity <=2e-5 does not claim raw-pipeline argmax bitwise across devices or replace saved scientific evaluation'],no_delete=True)
    write_json(OUT/'COST_END_TO_END.json',report);print(json.dumps({k:v for k,v in report.items() if k in ('status','timing','parameter_counts','total_unique_parameters','parameter_budget','budget_pass')}))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write_json(OUT/('FAILURE_'+str(time.time_ns())+'.json'),dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),no_old_source_change=True));raise
