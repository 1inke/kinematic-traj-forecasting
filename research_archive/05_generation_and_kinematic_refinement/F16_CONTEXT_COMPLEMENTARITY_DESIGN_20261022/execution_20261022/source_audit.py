"""Frozen128 hash-selected observations replay; no legacy run main is imported."""
from core import *
import ast,types,copy

def main():
    guard=Guard(E);guard.check('startup');stage('S0_SOURCE_INTERFACE_AUDIT',status='PREFLIGHT',active_worker=False)
    d=load_data();fit=d['FIT'];inn=d['inner']
    mean=fit['x'].astype(np.float64).mean(0);std=fit['x'].astype(np.float64).std(0);std[std<1e-6]=1
    np.savez_compressed(HERE/'statistics.npz',mean=mean.astype(np.float32),std=std.astype(np.float32),mean_float64=mean,std_float64=std,fit_ids=fit['ids'])
    np.savez_compressed(HERE/'split_ids.npz',FIT=fit['ids'],inner=inn['ids'])
    source=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    wanted={'fold_data','fold_run','load_module','build_baseline'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert len(nodes)==4
    astra=Path(r'<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20')
    for p in [astra,astra/'scripts',ROOT/'FRONT_ENCODER_SERIES']:sys.path.insert(0,str(p))
    os.chdir(astra);import sitecustomize
    from experiment_config import load_locked_q20
    cfg_path=ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml'
    cfg=load_locked_q20(cfg_path);cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.device='cuda:0';cfg.device_list=[0]
    ns=dict(Path=Path,importlib=__import__('importlib'),sys=sys,torch=torch,os=os,copy=copy,DATA_ROOT=astra/'datasets/eth_ucy/processed_data_diverse',BASE_ROOT=ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results')
    # Annotation names in selected functions are explicitly supplied.
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    front=module('_context_source_front',ROOT/'FRONT_ENCODER_SERIES/front_encoder_models.py')
    trainer=module('_context_source_trainer',ROOT/'FRONT_ENCODER_SERIES/train_front_variant.py')
    adapter=module('_context_source_adapter',ROOT/'E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py')
    net=ns['build_baseline']('eth',cfg,front,adapter,torch.device('cuda:0'));before=frozen_state(net)
    raw,num=ns['fold_data']('eth','train');dataset=trainer.TrajectoryDataset(raw,num,cfg)
    all_ids=np.concatenate((fit['ids'],inn['ids']));choice=sorted(range(len(all_ids)),key=lambda i:hashlib.sha256(f'context-s0-20261022::{int(all_ids[i])}'.encode()).digest())[:128]
    ids=all_ids[choice];past=torch.stack([dataset[int(i)][0] for i in ids]).cuda();nv=torch.stack([dataset[int(i)][2] for i in ids]).cuda().int()
    # Discard dataset future. Legacy num sidecar provenance is disclosed, not relabelled as new observation-only curation.
    local,_=trainer.normalize_to_focal_heading(past,past);center=local.mean(-2,keepdim=True)
    inp=torch.zeros((256,*local.shape[1:]),device='cuda');inp[:128]=local
    cent=torch.zeros((256,*center.shape[1:]),device='cuda');cent[:128]=center
    nvalid=torch.ones(256,dtype=torch.int32,device='cuda');nvalid[:128]=nv
    captured=[]
    h=net.decoder.register_forward_pre_hook(lambda owner,args:captured.append(args[0][:128,0].detach().cpu().numpy().copy()))
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():pred=net(inp-cent,None,None,mode='testing',num_valid=nvalid,graph_loc=inp)[2]
    torch.cuda.synchronize();latency=time.perf_counter()-start;h.remove()
    assert equal(before,frozen_state(net)) and all(p.grad is None and not p.requires_grad for p in net.parameters())
    z=np.load(prior.PATHS['encoder']);lookup={int(s):i for i,s in enumerate(z['sample_id'])};rows=[lookup[int(s)] for s in ids]
    tokenerr=float(np.abs(captured[0]-z['tokens'][rows]).max());pasterr=float(np.abs(local[:,0].cpu().numpy()-z['past'][rows]).max())
    relative=(pred[:128,0]+center[:,0,None]-local[:,0,-1,None,None]).cpu().numpy()
    with np.load(prior.PATHS['cache']) as cc:native=cc['candidate'][rows,:,:24].reshape(-1,20,12,2)
    patherr=float(np.abs(relative-native).max())
    assert max(tokenerr,pasterr,patherr)<=2e-5,(tokenerr,pasterr,patherr)
    np.savez_compressed(HERE/'source_audit/replay128.npz',sample_id=ids,tokens=captured[0],past=local[:,0].cpu().numpy(),paths=relative)
    encoder_count=sum(p.numel() for n,p in net.named_parameters() if not n.startswith('decoder.'))
    full_count=sum(p.numel() for p in net.parameters())
    # Actual runtime imported source files, plus raw identity-bearing inputs and checkpoints.
    paths=[source,cfg_path,raw,num,ROOT/'GENERATION_NATIVE_DECODER_SERIES/run_native.py',ROOT/'GENERATION_NATIVE_DECODER_SERIES/encoder_audit.json']+list(prior.PATHS.values())
    for stage_name in ['h3','v2']:paths += [ns['fold_run']('eth',stage_name)/'best_selected.pth',ns['fold_run']('eth',stage_name)/'split_indices.npz']
    exposure=[]
    for stage_name in ['h3','v2']:
        with np.load(ns['fold_run']('eth',stage_name)/'split_indices.npz') as sp:
            train=sp['train'];exposure.append(dict(stage=stage_name,FIT_seen=int(np.isin(fit['ids'],train).sum()),inner_seen=int(np.isin(inn['ids'],train).sum())))
    for m in list(sys.modules.values()):
        fn=getattr(m,'__file__',None)
        if fn:
            p=Path(fn).resolve()
            if p.suffix=='.py' and (p.is_relative_to(ROOT) or p.is_relative_to(astra)):paths.append(p)
    report=dict(status='PASS_EXPLORATORY_ONLY',sample_ids=ids.tolist(),selection='sha256 context-s0-20261022 + ID, no quality selection',compute_batch=256,real_rows=128,token_max_abs=tokenerr,past_max_abs=pasterr,native_path_max_abs=patherr,tolerance=2e-5,all_weights_bool_scalers_exact=True,encoder_parameters=encoder_count,whole_legacy_model_parameters=full_count,encoder_plus_student_parameters=encoder_count+82008,whole_baseline_forward256_seconds=latency,exposure=exposure,legacy_neighbor_sidecar='original eth_num_train.npy may encode future-availability curation; no new neighbor choice or future passed to normalize/model; not clean whole-model generalization',source_paths=[str(p.resolve()) for p in sorted(set(paths)) if p.is_file()])
    # Deployment path stops before the old decoder: frozen encoder -> compressed tokens -> new student.
    student=new_model(42);student.eval();mac=[0]
    def linear_mac(owner,args,out):mac[0]+=int(np.prod(out.shape[:-1]))*owner.in_features*owner.out_features
    hooks=[layer.register_forward_hook(linear_mac) for name,layer in net.named_modules() if isinstance(layer,nn.Linear) and not name.startswith('decoder.')]
    hooks += [layer.register_forward_hook(linear_mac) for layer in student.modules() if isinstance(layer,nn.Linear)]
    class EncoderDone(Exception):pass
    holder=[]
    def stop_decoder(owner,args):holder.append(args[0]);raise EncoderDone()
    hook=net.decoder.register_forward_pre_hook(stop_decoder)
    def deployment():
        holder.clear()
        try:net(inp-cent,None,None,mode='testing',num_valid=nvalid,graph_loc=inp)
        except EncoderDone:pass
        zt=holder[0][:,0];pp=inp[:,0]
        xx=torch.cat((pp.flatten(1),(pp[:,-1]-pp[:,-2])/.4,zt[:,-1],zt.mean(1)),1)
        return student(xx)
    with torch.inference_mode():
        for _ in range(4):deployment()
        mac[0]=0;torch.cuda.synchronize();timer=time.perf_counter()
        for _ in range(8):yy=deployment()
        torch.cuda.synchronize();sec=(time.perf_counter()-timer)/8
    hook.remove()
    for h in hooks:h.remove()
    report.update(encoder_plus_student_latency256_seconds=sec,encoder_plus_student_linear_MAC_per_window=mac[0]/8/256,MAC_scope='actual Linear multiply-accumulate hooks only; attention matmul, graph/RBF/nonlinear ops excluded, not total end-to-end MAC',deployment_legacy_decoder_not_called=True,latency_scope='GPU warm cached raw prepared observations; no file IO/dataset ingestion; 128real+128padding compute256')
    write_json(HERE/'source_audit/RESULT.json',report)
    write_json(HERE/'input_contract.json',dict(cache=str(prior.PATHS['encoder']),cache_sha256=sha256(prior.PATHS['encoder']),checkpoint=str(ns['fold_run']('eth','v2')/'best_selected.pth'),checkpoint_sha256=sha256(ns['fold_run']('eth','v2')/'best_selected.pth'),token_producer='GENERATION_NATIVE_DECODER_SERIES/run_native.py extract decoder-prehook args[0][:,0], exact same h3/v2 build_baseline',past_coordinate='focal-heading local meters, last observed zero',feature_dimension=210,FIT=len(fit['ids']),inner=len(inn['ids']),frozen_boundary='no old parameters trainable',exploratory_exposure=exposure))
    del net,before;torch.cuda.empty_cache();guard.check();write_json(HERE/'source_audit/RESOURCE.json',guard.observations)
    print('S0 PASS',tokenerr,pasterr,patherr,flush=True)

if __name__=='__main__':
    (HERE/'source_audit').mkdir(parents=True,exist_ok=True)
    try:main()
    except Exception:
        import traceback;write_json(HERE/'source_audit/FAILURE.json',dict(traceback=traceback.format_exc()));raise
