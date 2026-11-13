"""Actual frozen-F16 -> new C2 profile; old decoder/scorer never called.

Build factory functions are AST-isolated from the old runner. Only selected
FIT observations are fed. Matmul/conv/attention FLOPs use the installed PyTorch
operator counter (FMA=2); other pointwise ops are explicitly not MACs.
"""
import ast,copy,importlib,importlib.util,os,sys,time,types
from pathlib import Path
sys.dont_write_bytecode=True
from common import *
import torch
from training import make_model,clone_state,exact
from f16.resource import Guard

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def backend():
    astra=Path(r'<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20')
    # This process already imported the new decoder's models.py through training.
    # ASTRA itself owns a package also named models. Preserve the former under a
    # private alias; make_model retains the actual class reference it imported.
    prior_models=sys.modules.get('models')
    if prior_models is not None and not hasattr(prior_models,'__path__'):
        sys.modules['_three_directions_decoder_models']=prior_models
        del sys.modules['models']
    # ASTRA/models has no __init__.py: implicit namespace search would otherwise
    # lose to the later regular new models.py. Explicitly bind its package path.
    astra_models=types.ModuleType('models');astra_models.__path__=[str(astra/'models')]
    astra_models.__package__='models';sys.modules['models']=astra_models
    for p in (astra,astra/'scripts',ROOT/'FRONT_ENCODER_SERIES'):sys.path.insert(0,str(p))
    previous=Path.cwd();os.chdir(astra)
    try:
        import sitecustomize
        from experiment_config import load_locked_q20
        cfg=load_locked_q20(ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml')
        cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False
        cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.
        cfg.device='cuda:0';cfg.device_list=[0]
        source=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py'
        names={'fold_data','fold_run','load_module','build_baseline'}
        nodes=[n for n in ast.parse(source.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes}==names
        ns=dict(Path=Path,importlib=importlib,sys=sys,torch=torch,os=os,copy=copy,
            DATA_ROOT=astra/'datasets/eth_ucy/processed_data_diverse',
            BASE_ROOT=ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
        front=module('_three_directions_profile_front',ROOT/'FRONT_ENCODER_SERIES/front_encoder_models.py')
        trainer=module('_three_directions_profile_trainer',ROOT/'FRONT_ENCODER_SERIES/train_front_variant.py')
        adapter=module('_three_directions_profile_adapter',ROOT/'E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py')
        net=ns['build_baseline']('eth',cfg,front,adapter,torch.device('cuda:0'))
        assert sum(p.numel() for n,p in net.named_parameters() if not n.startswith('decoder.'))==267159
        raw,num=ns['fold_data']('eth','train')
        dataset=trainer.TrajectoryDataset(raw,num,cfg)
    finally:os.chdir(previous)
    return net,trainer,dataset

def prepared_observations(ids,trainer,dataset):
    past=torch.stack([dataset[int(i)][0] for i in ids]).cuda()
    nv=torch.stack([dataset[int(i)][2] for i in ids]).cuda().int()
    local,_=trainer.normalize_to_focal_heading(past,past)
    return local,local.mean(-2,keepdim=True),nv

def run(execution,estimate,*,smoke_only=False):
    guard=Guard(estimate);guard.check('startup');data=load_data();net,tr,dataset=backend()
    before=clone_state(net);ids=data['FIT']['ids'][:256]
    local,center,nv=prepared_observations(ids,tr,dataset);holder=[]
    class EncoderFinished(Exception):pass
    def hook(owner,args):holder.append(args[0]);raise EncoderFinished()
    handle=net.decoder.register_forward_pre_hook(hook)
    def encode(inp,cent,valid):
        holder.clear()
        try:net(inp-cent,None,None,mode='testing',num_valid=valid,graph_loc=inp)
        except EncoderFinished:pass
        if len(holder)!=1:raise ValueError('ENCODER_STOP_BOUNDARY_NOT_REACHED')
        token=holder[0][:,0];pp=inp[:,0]
        return torch.cat((pp.flatten(1),(pp[:,-1]-pp[:,-2])/.4,token[:,-1],token.mean(1)),1)
    with torch.inference_mode():features=encode(local,center,nv)
    cached=data['FIT']['x'][:256]
    maxerror=float(np.abs(features.cpu().numpy()-cached).max())
    if maxerror>2e-5:raise ValueError(('ENCODER_CACHE_ALIGNMENT',maxerror))
    results={}
    if not smoke_only:
        from torch.utils.flop_counter import FlopCounterMode,flop_registry
        from torch.utils._python_dispatch import TorchDispatchMode
        registered={str(k) for k in flop_registry}
        class SeenOps(TorchDispatchMode):
            def __init__(self):super().__init__();self.names=set()
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                self.names.add(str(func._overloadpacket));return func(*args,**(kwargs or {}))
        for arm in ('C2_isomlp','C2_setmix'):
            model=make_model(arm,42,execution).eval();mb=clone_state(model)
            def deploy(inp,cent,valid):return model(encode(inp,cent,valid))
            with torch.inference_mode(),FlopCounterMode(display=False) as counter,SeenOps() as seen:
                deploy(local[:1],center[:1],nv[:1])
            candidates={n for n in seen.names if any(k in n for k in ('mm','convolution','attention'))}
            unsupported=sorted(candidates-registered)
            if unsupported:raise ValueError(('UNCOUNTED_MATMUL_CONV_ATTENTION_OP',unsupported))
            flops=int(counter.get_total_flops());assert flops>0 and flops%2==0
            latency={}
            for batch in (1,256):
                inp,cent,valid=local[:batch],center[:batch],nv[:batch]
                with torch.inference_mode():
                    for _ in range(4):deploy(inp,cent,valid)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    for _ in range(16):deploy(inp,cent,valid)
                    torch.cuda.synchronize()
                latency[str(batch)]=dict(seconds_per_batch=(time.perf_counter()-start)/16,
                    actual_observed_rows=batch,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                guard.check()
            assert exact(mb,clone_state(model))
            results[arm]=dict(whole_parameters=365711,full_model_MAC_per_window=flops//2,
                FLOPs_per_window=flops,MAC_convention='all executed registered matrix/convolution/attention FMA counted as2FLOPs; pointwise/nonlinear ops not MAC',
                includes_frozen_encoder=True,counts_all_executed_matmul_convolution_ops=True,
                counted_operators=sorted(candidates),unsupported_matmul_conv_attention_ops=unsupported,
                all_executed_operators=sorted(seen.names),latency=latency,new_decoder_weights_buffers_exact=True)
            del model;torch.cuda.empty_cache()
    handle.remove();assert exact(before,clone_state(net))
    assert all(not p.requires_grad and p.grad is None for p in net.parameters())
    guard.check()
    result=dict(status='PASS',arms=results,encoder_parameters=267159,encoder_cache_feature_maxerror=maxerror,
        cache_tolerance=2e-5,old_whole_weights_bool_buffers_exact=True,old_decoder_scorer_forward_calls=0,
        observation_ids=ids,source_digest=verify_sources(execution/'source_manifest.json') if not smoke_only else None,
        resource_observations=guard.observations,scope='SMOKE_ENCODER_ONLY' if smoke_only else 'ACTUAL_FULL_DEPLOYMENT_PROFILE',
        legacy_neighbor_sidecar='historical num_valid may encode future availability curation; reused without new selection',
        timing_scope='warm GPU observations prepared; no disk/dataset IO; FIT observations only')
    write_json(execution/('ENCODER_SMOKE.json' if smoke_only else 'FULL_PIPELINE_PROFILE.json'),result)
    del net;torch.cuda.empty_cache();return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--execution',required=True,type=Path);p.add_argument('--estimated-peak-gib',type=float,required=True)
    p.add_argument('--smoke-only',action='store_true');a=p.parse_args();a.execution.mkdir(parents=True,exist_ok=True)
    run(a.execution,int(a.estimated_peak_gib*2**30),smoke_only=a.smoke_only)
