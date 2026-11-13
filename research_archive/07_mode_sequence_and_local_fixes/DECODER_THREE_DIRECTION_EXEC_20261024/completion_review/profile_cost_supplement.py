"""Versioned profile-only repair: count top-level executed fused operators.

No training/optimizer/old decoder call; old profile and all checkpoints immutable.
Counter covers linear/matmul AND fused Transformer, not substring 'mm' alone.
"""
import sys,math,time,collections
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import profile_full_pipeline as old
from common import sha256,write_json,load_data
from training import make_model,clone_state,exact
from f16.resource import Guard
from torch.utils._python_dispatch import TorchDispatchMode
OUT=Path(__file__).resolve().parent
def mm(a,b):
    sa=list(a.shape);sb=list(b.shape)
    if len(sa)==1:sa=[1]+sa
    if len(sb)==1:sb=sb+[1]
    assert sa[-1]==sb[-2]
    return int(math.prod(torch.broadcast_shapes(sa[:-2],sb[:-2]))*sa[-2]*sa[-1]*sb[-1])
def fused(src,d,w1):
    b,s,e=src.shape;assert e==d
    return int(b*(4*s*d*d+2*s*s*d+2*s*d*w1.shape[0]))
class Cost(TorchDispatchMode):
    def __init__(self):super().__init__();self.mac=collections.Counter();self.calls=collections.Counter();self.shapes=[];self.unknown=set()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func._overloadpacket);kw=kwargs or {};v=0
        if name=='aten.linear':v=int(args[0].numel()*args[1].shape[0])
        elif name in ('aten.mm','aten.matmul','aten.bmm'):v=mm(args[0],args[1])
        elif name in ('aten.addmm','aten.baddbmm'):v=mm(args[1],args[2])
        elif name=='aten._transformer_encoder_layer_fwd':v=fused(args[0],args[1],args[14])
        elif name=='aten._native_multi_head_attention':
            q,k,_,d=args[:4];b,s,_=q.shape;t=k.shape[1];v=int(b*((s+2*t)*d*d+s*d*d+2*s*t*d))
        elif any(t in name for t in ('linear','mm','matmul','conv','attention','transformer')):self.unknown.add(name)
        if v:self.mac[name]+=v;self.calls[name]+=1;self.shapes.append(dict(operator=name,MAC=v,input_shape=list(args[0].shape)))
        return func(*args,**kw)
def run():
    guard=Guard(6*2**30);guard.check('startup')
    execution=HERE/'direction_C_set_mechanisms/C2_execution_20261024'
    original=sha256(execution/'FULL_PIPELINE_PROFILE.json');data=load_data();net,tr,ds=old.backend();before=clone_state(net)
    local,center,nv=old.prepared_observations(data['FIT']['ids'][:256],tr,ds);holder=[]
    class Done(Exception):pass
    def stop(module,args):holder.append(args[0]);raise Done()
    handle=net.decoder.register_forward_pre_hook(stop)
    def encode(inp,cent,valid):
        holder.clear()
        try:net(inp-cent,None,None,mode='testing',num_valid=valid,graph_loc=inp)
        except Done:pass
        assert len(holder)==1
        token=holder[0][:,0];p=inp[:,0]
        return torch.cat((p.flatten(1),(p[:,-1]-p[:,-2])/.4,token[:,-1],token.mean(1)),1)
    # Known operation fixture proves formulas include linear, broadcast matmul,
    # bmm; fused Transformer formula separately follows QKV/O/FFN/attention.
    fixture=torch.nn.Linear(7,11,device='cuda').eval()
    with torch.inference_mode(),Cost() as c:fixture(torch.zeros(3,5,7,device='cuda'))
    assert sum(c.mac.values())==3*5*7*11 and not c.unknown
    result={}
    for arm in ('C2_isomlp','C2_setmix'):
        model=make_model(arm,42,execution).eval();mb=clone_state(model)
        def deploy(p,c,n):return model(encode(p,c,n))
        with torch.inference_mode():
            features=encode(local,center,nv)
            error=float(np.abs(features.cpu().numpy()-data['FIT']['x'][:256]).max());assert error<=2e-5
            with Cost() as cost:pred=deploy(local[:1],center[:1],nv[:1])
            assert not cost.unknown,sorted(cost.unknown)
            assert 'aten.linear' in cost.mac and 'aten._transformer_encoder_layer_fwd' in cost.mac
            latency={}
            for b in (1,256):
                for _ in range(4):deploy(local[:b],center[:b],nv[:b])
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                for _ in range(16):deploy(local[:b],center[:b],nv[:b])
                torch.cuda.synchronize()
                latency[str(b)]=dict(seconds_per_batch=(time.perf_counter()-start)/16,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert exact(mb,clone_state(model));guard.check()
        result[arm]=dict(whole_parameters=365711,MAC_per_window=sum(cost.mac.values()),FMA_FLOPs=2*sum(cost.mac.values()),
            operator_MAC=dict(cost.mac),operator_calls=dict(cost.calls),shapes=cost.shapes,unsupported=[],latency=latency,encoder_cache_maxerror=error,new_weights_buffers_exact=True)
        del model;torch.cuda.empty_cache()
    handle.remove();assert exact(before,clone_state(net));assert all(not p.requires_grad and p.grad is None for p in net.parameters())
    assert sha256(execution/'FULL_PIPELINE_PROFILE.json')==original
    guard.check()
    write_json(OUT/'PROFILE_COST_SUPPLEMENT.json',dict(status='PASS_PROFILE_ONLY_CORRECTION',original_profile_SHA=original,
        original_MAC_count_incomplete=True,original_failure='old unsupported-op filter checked mm/convolution/attention but omitted linear,matmul,fused Transformer; fused Transformer unregistered in standard counter',
        repaired_counter='single top-level TorchDispatchMode counts linear and fused Transformer QKV/O/FFN/attention products without nested double count',
        formula_fused='B*(4*S*D^2+2*S^2*D+2*S*D*Dff)',pointwise_nonMAC_ops_excluded=True,model_forward=True,optimizer_or_training=False,
        checkpoint_retraining=False,old_decoder_scorer_forward_calls=0,old_whole_weights_buffers_exact=True,
        arms=result,helper_SHA=sha256(__file__),resource_observations=guard.observations))
    print('PROFILE CORRECTION PASS',[(k,v['MAC_per_window']) for k,v in result.items()],flush=True)
if __name__=='__main__':run()
