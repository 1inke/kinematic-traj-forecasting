import hashlib,json
import common as c
import numpy as np
import torch
import training as tr

def donors(ids,tracks):
    tag=c.read('protocol.json')['cross_track_tag']
    keys=[hashlib.sha256((tag+json.dumps([str(t),int(i)],ensure_ascii=False,separators=(',',':'))).encode()).hexdigest() for i,t in zip(ids,tracks)]
    order=np.array(sorted(range(len(ids)),key=lambda i:(keys[i],int(ids[i]))));answer=np.full(len(ids),-1,int)
    for pos,row in enumerate(order):
        for j in range(1,len(order)):
            candidate=order[(pos+j)%len(order)]
            if tracks[candidate]!=tracks[row]:answer[row]=candidate;break
    return answer

@torch.no_grad()
def predict(model,d,ix,override=None):
    model.eval();before={k:v.detach().clone() for k,v in model.state_dict().items()};parts={k:[] for k in ('path','z','process_coeff')}
    for start in range(0,len(ix),128):
        c.guard();p,v,_,_,_=tr.tensors(d,ix[start:start+128])
        if override is None:out=model(p,v)
        else:out=model.decode_with_z(p,v,torch.as_tensor(override[start:start+128],dtype=torch.float32,device='cuda:0'),diagnostic=True)
        for k in parts:parts[k].append(out[k].cpu().numpy())
    tr.exact(before,model.state_dict());return {k:np.concatenate(v,0) for k,v in parts.items()}

def diagnostics(data,ix,out,H,normal=None):
    z=out['z'].astype(float);a=out['process_coeff'][:,:,0].astype(float);mask=data['m20'][ix];target=data['gt'][ix,-1] if H==12 else data['y20'][ix]
    result=dict(n=len(ix),m20=int(mask.sum()),mean_a_norm=float(np.linalg.norm(a,axis=-1).mean()),mean_z_norm=float(np.linalg.norm(z,axis=-1).mean()),mean_short_endpoint_minus_z=float(np.linalg.norm(out['path'][:,:,-1]-z,axis=-1).mean()),p12_z_a_identity_max_error=float(abs(out['path'][:,:,-1]-z-a).max()))
    if mask.any():
        err=np.linalg.norm(z[mask]-(12/H)*target[mask,None],axis=-1);result['goal_normalized_z_error_all_slots']=float(err.mean());result['goal_physical_m_error_all_slots']=float((H/12)*err.mean());result['goal_normalized_z_minK_error']=float(err.min(1).mean())
        e=np.linalg.norm(out['path'][mask]-data['gt'][ix[mask],None],axis=-1);j=np.maximum(e.mean(-1)/.2,e[:,:,-1]/.4).argmin(1);result['Jclosest_bound_goal_z_error']=float(err[np.arange(len(j)),j].mean())
    if normal is not None:
        result.update(mean_path_response=float(np.linalg.norm(out['path']-normal['path'],axis=-1).mean()),mean_endpoint_response=float(np.linalg.norm(out['path'][:,:,-1]-normal['path'][:,:,-1],axis=-1).mean()),mean_z_replacement=float(np.linalg.norm(z-normal['z'],axis=-1).mean()),normal_a_norm=float(np.linalg.norm(normal['process_coeff'][:,:,0],axis=-1).mean()))
    return result
