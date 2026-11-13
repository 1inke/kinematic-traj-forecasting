from core import *
from feature_blocks import assemble,read_stats
from train import torch,DiagnosticEditor,read_model,contract,G,exact,copy
def main(arm,seed,split):
    lock=read_json(HERE/'ALL_PROBE_FINALS_LOCK.json');folder=HERE/'runs'/f'{arm}_seed{seed}'
    assert sha256(folder/'final.pt')==lock['models'][f'{arm}_seed{seed}']
    c=contract(arm,seed);s=read_model(folder,c);m=DiagnosticEditor(*read_stats(seed),arm).cuda();m.load_state_dict(s['model'],strict=True);m.eval();before=copy.deepcopy(m.state_dict())
    out=HERE/'populations'/f'{arm}_seed{seed}'/split;out.mkdir(parents=True,exist_ok=True)
    pc=dict(source=source(),arm=arm,seed=seed,split=split,finals=sha256(HERE/'ALL_PROBE_FINALS_LOCK.json'))
    if (out/'done.json').exists():completed(out,pc);return
    if list(out.iterdir()):raise FileExistsError('Preserve partial population')
    ob=obs(split)
    with np.load(HERE/f'prepared/observed_blocks_{split}.npz') as z:token=z['token'];neighbor=z['neighbor'];assert np.array_equal(z['sample_id'],ob['sample_id'])
    fb=module('_step3_original_features',OLD/'feature_builder.py');lo=0;chunks=[]
    with torch.inference_mode():
        for j,z in enumerate(aux_chunks(seed,split)):
            n=len(z['sample_id']);assert np.array_equal(z['sample_id'],ob['sample_id'][lo:lo+n])
            base=np.array([fb.build(p,t,na,ki,mo) for p,t,na,ki,mo in zip(ob['past'][lo:lo+n],token[lo:lo+n],z['native'],z['kin'],z['motion'])],np.float32)
            scores=[]
            for k in range(0,n,32):
                raw=assemble(base[k:k+32],token[lo+k:lo+min(k+32,n)],neighbor[lo+k:lo+min(k+32,n)])
                scores.append(m(torch.as_tensor(raw,device='cuda')).cpu().numpy())
            scores=np.concatenate(scores).astype(np.float64);assert np.isfinite(scores).all()
            chosen=np.where(z['eligible'],scores,-np.inf).argmax(1);paths=z['native'].copy()
            for i,a in enumerate(chosen):
                if a:r,l=divmod(int(a)-1,20);paths[i,r]=z['kin'][i,l]
            name=f'chunk{j:04d}.npz';save_npz(out/name,scores=scores,chosen=chosen,paths=paths,sample_id=z['sample_id'],eligible=z['eligible'],B=z['B'],target=z['target']);chunks.append(name);lo+=n
            G.check();mark('STEP3_SAVE_ALL401',arm=arm,seed=seed,split=split,saved_windows=lo)
    exact(before,m.state_dict());resource_report(out,G);finish(out,pc,chunks=chunks,n=lo,eval_allparamsbuffers_exact=True)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),sys.argv[3])
