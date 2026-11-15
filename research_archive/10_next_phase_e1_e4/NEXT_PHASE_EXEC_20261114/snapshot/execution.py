"""Fresh, isolated phase runner. No historical launcher is imported or invoked."""
import argparse
import gc
import os
import time
import traceback
from common import *
from train_core import *
from evaluate_saved import save_population
import resource_guard as rg
import exchange_null as ex

OLD = ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'


def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def status(**kw):
    atomic_json(HERE/'CURRENT.json',dict(at=now(),**kw))


def data_read(split):
    return dict(np.load(HERE/'C0'/f'{split}.npz',allow_pickle=False))


def subset(data,indices):
    return {k:v[indices] for k,v in data.items()}


def sealed():
    verify(read_json(HERE/'SOURCE_MANIFEST.json')['files'])


def guard():
    return rg.Guard(read_json(HERE/'PREFLIGHT_E1.json')['estimated_peak_bytes'])


def spec_for(name,arm,seed,order,epochs,split='fit'):
    return dict(name=name,arm=arm,seed=seed,order=order,epochs=epochs,split=split)


def indices_for(spec,data):
    if spec['split']=='fit85':
        return np.load(HERE/'E1/calib_split.npz')['fit85_index']
    return np.arange(len(data['ids']))


def train(spec):
    sealed(); torch.set_num_threads(1)
    if spec['arm']!='B0':
        seal=read_json(HERE/'E2E_V2_SEAL.json')
        verify(seal['files'])
        if not (HERE/'PREFLIGHT_METHODS.json').exists(): raise ValueError('METHOD_PREFLIGHT_REQUIRED')
    out=HERE/'runs'/spec['name'];out.mkdir(parents=True,exist_ok=False)
    status(stage='TRAIN',run=spec['name'],epoch=0,epochs=spec['epochs'])
    start=time.perf_counter(); g=guard();data=data_read('fit');idx=indices_for(spec,data)
    t=Trainer(data,idx,epochs=spec['epochs'],seed=spec['seed'],order_seed=spec['order'],arm=spec['arm'])
    contract=dict(spec=spec,source=sha(HERE/'SOURCE_MANIFEST.json'),fit=sha(HERE/'C0/fit.npz'),
                  final_selection='FIXED_FINAL_EMA_0.995',batch=64,clip=5)
    if spec['arm']!='B0': contract['E2E_V2']=sha(HERE/'E2E_V2_CONTRACT.json')
    save_checkpoint(out/'initial.pt',t.snapshot(contract));write_json(out/'contract.json',contract)
    if spec['arm']!='B0':
        cc=read_json(HERE/'E2E_V2_CONTRACT.json')
        reference=cc['reference_names'][spec['seed']-42]
        reference_root=OLD/'B0_seed42' if reference=='OLD42' else HERE/'runs'/reference
        initial=torch.load(reference_root/'initial.pt',map_location='cpu',weights_only=False)
        shared=t.model.state_dict()
        for key,value in initial['model'].items():exact(value,shared[key],'FORMAL_SHARED_INIT/'+key)
        exact(initial['rng'],rng_state(t.generator),'FORMAL_PAIRED_ALL_INITIAL_RNG')
        del initial
    if spec['arm']=='B0' and spec['seed']==42:
        old=torch.load(OLD/'B0_seed42/initial.pt',map_location='cpu',weights_only=False)
        exact(old['model'],cpu_tree(t.model.state_dict()),'OLD_B0_INITIAL_SHARED');del old
    epoch_rows=[]
    for ep in range(1,spec['epochs']+1):
        st=time.perf_counter();losses=[]
        for _ in range(t.steps_per_epoch):
            row=t.step();losses.append(row['loss'])
            try:g.check()
            except MemoryError:
                save_checkpoint(out/f'resource_stop_{t.updates}.pt',t.snapshot(contract));raise
        row=dict(epoch=ep,mean_train_loss=float(np.mean(losses)),updates=t.updates,seconds=time.perf_counter()-st)
        epoch_rows.append(row);write_json(out/f'epoch{ep:02}.json',row)
        # All runs retain final; E1-B E64 retains every full resumable epoch as designed.
        if spec['split']=='fit85' and spec['epochs']==64:
            save_checkpoint(out/f'epoch{ep:02}.pt',t.snapshot(contract))
            # CALIB only, NEVER dev; eval preserves parameters and RNG.
            cal=subset(data,np.load(HERE/'E1/calib_split.npz')['calib_index'])
            paths,logits=infer(t.ema.module,cal,guard=g)
            summary,_=ex.summarize(dict(paths=paths,logits=logits),cal)
            write_json(out/f'calib_epoch{ep:02}.json',summary);del paths,logits,cal
        status(stage='TRAIN',run=spec['name'],epoch=ep,epochs=spec['epochs'],updates=t.updates,epoch_seconds=row['seconds'])
        print('EPOCH',spec['name'],row,flush=True)
    save_checkpoint(out/'final.pt',t.snapshot(contract));write_json(out/'epoch_summary.json',epoch_rows)
    if spec['arm']!='B0':
        reference_final=torch.load(reference_root/'final.pt',map_location='cpu',weights_only=False)
        exact(reference_final['batch_orders'],t.batch_orders,'FORMAL_ALL_EPOCH_BATCH_IDS')
        write_json(out/'PAIRED_CONTROL_AUDIT.json',dict(status='PASS',reference=reference,shared_init_all_tensors=True,
            global_initial_RNG=True,all_batch_IDs=True,epochs=spec['epochs'],updates=t.updates))
        del reference_final
    write_json(out/'FINAL_LOCK.json',dict(at=now(),files=seal([out/'initial.pt',out/'final.pt',out/'contract.json']),
        updates=t.updates,seconds=time.perf_counter()-start,resource=rg.snapshot()))
    sealed()


@torch.no_grad()
def infer_extra(model,data,g):
    """Save focal first-pass world paths and model-internal mechanism diagnostics."""
    model.eval();device=next(model.parameters()).device
    pass1=[];diagnostics={}
    for st in range(0,len(data['ids']),64):
        p=torch.as_tensor(data['past'][st:st+64],device=device)
        n=torch.as_tensor(data['num_valid'][st:st+64],device=device)
        out,logits,_,transform=prediction(model,p,n)
        dec=model.decoder
        if hasattr(dec,'last_pass1_relative'):
            relative=dec.last_pass1_relative[:,:1]
            cr=torch.einsum('tp,bakpc->baktc',dec.curve_W,relative)
            pp,_,graph,center,origin,matrix=prepare(p)
            first=pp[:,:1,None,-1:]+cr
            pass1.append(world_paths(first,transform).double().cpu().numpy())
            source=dec.refiner.diagnostics
        else: source=getattr(dec,'diagnostics',{})
        for key,value in source.items():
            if value is not None:
                arr=value.detach().cpu().numpy()
                if isinstance(dec, __import__('curve_decoder').CurveAnchorDecoder):arr=arr[:,0]
                else:arr=arr.reshape(len(p),9,*arr.shape[1:])[:,0]
                diagnostics.setdefault(key,[]).append(arr)
        g.check()
    return (np.concatenate(pass1) if pass1 else None), {k:np.concatenate(v) for k,v in diagnostics.items()}


def evaluate(spec,full=True):
    sealed();torch.set_num_threads(1);g=guard();out=HERE/'runs'/spec['name']
    lock=read_json(out/'FINAL_LOCK.json');verify(lock['files'])
    data=data_read('fit');t=Trainer(data,indices_for(spec,data),epochs=spec['epochs'],seed=spec['seed'],order_seed=spec['order'],arm=spec['arm'])
    state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    t.ema.module.load_state_dict(state['ema'],strict=True);del state
    splits=['fit','dev'] if full else ['calib']
    for split in splits:
        status(stage='EVALUATE_AND_AUDIT',run=spec['name'],split=split)
        d=subset(data,np.load(HERE/'E1/calib_split.npz')['calib_index']) if split=='calib' else data_read(split)
        paths,logits=infer(t.ema.module,d,guard=g)
        target=out/split
        if full:save_population(target,paths,logits,d,g)
        else:
            target.mkdir(exist_ok=False)
            np.savez_compressed(target/'predictions.npz',paths=paths,logits=logits,ids=d['ids'])
        summary,q=ex.summarize(dict(paths=paths,logits=logits),d,ex.fit_reference(HERE/'C0/MOTION_REFERENCE.json'))
        write_json(target/'summary_v2.json',summary)
        if spec['arm'].startswith(('E2_','E3_')):
            p1,diagnostics=infer_extra(t.ema.module,d,g)
            if p1 is not None:
                np.savez_compressed(target/'pass1.npz',paths=p1,logits=logits,ids=d['ids'])
                q1=ex.quality(p1,d['future'])
                write_json(target/'INTERNAL_EXCHANGE.json',ex.pair_exchange(q1['coverage'],q['coverage'],d['group']))
            if diagnostics:np.savez_compressed(target/'mechanisms.npz',**diagnostics,ids=d['ids'])
        del d,paths,logits,q;gc.collect()
    write_json(out/'EVALUATION_DONE.json',dict(at=now(),files=seal([p for p in out.rglob('*') if p.is_file() and p.name not in ('EVALUATION_DONE.json',)])))
    sealed()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','eval']);ap.add_argument('spec');a=ap.parse_args()
    spec=json.loads(a.spec)
    if a.mode=='train':train(spec)
    else:evaluate(spec,spec['split']!='fit85')


if __name__=='__main__':main()
