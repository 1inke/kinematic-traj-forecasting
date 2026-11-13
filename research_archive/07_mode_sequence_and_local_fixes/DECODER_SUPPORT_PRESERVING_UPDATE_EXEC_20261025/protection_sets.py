"""FIT-only frozen float64 success sets, exact parent contracts and source audit."""
from task_adapter import *

def prepare(data):
    folder=HERE/'identity_audit';folder.mkdir(exist_ok=True)
    records=[];parent_files=[]
    parentmf=read_json(OLD/'source_manifest.json')
    for v in parentmf['files'].values():assert Path(v['path']).is_file() and sha256(v['path'])==v['sha256'],('PARENT_SOURCE_CHANGED',v['path'])
    for s in SEEDS:
        run=OLD/f'runs/A_seed{s}';done=read_json(run/'done.json');mf=read_json(run/'run_manifest.json');blob=parent_blob(s)
        assert mf['contract']==done['contract']==blob['contract'];assert sha256(run/'checkpoint.pt')==mf['checkpoint_sha256']
        assert read_json(run/'history.json')==blob['history']
        assert all(rec['epoch']==i+1 and rec['steps']==74 for i,rec in enumerate(blob['history']))
        for rel,digest in done['outputs'].items():
            path=OLD/rel;assert path.is_file() and sha256(path)==digest,('PARENT_OUTPUT_CHANGED',rel);parent_files.append(path)
        for name,key in [('protocol.json','protocol'),('statistics.npz','stats'),('split_ids.npz','ids'),(f'initial_states/seed{s}.pt','initial'),(f'batch_orders/seed{s}.npz','orders'),(f'teachers/seed{s}.npz','teacher')]:assert sha256(OLD/name)==blob['contract'][key]
        with np.load(run/'predictions_FIT.npz',allow_pickle=False) as z:
            for k,target in [('sample_id','ids'),('native','native'),('gt','gt'),('groups','groups'),('track','track')]:assert np.array_equal(z[k],data['FIT'][target])
            q=errors(z['paths'],z['gt']);ng=errors(z['native'],z['gt'])['coverage'];ag=q['coverage'];membership=np.full(len(ag),-1,np.int8);membership[ng&ag]=0;membership[~ng&ag]=1
            np.savez_compressed(folder/f'seed{s}.npz',sample_id=z['sample_id'],native_good=ng,A0_good=ag,membership=membership,A0_ADE=q['ADE'],A0_FDE=q['FDE'],A0_J=q['J'],native_J=errors(z['native'],z['gt'])['J'])
        m,o,g=start(s);before=frozen_state(m)
        replay={}
        for split in ('FIT','inner'):
            with np.load(run/f'predictions_{split}.npz',allow_pickle=False) as z:
                for key,target in [('sample_id','ids'),('native','native'),('gt','gt'),('groups','groups'),('track','track')]:assert np.array_equal(z[key],data[split][target])
                p=predict(m,data[split]['x'][:128]);assert np.array_equal(p,z['paths'][:128]),'PARENT_SAVED_REPLAY_MISMATCH';replay[split]='128_bitwise'
        assert equal(before,frozen_state(m))
        records.append(dict(seed=s,n=len(ag),shared=int((membership==0).sum()),rescued=int((membership==1).sum()),native_only=int((ng&~ag).sum()),neither=int((~ng&~ag).sum()),label_SHA=sha256(folder/f'seed{s}.npz'),parent_replay=replay,optimizer_moments_step=592,parameter_order=parameter_order(m),parent_contract=blob['contract']))
        parent_files.extend([run/'done.json',run/'resume.pt',run/'run_manifest.json'])
        del m,o;torch.cuda.empty_cache()
    idx=sorted(range(18909),key=lambda i:hashlib.sha256(('support-step-v1|'+str(int(data['FIT']['ids'][i]))).encode()).digest())[:4096]
    np.savez_compressed(folder/'G0_fixed.npz',index=np.asarray(idx),sample_id=data['FIT']['ids'][idx])
    write_json(folder/'RESULT.json',dict(status='PASS_PARENT_SOURCE_ID_OPTIMIZER_RNG_REPLAY',records=records,unique_FIT=18909,unique_inner=1856,no_overlap=True,hash_G0_ids=arraysha(data['FIT']['ids'][idx]),inner_not_used_for_training_sets=True))
    return parent_files

def lock_sources(parent_files):
    inherited=read_json(OLD/'source_manifest.json');files=copy.deepcopy(inherited['files'])
    paths=list(DESIGN.glob('*.md'))+list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'protocol.json']+list((HERE/'identity_audit').glob('*.npz'))+parent_files+[OLD/'core.py',OLD/'source_manifest.json',ROOT/'src/f16/resource.py',ROOT/'src/f16/eval.py',ROOT/'src/f16/checkpoint.py',ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',OLD/'native_FIT.npz',OLD/'native_inner.npz']
    for p in paths:
        p=p.resolve();key=p.relative_to(ROOT).as_posix();v=dict(path=str(p),sha256=sha256(p))
        if key in files:assert v['sha256']==files[key]['sha256']
        else:files[key]=v
    digest=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in files.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(HERE/'source_manifest.json',dict(files=files,digest=digest,inherited_digest=inherited['digest'],inherited_count=len(inherited['files']),all_inherited_digests_preserved=True,locked=True))
    assert verify_manifest()==digest
    return len(files)
