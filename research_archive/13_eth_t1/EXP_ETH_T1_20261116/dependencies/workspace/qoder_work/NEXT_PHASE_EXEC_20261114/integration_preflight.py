"""FIT-only saved-output pipeline checks; no formal experiment population."""
import copy
import shutil
from execution import *
import analysis_phase as an
import evaluate_saved as ev


def main():
    torch.set_num_threads(1)
    out=HERE/'preflight_integration';out.mkdir(exist_ok=False)
    data=data_read('fit');indices=np.array(sorted(range(len(data['ids'])),key=lambda i:__import__('hashlib').sha256(str(data['ids'][i]).encode()).digest())[:513]);d=subset(data,indices)
    t=Trainer(data,indices,epochs=2)
    # Same actual GPU paths copied to labelled fixture populations to test exact-zero contrasts.
    paths,logits=infer(t.ema.module,d)
    (out/'C0').mkdir();np.savez_compressed(out/'C0/dev.npz',**d)
    for name in ('EVALUATION_CONTRACT.json','MOTION_REFERENCE.json'):shutil.copyfile(HERE/'C0'/name,out/'C0'/name)
    old_ev=ev.HERE;ev.HERE=out
    ev.save_population(out/'runs/control/dev',paths,logits,d)
    ev.HERE=old_ev
    for name in ('b1','b2','b3','m1','m2','m3'):
        shutil.copytree(out/'runs/control',out/'runs'/name)
    original=an.HERE;an.HERE=out
    nn=an.null(['b1','b2','b3'],'dev');contract={'null':{'dev':nn}}
    cmp=an.compare(['m1','m2','m3'],['b1','b2','b3'],'dev',contract)
    if not cmp['no_harm']:raise ValueError('IDENTITY_COMPARISON_HARM')
    boots=an.bootstrap_compare(['m1','m2','m3'],['b1','b2','b3'])
    if any(x['point']!=0 or x['CI95']!=[0.,0.] for g in boots.values() for x in g.values()):raise ValueError('IDENTITY_BOOTSTRAP')
    an.HERE=original
    g=guard();mechanisms={}
    for arm in ('E2_clamped_launch','E3_self','E3_set','E3_set_nobias'):
        t=Trainer(data,indices[:64],batch=8,epochs=1,arm=arm)
        for _ in range(4):t.step()
        p1,diag=infer_extra(t.ema.module,subset(d,np.arange(32)),g)
        if arm.startswith('E3_') and p1.shape!=(32,20,12,2):raise ValueError('FIRST_PASS_SHAPE')
        if any(v.shape[0]!=32 for v in diag.values()):raise ValueError('DIAGNOSTIC_ID_ALIGNMENT')
        mechanisms[arm]={k:list(v.shape) for k,v in diag.items()}
    write_json(HERE/'PREFLIGHT_INTEGRATION.json',dict(status='PASS',at=now(),fixture_split='513 hashed FIT windows only',
        identity_all_gates=True,identity_bootstrap_20000draws_4groups_4metrics=True,mechanisms=mechanisms))
    print('INTEGRATION_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
