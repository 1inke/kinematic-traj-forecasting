"""Independent CPU saved-vector/primal/geometry checks; never import model/optimizer."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def audit():
    out=HERE/'G0';r=read(out/'RESULT.json');packets=read(out/'PACKETS.json');assert len(packets)==48
    for name,digest in r['output_SHA'].items():assert sha(out/name)==digest
    max_cast=0.;max_R=0.;support={};harm={};conflict={};ratios=[];zeros=[]
    for packet in packets:
        with np.load(out/packet['vectors'],allow_pickle=False) as z:
            d=z['U_d0'];G=z['P_G'];P=z['P_dP'];before=z['U_theta0'];tol=1e-10*max(1,np.linalg.norm(d))
            assert np.array_equal(d,z['N_d0']) and np.array_equal(d,z['P_d0'])
            # Independent enumeration in the row span, not the production projector.
            candidates=[]
            if not len(G) or np.max(G@d)<=tol:candidates.append(d)
            for i in range(len(G)):
                v=d-G[i]*np.dot(G[i],d)
                if np.max(G@v)<=tol and np.dot(G[i],d)>=-tol:candidates.append(v)
            if len(G)==2:
                lam=np.linalg.lstsq(G@G.T,G@d,rcond=1e-12)[0];v=d-G.T@lam
                if np.max(G@v)<=tol and min(lam)>=-tol:candidates.append(v)
            assert candidates
            ref=min(candidates,key=lambda v:np.linalg.norm(v-d)**2)
            assert np.linalg.norm(P-ref)<=2e-9*max(1,np.linalg.norm(d))
            assert not len(G) or max(G@P)<=tol
            N=z['N_dN'];assert abs(np.linalg.norm(P)-np.linalg.norm(N))<=tol
            for arm in ('U','N','P'):
                ideal=z[arm+'_d0'] if arm=='U' else N if arm=='N' else P
                actual=z[arm+'_d_actual'];reconstructed=(before+ideal).astype(np.float32).astype(float)-before
                assert np.array_equal(actual,reconstructed)
                err=np.linalg.norm(actual-ideal);max_cast=max(max_cast,float(err))
                assert not len(G) or np.max(np.abs(G@actual-G@ideal))<=err+tol
        with np.load(out/packet['paths'],allow_pickle=False) as z:
            group=z['membership'];y=z['gt'].astype(float)
            quality={}
            for arm in ('before','U','N','P'):
                p=z[arm].astype(float);ee=np.sqrt(((p-y[:,None])**2).sum(-1));j=np.maximum(ee.sum(2)/12/.2,ee[:,:,11]/.4).min(1)
                quality[arm]=[float(j[group==k].mean()) if np.any(group==k) else None for k in (0,1)]
            for arm in ('U','N','P'):
                a=packet['arms'][arm]
                for values,key in [(quality['before'],'R_before'),(quality[arm],'R_after')]:
                    for v,old in zip(values,a[key]):
                        if old is not None:max_R=max(max_R,abs(v-old));assert abs(v-old)<=2e-6
                assert abs(a['h']-sum(max(v-b,0) for v,b in zip(a['R_after'],a['R_before']) if b is not None))<2e-12
        if packet['valid_double_group']:
            s=str(packet['seed']);support[s]=support.get(s,0)+1
            conflict[s]=conflict.get(s,0)+int(any(v>packet['arms']['U']['solver_tolerance'] for v in packet['arms']['U']['raw_Gd0']))
            harm.setdefault(s,{a:[] for a in ('U','N','P')})
            for a in harm[s]:harm[s][a].append(packet['arms'][a]['h'])
            if packet['arms']['U']['d0_norm']!=0:ratios.append(packet['arms']['P']['norm_ratio'])
            zeros.append(packet['arms']['P']['exact_zero_P'])
    d=read(out/'DECISION.json');avg={a:float(np.mean([np.mean(harm[str(s)][a]) for s in (42,43,44)])) for a in ('U','N','P')}
    assert d['valid_batches']==support
    for a in avg:assert abs(avg[a]-d['equal_seed_mean_h'][a])<=2e-12
    assert all(abs(conflict[s]/support[s]-d['conflict_fractions'][s])<=2e-12 for s in support)
    assert abs(float(np.median(ratios))-d['norm_ratio_median'])<=2e-12
    gates=dict(support=all(v>=12 for v in support.values()),conflict=sum(conflict[s]/support[s]>=.2 for s in support)>=2,actual_harm=avg['U']>0,mean_h_P_half_U=avg['U']>0 and avg['P']<=.5*avg['U'],mean_h_P_less_N=avg['P']<avg['N'],each_seed_P_le_U=all(np.mean(harm[s]['P'])<=np.mean(harm[s]['U']) for s in harm),norm_median=np.median(ratios)>=.25,zero_fraction=np.mean(zeros)<=.1,numerical_certificates=True)
    assert gates==d['gates']
    result=dict(status='PASS_INDEPENDENT_SAVED_G0',packets=48,max_cast_norm=max_cast,max_float32_R_vs_saved_double_geometry_error=max_R,linear_certificates_not_finite_step_guarantees=True,decision=d['status'],no_model_optimizer_gradient_replay=True)
    (out/'REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result

if __name__=='__main__':print(audit())
