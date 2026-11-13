"""Independent NumPy/SciPy saved-path audit, not training metric reuse."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def independent_metrics(z):
    paths=np.asarray(z['paths'],float);truth=np.asarray(z['gt'],float);native=np.asarray(z['native'],float);groups=z['groups'];ids=z['sample_id'];track=z['track']
    if paths.shape!=(len(ids),20,12,2) or truth.shape!=(len(ids),12,2) or native.shape!=paths.shape or len(np.unique(ids))!=len(ids) or len(groups)!=len(ids) or len(track)!=len(ids) or not set(np.unique(groups)).issubset({0,1,2}):raise ValueError('Invalid audited identity/shape')
    if not np.isfinite(paths).all() or not np.isfinite(truth).all() or not np.isfinite(native).all():raise ValueError('Nonfinite audited arrays')
    def err(x):
        distance=np.sqrt(np.sum((x-truth[:,None])**2,axis=-1));a=np.sum(distance,axis=2)/12;f=distance[:,:,11];jj=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);q=np.argmin(a+.5*f,axis=1);j=np.argmin(jj,axis=1);return distance,a,f,jj,good,q,j
    e,a,f,j,g,q,jw=err(paths);ne,na,nf,nj,ng,nq,njw=err(native);good=g.any(1);base=ng.any(1);cc=[]
    for row,gg in zip(paths,g):
        distances=[]
        for aa in range(20):
            for bb in range(aa+1,20):
                v=np.sqrt(np.sum((row[aa]-row[bb])**2,axis=-1));distances.append(max(float(v.mean()/.1),float(v[-1]/.2)))
        labels=fcluster(linkage(np.array(distances),method='complete'),1,criterion='distance');cc.append([len(set(labels)),len(set(labels[gg]))])
    cc=np.asarray(cc)
    def physics(x):
        v=np.concatenate([x[:,:,:1],x[:,:,1:]-x[:,:,:-1]],axis=2)/.4;acc=(v[:,:,1:]-v[:,:,:-1])/.4;jerk=(acc[:,:,1:]-acc[:,:,:-1])/.4
        sv,sa,sj=[np.sqrt(np.sum(xx*xx,axis=-1)) for xx in (v,acc,jerk)]
        return dict(speed_p95=float(np.percentile(sv,95)),accel_mean=float(sa.mean()),accel_p95=float(np.percentile(sa,95)),jerk_mean=float(sj.mean()),jerk_p95=float(np.percentile(sj,95)))
    out={}
    for name,number in [('overall',None),('stable',0),('transition',1),('change',2)]:
        take=np.ones(len(ids),bool) if number is None else groups==number;n=int(take.sum())
        if not n:out[name]={'n':0};continue
        indices=np.flatnonzero(take);mot=physics(paths[take]);nm=physics(native[take]);lost=~good&base&take;rescued=good&~base&take;den=int(base[take].sum());drift=np.sqrt(np.sum((paths[take]-native[take])**2,axis=-1))
        out[name]=dict(n=n,tracks=len(np.unique(track[take])),good=int(good[take].sum()),native_good=den,coverage=float(np.mean(good[take])),rescue=int(rescued.sum()),lost=int(lost.sum()),retained=int((good&base&take).sum()),lost_rate_native_good=float(lost.sum()/den) if den else None,ADE=float(np.mean(a[take].min(1))),FDE=float(np.mean(f[take].min(1))),J=float(np.mean(j[take].min(1))),Q_winner_ADE=float(a[indices,q[take]].mean()),Q_winner_FDE=float(f[indices,q[take]].mean()),Q_segments=e[indices,q[take]].reshape(-1,3,4).mean(0).mean(-1).tolist(),J_segments=e[indices,jw[take]].reshape(-1,3,4).mean(0).mean(-1).tolist(),motion=mot,native_motion=nm,motion_pass=all(mot[k]<=1.01*nm[k] for k in mot),correction_mean=float(drift.mean()),correction_p95=float(np.percentile(drift,95)),correction_max=float(drift.max()),rescue_IDs=ids[rescued].tolist(),lost_IDs=ids[lost].tolist(),distinct=float(cc[take,0].mean()),effective_good=float(cc[take,1].mean()))
    return out

def compare(saved,recomputed,tolerance=2e-12,prefix=''):
    diffs={};mismatches=[]
    def walk(a,b,key):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b)):
                if k not in a or k not in b:diffs[key+'/'+k]=float('inf');mismatches.append(key+'/'+k)
                else:walk(a[k],b[k],key+'/'+k)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):diffs[key]=float('inf');mismatches.append(key)
            else:
                for i,(x,y) in enumerate(zip(a,b)):walk(x,y,key+f'/{i}')
        else:
            if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):
                diff=abs(float(a)-float(b));allowed=0 if isinstance(a,int) and isinstance(b,int) else tolerance
            else:diff=0. if a==b else float('inf');allowed=0
            diffs[key]=diff
            if not math.isfinite(diff) or diff>allowed:mismatches.append(key)
    walk(saved,recomputed,prefix)
    largest=max(diffs.values(),default=0)
    return dict(status='PASS' if not mismatches else 'FAIL',max_abs_diff=largest if math.isfinite(largest) else None,n_fields=len(diffs),mismatches=mismatches)

def audit_file(prediction,result,tolerance=2e-12,*,frozen_cache=None,frozen_labels=None,frozen_metadata=None,frozen_manifest=None,root=None):
    paths=[Path(prediction),Path(result)];before={str(p):sha(p) for p in paths}
    frozen_verified=False
    with np.load(prediction,allow_pickle=False) as z:
        if frozen_cache is not None or frozen_labels is not None:
            if frozen_cache is None or frozen_labels is None or frozen_metadata is None or frozen_manifest is None or root is None:raise ValueError('Complete frozen input contract required')
            manifest=json.loads(Path(frozen_manifest).read_text(encoding='utf-8-sig'))['files']
            for fp in [Path(frozen_cache),Path(frozen_labels),Path(frozen_metadata)]:
                key=fp.relative_to(root).as_posix()
                if manifest.get(key)!=sha(fp):raise ValueError('Frozen input digest mismatch')
                before[str(fp)]=sha(fp);paths.append(fp)
            with np.load(frozen_cache,allow_pickle=False) as cache,np.load(frozen_labels,allow_pickle=False) as labels:
                cids=cache['sample_id'];lids=labels['sample_id'];lookup={int(s):i for i,s in enumerate(cids)};llook={int(s):i for i,s in enumerate(lids)}
                ci=np.array([lookup[int(s)] for s in z['sample_id']]);li=np.array([llook[int(s)] for s in z['sample_id']])
                if not np.array_equal(z['native'],cache['candidate'][ci,:,:24].reshape(-1,20,12,2)) or not np.array_equal(z['gt'],labels['future'][li]) or not np.array_equal(z['groups'],cache['change_group'][ci]):raise ValueError('Saved native/GT/groups do not equal frozen sources')
                with np.load(frozen_metadata,allow_pickle=False) as md:
                    ml={int(s):i for i,s in enumerate(md['sample_id'])};mi=np.array([ml[int(s)] for s in z['sample_id']]);expected=np.array([str(s)+'::'+str(int(p)) for s,p in zip(md['sequence'][mi],md['pedestrian_id'][mi])])
                    if not np.array_equal(z['track'],expected):raise ValueError('Saved tracks do not equal frozen metadata')
                frozen_verified=True
        recomputed=independent_metrics(z)
    stored=json.loads(Path(result).read_text(encoding='utf-8-sig'));report=compare(stored.get('metrics',stored),recomputed,tolerance)
    after={str(p):sha(p) for p in paths}
    if before!=after:report.update(status='FAIL',mismatches=report['mismatches']+['INPUT_CHANGED_DURING_AUDIT'])
    input_hashes={Path(k).relative_to(root).as_posix():v for k,v in before.items()} if root is not None else before
    report.update(inputs_sha=input_hashes,script_sha=sha(__file__),tolerance=tolerance,independent_implementation=True,frozen_labels_verified=frozen_verified)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('prediction');p.add_argument('result');p.add_argument('--output',required=True);p.add_argument('--tolerance',type=float,default=2e-12);p.add_argument('--root');p.add_argument('--frozen-cache');p.add_argument('--frozen-labels');p.add_argument('--frozen-metadata');p.add_argument('--frozen-manifest');p.add_argument('--saved-only',action='store_true');a=p.parse_args()
    if not a.saved_only and not all([a.root,a.frozen_cache,a.frozen_labels,a.frozen_metadata,a.frozen_manifest]):p.error('Frozen inputs required; --saved-only is explicitly limited scope')
    report=audit_file(a.prediction,a.result,a.tolerance,frozen_cache=a.frozen_cache,frozen_labels=a.frozen_labels,frozen_metadata=a.frozen_metadata,frozen_manifest=a.frozen_manifest,root=Path(a.root) if a.root else None)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
