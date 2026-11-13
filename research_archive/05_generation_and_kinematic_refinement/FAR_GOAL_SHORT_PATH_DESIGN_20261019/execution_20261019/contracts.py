import common as c
from pathlib import Path
import numpy as np

def source_files():
    prior=c.read(c.R/'MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017/P1_execution_final_20261018/source_manifest.json')
    files=prior['files'].copy();c.verify(files)
    h0=c.read('H0_RESULT.json');assert h0['status']=='PASS_H0_ALLOW_H1_PREFLIGHT';c.verify(h0['source_files']);c.verify(h0['output_sha'])
    add=list(c.S.glob('*.md'))+list(c.H.glob('*.py'))+list(c.H.glob('*.ps1'))
    add += [c.H/k for k in ('README.md','protocol.json','MEMORY_ESTIMATE.json','shared_statistics.npz','STATISTICS_CONTRACT.json','H0_RESULT.json')]
    add += [Path(p) for p in h0['source_files']]+[Path(p) for p in h0['output_sha']]
    add += [p for p in (c.H/'failures').rglob('*') if p.is_file()]
    for p in add:
        value=c.sha(p);key=str(p)
        if key in files:assert files[key]==value,('INHERITED_DIGEST_NOT_OVERWRITTEN',key)
        else:files[key]=value
    return files,dict(inherited_P1_sources=len(prior['files']),H0_outputs=len(h0['output_sha']),not_all_historical_files=True)

def check_done(done,source,contract=None):
    assert done['status']=='COMPLETE' and done['source_digest']==source,'BAD_DONE_SOURCE'
    if contract is not None:assert done['contract']==contract,'BAD_DONE_CONTRACT'
    c.verify(done['files']);return True

def bad_contract_fixtures():
    folder=c.H/'preflight_contract';folder.mkdir(exist_ok=True)
    a=folder/'payload.txt';a.write_text('saved-contract-fixture',encoding='utf8')
    done=dict(status='COMPLETE',source_digest='expected',contract={'H':20},files={str(a):c.sha(a)})
    assert check_done(done,'expected',{'H':20})
    rejected=[]
    for kind,dd,ss,cc in [('source',done,'bad',{'H':20}),('representation',done,'expected',{'H':12}),('output',dict(done,files={str(a):'0'*64}),'expected',{'H':20})]:
        try:check_done(dd,ss,cc)
        except AssertionError:rejected.append(kind)
        else:raise AssertionError('BAD_CONTRACT_ACCEPTED '+kind)
    return dict(status='PASS',completed_SHA_skip=True,rejected=rejected)

def h0_saved_audit(d):
    h0=c.read('H0_RESULT.json');c.verify(h0['source_files']);c.verify(h0['output_sha']);h=c.npz(c.H/'H0_labels.npz')
    assert np.array_equal(d['ids'],h['sample_id']);md=d['meta'];fit=d['split']['FIT'];raw=np.load(c.INPUTS['raw'],mmap_mode='r')
    # Independent scalar rotation, reproduces old20; no new model or data.
    maxerr=0.
    for i in range(len(d['ids'])):
        if i%512==0:c.guard()
        world=np.asarray(raw[int(d['ids'][i]),0,:20,-2:],np.float32);vec=world[7]-world[4];norm=float(np.sqrt(np.sum(vec*vec)))
        u=vec/max(norm,1e-8) if norm>.03 else np.array([1.,0.],np.float32);dx=world[:,0]-world[7,0];dy=world[:,1]-world[7,1]
        xy=np.array([dx*u[0]+dy*u[1],-dx*u[1]+dy*u[0]]).T
        maxerr=max(maxerr,float(abs(xy[:8]-d['past'][i]).max()),float(abs(xy[8:]-d['gt'][i]).max()))
        assert not h['m20'][i] or (h['continuous20'][i] and np.isfinite(h['y20'][i]).all())
    assert maxerr<=2e-5
    held=np.concatenate([d['split'][k] for k in ('outer','inner','original_CAL')]);extra_cross=[]
    for i in fit:
        if not h['m20'][i]:continue
        same=held[d['track'][held]==d['track'][i]]
        for j in same:
            assert h['t0'][j]>=0,'VALID_FIT_CROSSES_UNKNOWN_HELD_OWNER'
            if h['t0'][i]+13<=h['t0'][j]+12 and h['t0'][j]-7<=h['t0'][i]+20:extra_cross.append((int(i),int(j)))
    assert not extra_cross
    return dict(status='PASS',old8plus12_reconstruction_max_error=maxerr,valid_FIT_new_held_overlap=0,all_ids=28791,training=18909,primary_outer=1035,actual_output_SHA_PASS=True)
