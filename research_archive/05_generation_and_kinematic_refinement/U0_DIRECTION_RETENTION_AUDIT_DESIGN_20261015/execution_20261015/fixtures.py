import numpy as np
import common as c,direction_metrics as dm,retention_intervals as ri,independent_review as rev,attribution as at,cluster_bootstrap as bs

def geometry():
    p=np.zeros((8,20,12,2));q=p.copy();gt=np.zeros((8,12,2))
    p[1,...,0]=.15;q[1,...,0]=.1 # towardGT
    p[2,...,0]=.15;q[2,...,0]=.25 # wrongdirection andexit
    p[3,...,0]=.15;q[3,...,0]=-.25 # correctdirection overshoot
    q[4,...,0]=.1 # zeroerror derivative norm(r)
    p[5,:,6:,0]=.25;q[5]=p[5];q[5,:,-1,0]+=.1 # exactmax tie
    p[6,:,-1,0]=.4;q[6,:,-1,0]=.5 # exactFDEboundary immediatelyexits
    p[7,:,-1,0]=.2;q[7,:,-1,0]=.4 # exactFDEboundary atone,noexit
    d=dm.candidate(p,q,gt);di=rev.candidate_loop(p,q,gt);rev.arrays_equal(d,di);dm.check_identities(d)
    assert np.all(d['dJ'][0]==0) and np.all(d['uncertain'][0])
    assert np.all(d['kind'][1]==3) and np.all(d['kind'][2]==1) and np.all(d['kind'][3]==2)
    c.close(d['dpoint'][4],np.full_like(d['dpoint'][4],.1));c.close(d['dJ'][4],np.full_like(d['dJ'][4],.5))
    assert np.all(d['A0'][5]/.2==d['F0'][5]/.4) and np.all(d['dJ'][5]==np.maximum(d['dA'][5]/.2,d['dF'][5]/.4))
    assert np.all(d['near_tie'][5]) and np.all(d['kind'][5]==0)
    x=ri.exits(p,q,gt,d);xi=ri.exits(p,q,gt,di,True);rev.arrays_equal(x,xi)
    assert np.all(x['exit_hi'][6]<1e-12) and np.all(x['native_good_no_exit'][7]) and np.all(x['exit_lo'][7]==1)
    # Originalfailed candidate can be good in themiddle: never include in G0.
    pp=np.full((1,20,12,2),0.);pp[...,0]=.3;qq=pp.copy();qq[...,0]=-.3;dd=dm.candidate(pp,qq,np.zeros((1,12,2)));xx=ri.exits(pp,qq,np.zeros((1,12,2)),dd)
    assert not dd['good0'].any() and not xx['exited'].any() and np.isnan(xx['exit_lo']).all()
    # Different slots attain separate ADE/FDE: no completegood slot.
    pp=np.ones((1,20,12,2));pp[:]=0;pp[...,0]=1.;pp[0,0,:11,0]=0;pp[0,0,-1,0]=.5;pp[0,1,:,0]=.3;pp[0,1,-1,0]=0
    aa,ff,j,g=dm.quality(pp,np.zeros((1,12,2)));assert aa.min()<=.2 and ff.min()<=.4 and not g.any()
    # Near-tie band does notalter true branch derivative or quality.
    nearp=p[5:6].copy();nearp[:,:,-1,0]+=1e-12;nearq=nearp+.01;nd=dm.candidate(nearp,nearq,gt[:1]);assert nd['near_tie'].all() and nd['uncertain'].all()
    # Smooth mathematical directional finite differences, fixedtwoepsonly.
    smooth=np.full((1,20,12,2),.13);r=np.full_like(smooth,.002);zero=np.zeros((1,12,2));sd=dm.candidate(smooth,smooth+r,zero)
    fd=[]
    for eps in (1e-4,1e-5):
        j0=dm.quality(smooth,zero)[2];j1=dm.quality(smooth+eps*r,zero)[2];err=float(np.max(abs((j1-j0)/eps-sd['dJ'])));assert err<1e-8;fd.append(err)
    # all20 identical/counterdirected/multipleG0/g0zero; emptygroup summary.
    meta=dict(sample_id=np.arange(8),groups=np.zeros(8,int),track=np.array(['t'+str(i) for i in range(8)]),scene=np.array(['s']*8),sequence=np.array(['seq']*8),native=p,paths=q,gt=gt)
    _,w,s=rev.process(meta);assert s['change']['overall']['n']==0 and np.all(w['g0']>=0)
    for name,mask in c.masks(w['groups']):
        a=s[name]['accel_attribution']
        if a['n']:c.close(a['delta_mean'],sum(v['weighted_signed_contribution'] for v in a['events']))
    # Explicit g0=1/2/3 and conflictingcandidate directions.
    pp=np.full((3,20,12,2),1.);pp[:,:, :,1]=0
    for i in range(3):pp[i,:i+1,:,0]=.1
    qq=pp.copy();qq[:,::2,:,0]+=.03;qq[:,1::2,:,0]-=.03;dd=dm.candidate(pp,qq,np.zeros((3,12,2)));assert np.array_equal(dd['good0'].sum(1),[1,2,3])
    return dict(status='PASS',zero_error_zero_residual_endpoint_zero=True,exact_tie_max_right_derivative=True,near_tie_uncertain_not_quality_epsilon=True,wrong_direction_and_finite_overshoot=True,disjointminima_rejected=True,originalgood_exit_brackets80=True,initialfailed_midgood_not_G0=True,multipleG0_conflictingdirections_emptygroup=True,smooth_fixed_eps=[1e-4,1e-5],smooth_FD_absolute_errors=fd)

def bootstrap():
    jobs={};root=c.H/'preflight_smoke/bootstrap'
    for seed in c.SEEDS:
        for split in ('FIT','outer'):
            ss=(seed-42)*.01;isouter=split=='outer';ns=np.array([1,0,1,0,1,0],bool)
            w=dict(groups=np.full(6,2),track=np.array(['a','a','b','b','c','c']),sample_id=np.arange(6)+(100 if isouter else 0),C0=ns,event=np.array([2,3,1,0,1,0]),dJ_fixed=np.arange(6)*.01+ss+isouter*.1,deltaJ_fixed=np.arange(6)*.02+ss+isouter*.2,uncertain_fixed=np.zeros(6,bool))
            path=root/f'{seed}_{split}.npz';c.save(path,**w);jobs[f'integrated_acceleration/seed{seed}/{split}']=dict(window=str(path))
    path=root/'draws.npz';a=bs.compute(jobs,draw_path=path);b=bs.compute(jobs,True,draw_path=path);err=c.compare(a,b)
    assert all(r['valid_draws']==20000 for r in a['rows']);assert np.isnan(bs.ratio(np.zeros((2,8)))).all()
    # Actual allzero native-success denominator: reportinvalid, never resample.
    for row in jobs.values():
        w=c.npz(row['window']);w['C0'][:]=False;c.save(row['window'],**w)
    bad=bs.compute(jobs,draw_path=root/'invalid_draws.npz');assert bad['rows'][1]['valid_draws']==0 and not bad['rows'][1]['CI_usable_for_branch']
    return dict(status='PASS',independent_track_sums_same20kdraws_maxerror=err,invalid_denominator_valid_draws=bad['rows'][1]['valid_draws'],no_resampling=True,PCG64seed=20261015,family4=True)

def contracts(z,reference,row,source):
    c.check_input(z,reference,row)
    for field in ('sample_id','native'):
        bad=dict(z)
        if field=='sample_id':bad[field]=z[field].copy();bad[field][0]+=10000000
        else:bad[field]=z[field][:,np.roll(np.arange(20),1)].copy();assert not np.array_equal(bad[field],z[field])
        try:c.check_input(bad,reference,row)
        except AssertionError:pass
        else:raise AssertionError('BAD_ID_OR_SLOT_ACCEPTED')
    path=c.H/'preflight_smoke/contract.fixture';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    good=dict(status='PASS',source_digest=source,files={str(path):c.sha(path)})
    assert c.valid_done(good,source)
    for bad in (dict(good,source_digest='bad'),dict(good,files={str(path):'0'*64})):
        try:c.valid_done(bad,source)
        except AssertionError:pass
        else:raise AssertionError('BAD_SOURCE_OR_OUTPUT_ACCEPTED')
    return dict(status='PASS',completed_analysis_SHA_skip_compatible=True,bad_ID_slot_source_output_rejected=True,not_training_RNG_resume=True)
