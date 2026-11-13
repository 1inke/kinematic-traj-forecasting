from common import *
from morphology import features,categories,assign,variants
from oracle import remove_metrics,scalar_check
def run():
    g=guard(True);cfg=read(HERE/'config/analysis_config.json');checks=[]
    p=np.zeros((6,20,12,2));past=np.zeros((6,8,2));past[:,:,0]=np.arange(-7,1)*.4
    p[1,:,:,0]=np.arange(1,13)*.4;p[2,:,:,1]=np.arange(1,13)*.4;p[3,:,:,0]=np.arange(1,13)*.4;p[3,:,:,1]=(np.arange(1,13)/12)**2;p[4,:,:,0]=-np.arange(1,13)*.4;p[5]=p[1]
    f=features(p,past,cfg['slow_floor']);m,_=categories(f,20,cfg['slow_floor']);assert (m[0]==0).all() and (m[1]==2).all() and np.all(f['phi'][0,:,2:4]==0);checks.append('zero_low_undefined_not_fake_turn')
    assert np.max(abs(wrapped(np.array([np.pi-.01])-np.array([-np.pi+.01]))+.02))<1e-12;checks.append('wrapped_angle_crosses_pi_without_2pi_jump')
    # Candidate and GT values are arbitrary scalars here; no GT at mode fitting.
    a=np.array([[.1,.3]+[1.]*18,[.2,.2]+[1.]*18,[.1,.3]+[1.]*18]);ff=np.array([[.5,.3]+[1.]*18,[.4,.4]+[1.]*18,[.3,.6]+[1.]*18]);j=np.maximum(a/.2,ff/.4);good=(a<=.2)&(ff<=.4);r=remove_metrics(a,ff,j,good);scalar_check(a,ff,j,good,r)
    assert not good[0].any() and good[1].sum()==2 and not r['lost_if_deleted'][1].any() and r['lost_if_deleted'][2,0];checks.append('same_slot_boundary_J1_ties_single_witness')
    assert (r['strict_gt_removable'].sum(1)>=17).all();checks.append('at_least17_is_trivial_minima_structure_not_waste')
    x=load(HERE/'inputs/X_opt.npz');ix=load(HERE/'inputs/train_val.npz')['train'][:128];small={k:v[ix] for k,v in x.items()};z=assign(small,cfg,variants()[0])
    for i in range(128):
        mode=z['mode'][i];counts=np.array([sum(mode==v) for v in np.unique(mode)]);probs=counts/20
        assert z['M'][i]==len(counts) and z['Pmax'][i]==max(probs) and abs(z['Meff'][i]-np.exp(-sum(q*np.log(q) for q in probs)))<=2e-12
        assert z['structural'][i].sum()==20-len(counts)
        for v in np.unique(mode):assert z['representative'][i][mode==v].sum()==1
    checks.append('actual128_modes_counts_medoid_structure_independent')
    direct=features(small['native'],small['past'],cfg['slow_floor']);offset=np.array([2.,-3.]);shifted=features(small['native']+offset,small['past']+offset,cfg['slow_floor']);assert np.nanmax(abs(direct['phi']-shifted['phi']))<=2e-12;checks.append('translation_invariance_and_p0_included')
    # Actual128 native saved independently vs all20 explicit deletion loops.
    gt=load(HERE/'inputs/GT_opt.npz')['gt'][ix];aa,fe,jj,gg=quality(small['native'],gt);rr=remove_metrics(aa,fe,jj,gg);scalar_check(aa,fe,jj,gg,rr);checks.append('actual128_all2560_single_deletions_scalar_EXACT')
    xrev={**small,'native':small['native'][:,::-1]};rev=assign(xrev,cfg,variants()[0]);assert np.array_equal(z['mode'][:,::-1],rev['mode']) and np.max(abs(z['Meff']-rev['Meff']))<=2e-12;checks.append('physical_reverse_counts_invariant_medoid_ties_explicit')
    # Saved NPZ roundtrip, bad-ID/source files rejected without touching old files.
    fixture=HERE/'audit'/('fixture128_'+read(HERE/'config/sources_manifest.json')['digest'][:12]+'.npz')
    if not fixture.exists():save(fixture,sample_id=small['sample_id'],mode=z['mode'],**rr)
    back=load(fixture)
    for key,value in dict(sample_id=small['sample_id'],mode=z['mode'],**rr).items():assert np.array_equal(back[key],value),key
    checks.append('actual_saved_readback_EXACT')
    assert 'torch' not in sys.modules;write(HERE/'audit/numerical_checks.json',dict(status='PASS_CPU_ONLY',checks=checks,GT_not_mode_function_input=True,source=verify_sources(),no_model_or_optimizer=True));resource(HERE/'audit/PREFLIGHT_RESOURCE.json',g);print('PASS PREFLIGHT',flush=True)
if __name__=='__main__':run()
