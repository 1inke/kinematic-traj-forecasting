"""Eight-arm CPU endpoint probe, never a full-trajectory generator."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from prior import ARMS,Record,EndpointPrior,bucket,modes
from f16.resource import Guard

def decision(nll,opp,ci,mode):
    p=float(nll['PRIOR_'+mode]);base=float(nll['M0_motion_iso'])
    denom=base-p;gain=denom/base
    adv=(nll['SHUF_'+mode]-p)/denom if denom>0 else None
    gates=dict(B1=gain>.03 and p<nll['M1_heading_global'] and ci.get('status')=='OK' and ci['CI95'][1]<0,
        B2=adv is not None and adv>.5 and p<nll['XSCENE_'+mode],
        B3=opp['PRIOR_'+mode] is not None and opp['PRIOR_'+mode]>.10 and opp['PRIOR_'+mode]>opp['SHUF_'+mode])
    return dict(passed=all(gates.values()),gates=gates,relative_gain=gain,advantage_ratio=adv,
        no_gain_over_M0=denom<=0,paired_track_CI=ci)

def run(records_file,out,estimate):
    out=owned_output(out,'direction_B_scene_prior')
    if out.exists(): raise FileExistsError('New probe directory required; preserve old evidence')
    out.mkdir(parents=True)
    records_sha_before=sha256(records_file)
    guard=Guard(estimate);guard.check('startup');lock_sources(out/'source_manifest.json')
    data=load_data();bundle=read_json(records_file);rows=bundle['records']
    prov=bundle['provenance']
    if prov['adapter_source_SHA']!=sha256(Path(__file__).with_name('prepare_records.py')):
        raise ValueError('METADATA_ADAPTER_SOURCE_MISMATCH')
    if prov['raw_sha256']!=sha256(prov['raw']) or prov['metadata_sha256']!=sha256(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'):
        raise ValueError('WORLD_OR_TIME_SOURCE_CHANGED')
    if len({r['sample_id'] for r in rows})!=20765: raise ValueError('RECORD_ID_POPULATION')
    byid={r['sample_id']:r for r in rows}
    for split,d in data.items():
        assert set(r['sample_id'] for r in rows if r['split']==split)==set(map(int,d['ids']))
        for i,sid in enumerate(d['ids']):
            r=byid[int(sid)];assert r['track']==d['track'][i]
            np.testing.assert_array_equal(r['endpoint'],d['gt'][i,-1])
    # Only FIT contributes historical endpoints, including causal histories.
    bank=[]
    for row in rows:
        if row['split']=='FIT':
            r={k:v for k,v in row.items() if k!='split'};r['cell']=tuple(r['cell']);r['endpoint']=tuple(r['endpoint']);bank.append(Record(**r))
    prior=EndpointPrior(bank,data['FIT']['ids']);d=data['inner'];ne=errors(d['native'],d['gt'])
    counts=np.zeros((8,len(d['ids']),257),np.int32);nll=np.full((8,len(d['ids'])),np.nan)
    opportunity=np.zeros((8,len(d['ids'])),bool);valid=np.ones_like(opportunity)
    support=np.zeros_like(opportunity,dtype=np.int32);levels=np.empty(opportunity.shape,dtype='<U32')
    redundancy=np.full((8,len(d['ids']),4),np.nan);cold=np.zeros_like(opportunity)
    # Bounded packed membership preserves every source ID without building a
    # many-million-element Python list of per-query provenance dictionaries.
    fit_ids=data['FIT']['ids'];fit_lookup={int(v):j for j,v in enumerate(fit_ids)}
    membership=np.zeros((8,len(d['ids']),(len(fit_ids)+7)//8),np.uint8)
    for i,sid in enumerate(d['ids']):
        if i%128==0: guard.check()
        q=byid[int(sid)];api={k:q[k] for k in ('sample_id','track','sequence','scene','t0','cell','heading','verified_time')}
        for j,arm in enumerate(ARMS):
            a=prior.query(arm=arm,**api);levels[j,i]=a['level'];cold[j,i]=a['cold_start']
            if a.get('status')=='time_unverifiable': valid[j,i]=False;continue
            assert q['track'] not in a['source_tracks']
            if arm in ('PRIOR_causal','SHUF_causal') and not a['cold_start']:
                assert all(end<=q['t0'] for end in a['source_end_indices'])
            c=a['counts'];counts[j,i]=c;support[j,i]=a['support']
            nll[j,i]=-np.log(a['probability'][bucket(d['gt'][i,-1])])
            centers=modes(c);opportunity[j,i]=np.min(np.linalg.norm(centers-d['gt'][i,-1],axis=-1))<=.4
            redundancy[j,i]=np.linalg.norm(centers[:,None]-d['native'][i,None,:,-1],axis=-1).min(1)
            chosen=np.zeros(len(fit_ids),np.uint8)
            chosen[[fit_lookup[int(v)] for v in a['source_ids']]]=1
            membership[j,i]=np.packbits(chosen,bitorder='little')
    # Each causal comparison uses the SAME verified-time query population in all
    # eight arms. Static full497 and causal-valid subset are reported separately.
    summaries={};decisions={};change=d['groups']==2;miss=~ne['coverage']
    for mode in ('static','causal'):
        population=change.copy()
        if mode=='causal': population &= valid[ARMS.index('PRIOR_causal')]
        means={arm:float(nll[j,population&valid[j]].mean()) if (population&valid[j]).any() else None for j,arm in enumerate(ARMS)}
        opp={arm:float(opportunity[j,population&miss&valid[j]].mean()) if (population&miss&valid[j]).any() else None for j,arm in enumerate(ARMS)}
        pidx=ARMS.index('PRIOR_'+mode);ci=paired_ci(nll[pidx,population]-nll[0,population],d['track'][population]) if population.any() else {'status':'EMPTY'}
        decisions[mode]=decision(means,opp,ci,mode) if population.any() else dict(passed=False,reason='NO_VERIFIABLE_TIME')
        if mode=='causal' and int(population.sum())!=497:
            decisions[mode].update(passed=False,reason='FULL_PRIMARY_TIME_UNVERIFIABLE_SUBSET_DESCRIPTIVE_ONLY')
        summaries[mode]=dict(n=int(population.sum()),miss=int((population&miss).sum()),NLL=means,Opp4=opp)
    assert np.array_equal(counts[4],counts[7])
    np.savez_compressed(out/'probe_arrays.npz',sample_id=d['ids'],arms=np.array(ARMS),counts=counts,
        NLL=nll,opportunity=opportunity,valid=valid,levels=levels,support=support,cold_start=cold,
        endpoint_redundancy=redundancy,FIT_source_registry=fit_ids,packed_query_source_membership=membership)
    write_json(out/'QUERY_SOURCES.json',dict(format='np.unpackbits(bits,bitorder=little)[:len(FIT_source_registry)]',
        arrays='probe_arrays.npz',target_track_purge_all_levels=True,source_library='FIT only'))
    # Independent slot loop checks saved 257-bin arithmetic, not another probe.
    with np.load(out/'probe_arrays.npz') as saved:
        assert np.array_equal(saved['sample_id'],d['ids'])
        for j in range(8):
            for i in np.flatnonzero(saved['valid'][j]):
                c=saved['counts'][j,i];pr=(c+.10)/(sum(map(int,c))+257*.10)
                expected=-np.log(pr[bucket(d['gt'][i,-1])])
                assert abs(expected-saved['NLL'][j,i])<=2e-12
                assert int(c.sum())==int(saved['support'][j,i])
    write_json(out/'ARITHMETIC_AUDIT.json',dict(status='PASS_SAVED_BUCKET_ARITHMETIC',
        counts_probability_NLL=True,source_track_and_causal_checks=True,
        independent_second_eligibility_implementation=False))
    conclusion=('ALLOW_CONDITIONAL_GENERATOR_DESIGN_ONLY' if decisions['causal']['passed'] else
        'STATIC_ONLY_SIGNAL_CAUSAL_INSUFFICIENT' if decisions['static']['passed'] else 'SCENE_HISTOGRAM_PRIOR_FALSIFIED')
    groups={}
    for j,arm in enumerate(ARMS):
        groups[arm]={}
        for g,num in [('overall',None),('stable',0),('transition',1),('change',2)]:
            original=np.ones(len(d['ids']),bool) if num is None else d['groups']==num
            take=original&valid[j];missed=take&miss
            labels,ns=np.unique(levels[j,take],return_counts=True)
            groups[arm][g]=dict(n_original=int(original.sum()),n_valid=int(take.sum()),
                n_native_miss=int(missed.sum()),NLL=float(nll[j,take].mean()) if take.any() else None,
                Opp4=float(opportunity[j,missed].mean()) if missed.any() else None,
                cold_start=int(cold[j,take].sum()),levels=dict(zip(labels,ns)))
    guard.check()
    if sha256(records_file)!=records_sha_before:raise ValueError('RECORDS_CHANGED_DURING_PROBE')
    write_json(out/'RESULT.json',dict(status='COMPLETED',decisions=decisions,populations=summaries,
        records_SHA=sha256(records_file),arrays_SHA=sha256(out/'probe_arrays.npz'),source_digest=verify_sources(out/'source_manifest.json'),
        conclusion=conclusion,all_groups=groups,causal_time_unverifiable_IDs=d['ids'][~valid[5]],
        endpoint_opportunity_is_not_full_path_coverage=True))
    write_json(out/'RESOURCE.json',dict(estimate=estimate,factor=1.3,observations=guard.observations))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--records',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--estimated-peak-gib',type=float,required=True)
    a=ap.parse_args();run(a.records,a.output,int(a.estimated_peak_gib*2**30))
