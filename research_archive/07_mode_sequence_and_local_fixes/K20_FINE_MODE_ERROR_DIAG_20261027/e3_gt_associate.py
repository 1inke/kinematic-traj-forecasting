from common import *
from fine_features import *
def main():
    verify_done(HERE/'E3_gt_free');g=guard(True);rows=[];transfer=[]
    for split in SPLITS:
        x=xload(split);y=yload(split);assert np.array_equal(x['sample_id'],y['sample_id'])
        native=load(OLD/f'E2_oracle_coverage/{split}/candidate_gt_metrics.npz')
        a,f,j,good=quality(x['native'],y['gt'])
        for key,value in [('ADE',a),('FDE',f),('J',j)]:assert np.array_equal(native[key],value),key
        assert np.array_equal(native['good'],good)
        save(HERE/f'E3_gt_audit/{split}/native_quality.npz',sample_id=x['sample_id'],ADE=a,FDE=f,J=j,good=good,native_good=good.any(1))
        old=load(OLD/f'E2_oracle_coverage/{split}/coarse20/slot_removal_oracle.npz')
        for name,angle,hkind,sustain in variants():
            ff=load(HERE/f'E3_gt_free/{split}/{name}.npz');gf=extract(y['gt'][:,None],x['past'],angle,hkind,sustain);assoc=associate(ff,gf)
            save(HERE/f'E3_gt_audit/{split}/{name}/gt_features.npz',sample_id=x['sample_id'],**gf)
            save(HERE/f'E3_gt_audit/{split}/{name}/presence.npz',sample_id=x['sample_id'],**assoc)
            for group,mask in masks(x):
                fail=mask&~good.any(1);common=fail&assoc['common_valid'];r=assoc['reason']
                rr=dict(population=split,variant=name,group=group,N=int(mask.sum()),native_fail=int(fail.sum()),GT_common_valid=int(common.sum()),GT_invalid=int((fail&assoc['GT_invalid']).sum()),GT_boundary=int((fail&assoc['GT_boundary']).sum()),partial=int((fail&assoc['partial_valid']).sum()))
                rr.update({label:int((fail&(r==k)).sum()) for k,label in enumerate(REASONS)});assert sum(rr[label] for label in REASONS)==rr['native_fail']
                for label,key in [('entry','entry_match'),('onset','onset_match'),('speed','speed_match'),('entry_onset','entry_onset_match'),('refined','full_match')]:
                    rr[label+'_all_failure_present']=int((fail&assoc[key].any(1)).sum());rr[label+'_common_present']=int((common&assoc[key].any(1)).sum())
                rows.append(rr)
                for label,base in [('all_native_failure',fail),('old_present_inaccurate',fail&(old['failure_reason']==1))]:
                    tr=dict(population=split,variant=name,group=group,old_condition=label,denominator=int(base.sum()))
                    tr.update({z:int((base&(r==k)).sum()) for k,z in enumerate(REASONS)})
                    tr['old_L0_present_raw']=int((base&old['GT_mode_present']).sum());transfer.append(tr)
            mark('E3_GT_ASSOCIATION',population=split,variant=name);g.check()
    table(HERE/'E3_gt_audit/mode_presence.csv',rows);table(HERE/'E3_gt_audit/old_present_new_absent.csv',transfer)
    resource(HERE/'E3_gt_audit/RESOURCE.json',g);finish(HERE/'E3_gt_audit',all_quality_EXACT=True,all_mutually_exclusive_denominators_conserved=True)
if __name__=='__main__':main()
