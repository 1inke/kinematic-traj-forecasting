from common import *
from fine_features import *
def main():
    # Enforce GT separation at actual filesystem access, not only by convention.
    access=[]
    def hook(event,args):
        if event=='open':
            try:p=Path(args[0])
            except (TypeError,ValueError):return
            if p.suffix=='.npz':
                assert not p.name.startswith('GT_') and 'E2_oracle_coverage' not in p.parts,'GT access prohibited in E3a'
                access.append(str(p))
    sys.addaudithook(hook)
    g=guard(True);cross=[];windows=[]
    for split in SPLITS:
        x=xload(split)
        for name,a,h,n in variants():
            f=extract(x['native'],x['past'],a,h,n)
            oldname={20:'coarse20',15:'coarse15',30:'coarse30'}[a]
            old=load(OLD/f'E1_mode_distribution/{split}/{oldname}/modes.npz',('mode',))['mode']
            assert np.array_equal(f['L0'],old),name
            save(HERE/f'E3_gt_free/{split}/{name}.npz',sample_id=x['sample_id'],**f)
            for group,mask in masks(x):
                for l0 in range(5):
                    base=(f['L0']==l0)&mask[:,None]
                    for level,labels,num in [('entry',f['entry'],4),('onset',f['onset'],6)]:
                        for label in range(num):
                            cross.append(dict(population=split,variant=name,group=group,N=int(mask.sum()),tracks=len(np.unique(x['track_key'][mask])),L0=l0,level=level,new_label=label,candidate_count=int((base&(labels==label)).sum()),old_L0_candidate_den=int(base.sum()),window_any=int((base&(labels==label)).any(1)[mask].sum())))
            for i,id_ in enumerate(x['sample_id']):
                windows.append(dict(population=split,variant=name,sample_id=int(id_),old_straight=int((f['L0'][i]==2).sum()),old_straight_entry_turn=int(((f['L0'][i]==2)&np.isin(f['entry'][i],[1,2])).sum()),any_old_straight_entry_turn=bool(((f['L0'][i]==2)&np.isin(f['entry'][i],[1,2])).any()),entry_counts=np.bincount(f['entry'][i],minlength=4).tolist(),onset_counts=np.bincount(f['onset'][i],minlength=6).tolist(),history_valid=bool(f['history_heading_valid'][i]),future_low=int(f['future_low'][i].sum()),entry_abs_median_deg=finite(np.nanmedian(abs(f['entry_angle'][i]))*180/np.pi) if f['entry_valid'][i].any() else None))
            mark('E3_GT_FREE',population=split,variant=name);g.check()
    table(HERE/'E3_gt_free/old_vs_new_mode.csv',cross);table(HERE/'E3_gt_free/window_summary.csv',windows)
    write(HERE/'E3_gt_free/ACCESS.json',dict(status='PASS_ACTUAL_GT_ACCESS_DENIED',NPZ_files=sorted(set(access)),GT_read=False,no_GT_feature_or_threshold_fitting=True))
    resource(HERE/'E3_gt_free/RESOURCE.json',g);finish(HERE/'E3_gt_free',old_L0_ALL_EXACT=True,no_GT_read=True)
    write(HERE/'E3_gt_free/E3_LOCK.json',dict(at=now(),source=read(HERE/'manifest.json')['digest'],config_SHA=sha(HERE/'configs/locked_protocol.json'),done_SHA=sha(HERE/'E3_gt_free/done.json'),GT_association_started=False))
if __name__=='__main__':main()
