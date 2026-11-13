from pathlib import Path
import sys,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def csvread(p):
    with Path(p).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    rows=read(HERE/'statistics/bootstrap_ci.json');out={}
    metrics=('candidate_pooled_turn_given_old_straight','old_straight_entry_turn_window_rate','history_invalid_fraction','entry_present_common','entry_onset_present_common','refined_present_common','UNDEFINED_OR_BOUNDARY','ENTRY_MODE_ABSENT','ENTRY_PRESENT_ONSET_MISSING','ENTRY_ONSET_PRESENT_SPEED_MISSING','REFINED_MODE_PRESENT_BUT_INACCURATE')
    out['main_statistics']=[r for r in rows if r['variant']=='a20_history3_s2' and r['metric'] in metrics and ((r['population'] in ('hold','inner') and r['group'] in GROUPS) or (r['population'] in ('train','val') and r['group']=='change'))]
    out['oracle_statistics']=[r for r in rows if r['variant']=='oracle' and r['group']=='change' and r['metric'] in ('Native_coverage','END_b02_coverage','END_b04_coverage','END_unbounded_coverage','TIME_grid_coverage','END_b04_rescue_of_failure','TIME_grid_rescue_of_failure')]
    out['sensitivity']=[r for r in csvread(HERE/'E3_gt_audit/mode_presence.csv') if r['population'] in ('hold','inner') and r['group']=='change']
    out['old668_transition']=[r for r in csvread(HERE/'E3_gt_audit/old_present_new_absent.csv') if r['population']=='hold' and r['variant']=='a20_history3_s2' and r['group']=='change']
    out['main_oracle']=[r for r in csvread(HERE/'E4_oracle/oracle_coverage.csv') if r['population'] in ('hold','inner') and r['group']=='change']
    out['main_motion']=[r for r in csvread(HERE/'E4_oracle/motion_audit.csv') if r['population'] in ('hold','inner') and r['group']=='change']
    out['overlap']=[r for r in csvread(HERE/'E4_oracle/oracle_overlap.csv') if r['population'] in ('hold','inner') and r['group']=='change']
    out['errors']=[r for r in csvread(HERE/'E4_error/mode_matched_error.csv') if r['population']=='hold' and r['group']=='change' and r['reference'] in ('kJ','kMode') and r['metric'] in ('FDE','ADE_first','ADE_middle','ADE_last','shape_distance','entry_heading_error','internal_heading_error','onset_time_diff','early_speed_diff','late_speed_diff') and r['state'] in ('success','refined_valid_failure')]
    extra=[]
    for split in SPLITS:
        x=xload(split);f=load(HERE/f'E3_gt_free/{split}/a20_history3_s2.npz');gt=load(HERE/f'E3_gt_audit/{split}/a20_history3_s2/gt_features.npz');q=load(HERE/f'E3_gt_audit/{split}/native_quality.npz')
        for group,mask in masks(x):
            oldstraight=gt['L0'][:,0]==2;entryturn=np.isin(gt['entry'][:,0],[1,2]);fail=mask&~q['native_good'];speed=gt['speed_boundary'][:,0];angle=gt['angle_boundary'][:,0]
            extra.append(dict(population=split,group=group,N=int(mask.sum()),GT_old_straight=int((mask&oldstraight).sum()),GT_old_straight_entry_turn=int((mask&oldstraight&entryturn).sum()),failure_angle_boundary=int((fail&angle).sum()),failure_speed_boundary=int((fail&speed).sum()),failure_both_boundary=int((fail&speed&angle).sum()),GT_onset_failure_counts=np.bincount(gt['onset'][:,0][fail],minlength=6).tolist(),native_onset_candidate_counts=np.bincount(f['onset'][mask].ravel(),minlength=6).tolist()))
    out['extra_crosswalk']=extra
    out['scene_main_change']=[r for r in csvread(HERE/'statistics/track_scene.csv') if r['scope']=='scene' and r['population'] in ('hold','inner') and r['group']=='change']
    out['source_digest']=read(HERE/'manifest.json')['digest'];out['source_count']=len(read(HERE/'manifest.json')['files'])
    write(HERE/'completion_review/SUMMARY.json',out)
    print(json.dumps(dict(main_hold=[r for r in out['main_statistics'] if r['population']=='hold'],oracle_hold=[r for r in out['oracle_statistics'] if r['population']=='hold'],extra_hold=[r for r in extra if r['population']=='hold'],scene=out['scene_main_change'],errors=[r for r in out['errors'] if r['reference']=='kJ']),ensure_ascii=False))
if __name__=='__main__':main()
