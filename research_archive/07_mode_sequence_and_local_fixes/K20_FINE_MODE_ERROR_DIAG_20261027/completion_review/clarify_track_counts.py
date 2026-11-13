from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def main():
    rows=read(HERE/'statistics/bootstrap_ci.json');reviewed=[];cache={}
    for pop in ('opt','train','val','hold','inner'):
        split='opt' if pop in ('train','val') else pop;x=xload(split)
        if pop in ('train','val'):
            sel=load(OLD/'inputs/train_val.npz')[pop];x={k:v[sel] for k,v in x.items()}
        cache[pop]=(len(np.unique(x['track_key'])),{g:len(np.unique(x['track_key'][m])) for g,m in masks(x)})
    for r in rows:
        row=r.copy();row['original_track_count_field']=row.pop('track_count');row['bootstrap_frame_track_count']=cache[r['population']][0];row['group_track_count']=cache[r['population']][1][r['group']];reviewed.append(row)
    write(HERE/'statistics/bootstrap_ci_REVIEWED.json',reviewed);table(HERE/'statistics/groups_REVIEWED.csv',reviewed)
    write(HERE/'completion_review/TRACK_COUNT_CLARIFICATION.json',dict(status='NEW_EXPLICIT_TRACK_FIELDS_OLD_PRESERVED',issue='candidate_pooled_turn_given_old_straight track_count stored full-pop resampling frame; other records stored group union',resolution='explicit full-pop bootstrap_frame_track_count and actual group_track_count; original field retained with original_ prefix, no arithmetic/CI change',hold_change=dict(bootstrap_frame_track_count=139,group_track_count=74),inner_change=dict(bootstrap_frame_track_count=76,group_track_count=35)))
if __name__=='__main__':main()
