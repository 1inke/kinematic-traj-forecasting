"""Finite synthesis of all existing U1 summaries, no worker/PID/model access."""
import common as c
import numpy as np

def main():
    jobs=c.read(c.U1/'JOBS.json');assert len(jobs)==24
    c.verify(c.read(c.U1/'source_manifest.json')['files']);rows=[]
    for key,row in jobs.items():
        c.valid_done(row,c.read(c.U1/'PRESERVATION.json')['source_digest']);s=c.read(row['summary'])
        for group,item in s.items():rows.append(dict(job=key,group=group,overall=item['overall'],thin_strata=[x for x in item['strata'] if x.get('thin')],ceiling=item['retention_ceiling'],accel=item['accel_attribution']))
    main=c.read(c.U1/'PRIMARY_COMPARISONS.json');sens=c.read(c.U1/'SENSITIVITY_COMPARISONS.json');branch=c.read(c.U1/'NEXT_BRANCH.json')
    coupled=[r for r in rows if r['job'].startswith('integrated') and r['group']=='change'];all20=[r['overall']['all20_dJ_mean'] for r in coupled if r['job'].endswith('/outer')]
    out=dict(status='REVIEWED_EXISTING_SAVED_EVIDENCE',jobs=24,group_rows=len(rows),source_outputs_SHA_verified=True,primary=main,sensitivity=sens,preliminary_flags=branch['flags'],all_group_records=rows,final_interpretation=dict(critical_original_candidate_direction_transfer_gap=True,native_success_retention_gap_supported=True,all20_outer_dJ_still_negative=bool(np.all(np.array(all20)<0)),not_all_candidate_directions_wrong=True,rescue_gap_CI_crosses_zero=True,majority_overshoot_not_supported=True,mostly_fragile_margin_not_supported=True,near_boundary_risk_present_but_thin=True,unchanged_rescue_retention_ceiling_not_new_training_bound=True),inline_DA2='All20 average directional benefit is preserved: fixed original J slot findings cannot be universalized. Inner positive and outer seed44 net rescue remain counterexamples.',inline_DA3='Retained-success gap does not establish unique objective cause; population/margin/G0 differences and finite optimization remain. GT-fixed slots/exit brackets are not deployable selectors. Same-assistant review, not external.',no_production_rerun_or_model=True)
    c.write('U1_EVIDENCE_REVIEW.json',out)

if __name__=='__main__':main()
