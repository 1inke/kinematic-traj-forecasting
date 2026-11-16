"""Light descriptive summaries; no dead/promising inference from surviving seeds."""
from explore_common import *

REF64={'T1':{'overall':1.6621081422259625,'change':3.8541666666666665},
       'E3_self':{'overall':2.608878602987587,'change':4.583333333333333}}


def calibrate(epochs):
    base=validate_result(ROOT/'runs'/f'B0_E{epochs}_s42')
    second=validate_result(ROOT/'runs'/f'B0_E{epochs}_s43')
    noise={g:100*(base['dev'][g]['coverage']-second['dev'][g]['coverage']) for g in ('overall','change')}
    controls={}
    for arm,ref in REF64.items():
        result=validate_result(ROOT/'runs'/f'{arm}_E{epochs}_s42')
        differences={g:100*(result['dev'][g]['coverage']-base['dev'][g]['coverage']) for g in ref}
        reproduction=all(differences[g]>=.5*ref[g] for g in ref)
        distinguishable=all(differences[g]>abs(noise[g]) for g in ref)
        controls[arm]=dict(diff_pp=differences,half_reference_reproduced=reproduction,
                           exceeds_single_B0_pair_difference=distinguishable,
                           known_full_screen_success=False)
    accepted=all(v['half_reference_reproduced'] and v['exceeds_single_B0_pair_difference'] for v in controls.values())
    decision=dict(at=now(),epochs=epochs,usable_as_initial_screen=accepted,controls=controls,
                  B0_pair_difference_pp=noise,noise_estimate_is_reliable=False,
                  meaning='Heuristic reproduction of two known coverage signals, not proof of ranking or efficacy.')
    replace_json(ROOT/f'CALIBRATION_E{epochs}.json',decision)
    return decision


def summarize():
    import csv
    rows=[]
    for path in sorted((ROOT/'runs').glob('*/result.json')):
        r=validate_result(path.parent)
        d=r['dev'];pair=r.get('paired')
        rows.append(dict(arm=r['arm'],epochs=r['epochs'],seed=r['seed'],status=r['status'],
            minADE=d['overall']['minADE'],minFDE=d['overall']['minFDE'],
            coverage_pp=100*d['overall']['coverage'],change_coverage_pp=100*d['change']['coverage'],
            top1_coverage_pp=100*d['overall']['top1_good'],
            change_diff_pp=pair['change']['coverage_diff_pp'] if pair else None,
            lost_reference_good=pair['overall']['lost_fraction_of_reference_good'] if pair else None,
            acceleration_p95=d['motion_allK20']['acceleration']['p95'],
            jerk_p95=d['motion_allK20']['jerk']['p95'],wall_seconds=r['wall_seconds']))
    if rows:
        temp=ROOT/'LEDGER.csv.pending'
        with temp.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temp,ROOT/'LEDGER.csv')
    replace_json(ROOT/'EXPLORATION_RESULTS.json',dict(at=now(),scope='EXPLORATORY_ONLY',rows=rows))
    return rows

