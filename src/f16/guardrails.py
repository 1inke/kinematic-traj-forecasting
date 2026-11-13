"""Guardrail measurements. Aggregation choices require preregistration.

This helper requires explicit per-seed G3/G4; it does not silently settle the
author's outstanding seed-aggregation choice. No historical verdicts are changed.
"""
def measure_guardrails(treatment,control,overall_CI95,*,total_parameters,K=20,budget_exception=False,seed_policy):
    if seed_policy!='all_seeds':raise ValueError('Explicit preregistered seed policy required')
    if not treatment or len(treatment)!=len(control):raise ValueError('Paired seeds required')
    g1=[];g3=[];g4=[];overall=[]
    for t,c in zip(treatment,control):
        tc=t['change'];cc=c['change'];g1.append(tc['motion_pass'])
        g3.append(all(tc['Q_winner_'+k]<=1.03*cc['Q_winner_'+k] for k in ['ADE','FDE']))
        g4.append(None if tc['lost_rate_native_good'] is None else tc['lost_rate_native_good']<=.1)
        overall.append(t['overall']['coverage']-c['overall']['coverage'])
    return dict(G1=all(g1),G2=sum(overall)/len(overall)>=-.01 and overall_CI95[0]>=-.03,G3=all(g3),G4=None if None in g4 else all(g4),G5=K==20 and (total_parameters<=305000 or budget_exception)),dict(G1=g1,G3=g3,G4=g4,overall_delta_by_seed=overall,seed_policy=seed_policy)
