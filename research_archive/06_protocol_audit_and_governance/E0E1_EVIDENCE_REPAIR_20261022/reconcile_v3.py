"""Explicit legacy42 schema adapter, no missing field is fabricated as stored."""
def reconcile(metrics,old,tolerance=2e-6):
    rows=[];keys={'N':'n','good_count':'good','native_good_count':'native_good','coverage':'coverage','rescue':'rescue','lost':'lost','minADE':'ADE','minFDE':'FDE','motion_gate':'motion_pass'}
    for group in ['overall','stable','transition','change']:
        for oldkey,key in keys.items():
            provenance='STORED_OLD_FIELD'
            if oldkey not in old[group]:
                if oldkey=='native_good_count' and all(k in old[group] for k in ['good_count','rescue','lost']):
                    a=old[group]['good_count']-old[group]['rescue']+old[group]['lost'];provenance='DERIVED_IDENTITY_good_minus_rescue_plus_lost_NOT_STORED'
                else:raise ValueError('UNRECONCILABLE_MISSING_OLD_FIELD '+group+'/'+oldkey)
            else:a=old[group][oldkey]
            b=metrics[group][key]
            if oldkey=='motion_gate':b='PASS' if b else 'FAIL'
            discrete=oldkey in ['N','good_count','native_good_count','rescue','lost','motion_gate'];diff=0 if a==b else (abs(a-b) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None);ok=a==b if discrete else diff is not None and diff<=tolerance
            rows.append(dict(group=group,field=oldkey,old=a,replay=b,old_value_provenance=provenance,abs_diff=diff,pass_=bool(ok),tolerance=0 if discrete else tolerance))
    return dict(status='RECONCILED' if all(r['pass_'] for r in rows) else 'REPLAY_NOT_RECONCILED',fields=rows)
