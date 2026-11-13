"""Full second implementation rerun over corrected junction outputs; v1 retained."""
import attribution_core as c

if __name__=='__main__':
    p=c.ROOT/'review_saved.py';src=p.read_text(encoding='utf-8')
    repairs={
        "hm=ut['H_main_executed_gap'].mean(1).reshape(128,20).mean(1);hs=ut['hash_main_executed_gap'].mean(1).reshape(128,20).mean(1)":
        "hm=np.nanmean(ut['H_main_executed_gap'],axis=1).reshape(128,20).mean(1);hs=np.nanmean(ut['hash_main_executed_gap'],axis=1).reshape(128,20).mean(1)",
        'def physics(p,past):\n':'def physics(p,past):\n    past=np.asarray(past,dtype=np.float64)\n',
        "vo0=(z['past'][i,-1]-z['past'][i,-2])/.4":"vo0=(z['past'][i,-1].astype(float)-z['past'][i,-2].astype(float))/.4",
        "'counterfactual_metrics.npz'":"'counterfactual_metrics_v2.npz'",
        "out=c.ROOT/'completion_review'":"out=c.ROOT/'completion_review_v2'"}
    for old,new in repairs.items():assert src.count(old)==1;src=src.replace(old,new)
    c.write_json(c.ROOT/'REVIEW_V2_EXTENSION.json',dict(status='EXPLICIT_COMPLETE_REVIEW_PRECISION_ADAPTER',
        files={str(p):c.sha(p),str(c.ROOT/'review_saved_v2_adapter.py'):c.sha(c.ROOT/'review_saved_v2_adapter.py')},
        exact_replacements=repairs,no_tolerance_or_scientific_rule_change=True,all_original_sources_preserved=True))
    ns={'__name__':'saved_review_v2_namespace','__file__':str(p)}
    exec(compile(src,str(p),'exec'),ns);ns['main']()
