"""Independent group review with registered pre-subtraction float64 observation arithmetic."""
import attribution_core as c

if __name__=='__main__':
    p=c.ROOT/'review_group_supplement.py';src=p.read_text(encoding='utf-8')
    repairs={'all192_gain_cells=True':'all_gain_cells_checked=True',
        "past=z['past'][mask];vo=":"past=z['past'][mask].astype(np.float64);vo=",
        "'GROUP_REVIEW_EXTENSION.json'":"'GROUP_REVIEW_EXTENSION_v2.json'",
        "'completion_review/GROUP_SUPPLEMENT.json'":"'completion_review_v2/GROUP_SUPPLEMENT.json'"}
    for old,new in repairs.items():assert src.count(old)==1;src=src.replace(old,new)
    c.write_json(c.ROOT/'GROUP_REVIEW_V2_ADAPTER.json',dict(files={str(p):c.sha(p),str(c.ROOT/'review_group_v2_adapter.py'):c.sha(c.ROOT/'review_group_v2_adapter.py')},
        replacements=repairs,original_rule_float64_before_subtraction=True,no_tolerance_change=True))
    ns={'__name__':'group_review_v2_namespace','__file__':str(p)}
    exec(compile(src,str(p),'exec'),ns);ns['main']()
