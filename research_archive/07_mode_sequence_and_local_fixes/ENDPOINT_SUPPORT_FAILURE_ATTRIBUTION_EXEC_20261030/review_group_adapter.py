"""Correct a pre-execution receipt field label; saved arithmetic is identical."""
import attribution_core as c

if __name__=='__main__':
    p=c.ROOT/'review_group_supplement.py';src=p.read_text(encoding='utf-8')
    old='all192_gain_cells=True';new='all_gain_cells_checked=True';assert src.count(old)==1
    c.write_json(c.ROOT/'GROUP_REVIEW_RECEIPT_ADAPTER.json',dict(
        files={str(p):c.sha(p),str(c.ROOT/'review_group_adapter.py'):c.sha(c.ROOT/'review_group_adapter.py')},
        repair='Correct static receipt field name before execution; actual48 cells remains len(gains). No numerical function changes.',
        old=old,new=new,original_source_preserved=True))
    ns={'__name__':'group_review_receipt_namespace','__file__':str(p)}
    exec(compile(src.replace(old,new),str(p),'exec'),ns);ns['main']()
