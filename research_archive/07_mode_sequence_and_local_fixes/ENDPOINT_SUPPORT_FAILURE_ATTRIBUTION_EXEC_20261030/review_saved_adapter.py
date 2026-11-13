"""Narrow missing-neighbor review adapter; original locked reviewer remains intact."""
from pathlib import Path
import attribution_core as c

if __name__=='__main__':
    path=c.ROOT/'review_saved.py';text=path.read_text(encoding='utf-8')
    old="hm=ut['H_main_executed_gap'].mean(1).reshape(128,20).mean(1);hs=ut['hash_main_executed_gap'].mean(1).reshape(128,20).mean(1)"
    new="hm=np.nanmean(ut['H_main_executed_gap'],axis=1).reshape(128,20).mean(1);hs=np.nanmean(ut['hash_main_executed_gap'],axis=1).reshape(128,20).mean(1)"
    assert text.count(old)==1
    c.write_json(c.ROOT/'REVIEW_ADAPTER_EXTENSION.json',dict(status='EXPLICIT_REVIEW_MISSING_NEIGHBOR_ADAPTER',
        files={str(path):c.sha(path),str(Path(__file__).resolve()):c.sha(__file__)},
        detected_before_reviewer_execution=True,
        issue='60 candidate queries have only2 same-sequence reference tracks. Plain mean in second reviewer would discard their valid2-neighbor mean.',
        repair='np.nanmean for missing padded neighbor ranks only; production already divides by actual finite count; no scientific exclusion or modified outputs.',
        exact_one_string_replacement=dict(old=old,new=new),old_source_preserved=True))
    ns={'__name__':'saved_review_adapter_namespace','__file__':str(path)}
    exec(compile(text.replace(old,new),str(path),'exec'),ns)
    ns['main']()
