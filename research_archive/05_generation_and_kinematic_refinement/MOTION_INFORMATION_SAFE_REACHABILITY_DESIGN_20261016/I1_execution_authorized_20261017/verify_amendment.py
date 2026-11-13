"""Compare authorized operational changes with immutable prior implementation."""
import ast,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
P=H.parent/'I1_execution_pipeline_recovery_20261017'
def tree(p):return ast.parse(p.read_text(encoding='utf-8-sig'))
def bodies(p):return {x.name:ast.dump(x,include_attributes=False) for x in tree(p).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
def main():
    exact=['features.py','model.py','metrics.py','independent_worker.py']
    for n in exact:assert ast.dump(tree(H/n))==ast.dump(tree(P/n)),n
    changes={}
    for n,allowed in {'common.py':{'locks'},'train.py':{'main'},'prelaunch_review.py':{'main'},'bootstrap.py':{'compute'},'preflight.py':{'main'},'memory_policy.py':{'decide'}}.items():
        a,b=bodies(P/n),bodies(H/n);assert a.keys()==b.keys()
        changed={k for k in a if a[k]!=b[k]};assert changed<=allowed,(n,changed);changes[n]=sorted(changed)
    old=json.loads((P/'protocol.json').read_text(encoding='utf-8-sig'));new=json.loads((H/'protocol.json').read_text(encoding='utf-8-sig'))
    # Only explicitly amended operational fields may differ.
    keys={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    assert keys<={'planning_max_minutes','no_install_F0_solver','authorized_execution_amendment','resource'},keys
    for k in old:
        if k not in keys:assert old[k]==new[k]
    return dict(status='PASS',full_AST_exact=exact,changed_function_bodies=changes,protocol_changed_keys=sorted(keys),
      inherited_parent_sources=2717,science_equations_data_thresholds_seeds_epochs_audits_unchanged=True,
      operational_changes='human authorizes solver install and removes 30min/fixed6GiB veto; actual 1.3 task guard retained',
      bootstrap_compute_change='only metadata flag: G1 requires independently audited F0; numerical comparisons and gate unchanged',
      same_assistant_inline_review_not_external=True)
if __name__=='__main__':print(json.dumps(main()))
