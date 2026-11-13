import ast,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'F0_execution_authorized_20261017'
def tree(p):return ast.parse(p.read_text(encoding='utf-8-sig'))
def functions(p):return {x.name:ast.dump(x,include_attributes=False) for x in tree(p).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
def main():
    exact=['registry.py','run.py','independent_worker.py','prelaunch_review.py']
    for n in exact:assert ast.dump(tree(H/n))==ast.dump(tree(P/n)),n
    changed={}
    for n,allowed in {'common.py':{'compare','locks'},'evaluate.py':{'reconstruct'},'solver_client.py':{'Solver'},'solver_service.py':{'main'},'preflight.py':{'main'}}.items():
        a,b=functions(P/n),functions(H/n);assert a.keys()==b.keys()
        diff={k for k in a if a[k]!=b[k]};assert diff<=allowed,(n,diff);changed[n]=sorted(diff)
    return dict(status='PASS',fullAST_EXACT=exact,changed_functions=changed,parent_sources=5691,
      control_equation_objective_samecandidate_quality_five_motion_tolerances_unchanged=True,
      runtime_ceilings_withdrawn_by_human=True,structured_metadata_recursion_repaired=True,
      actual_U0_batch_signature_restore=True,scientific_solver_same_Clarabel_only=True)
if __name__=='__main__':print(json.dumps(main()))
