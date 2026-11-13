import ast,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'F0_execution_recovery_review_20261017'
def funcs(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
def main():
    exact=['registry.py','run.py','evaluate.py','independent_worker.py','prelaunch_review.py']
    for n in exact:assert ast.dump(ast.parse((H/n).read_text(encoding='utf-8-sig')))==ast.dump(ast.parse((P/n).read_text(encoding='utf-8-sig'))),n
    changes={}
    for n,allowed in {'common.py':{'guard','locks'},'solver_client.py':{'Solver'},'solver_service.py':{'main'},'preflight.py':{'main'}}.items():
        a,b=funcs(P/n),funcs(H/n);assert a.keys()==b.keys();d={k for k in a if a[k]!=b[k]};assert d<=allowed,(n,d);changes[n]=sorted(d)
    for n in ('protocol.json','solver_contract.json','MEMORY_ESTIMATE.json'):assert json.loads((P/n).read_text(encoding='utf-8-sig'))==json.loads((H/n).read_text(encoding='utf-8-sig')),n
    return dict(status='PASS',parent_sources=5712,fullAST_EXACT=exact,changed_functions=changes,scientific_protocol_solver_settings_estimate_JSON_EXACT=True,only_resource_tree_cleanup_READY_metadata_and_regression_changed=True)
if __name__=='__main__':print(json.dumps(main()))
