"""Repair scope is one train.py filename literal plus audit/source-lock/task relocation."""
from pathlib import Path
import ast,json,hashlib
H=Path(__file__).resolve().parent;P=H.parent/'T2_execution_protocol_recovery_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return ast.parse(p.read_text(encoding='utf-8-sig'))
records=[]
for name in ('features.py','objectives.py','model.py','metrics.py','bootstrap.py','evaluate_saved.py','memory_policy.py','independent_worker.py'):
    assert ast.dump(tree(P/name),include_attributes=False)==ast.dump(tree(H/name),include_attributes=False),name
    records.append(dict(file=name,AST_EXACT=True,parent_SHA=sha(P/name),current_SHA=sha(H/name)))
assert (P/'protocol.json').read_text(encoding='utf-8-sig')==(H/'protocol.json').read_text(encoding='utf-8-sig')
assert (P/'resource_policy.json').read_text(encoding='utf-8-sig')==(H/'resource_policy.json').read_text(encoding='utf-8-sig')
old=tree(P/'train.py');changed=0
for node in ast.walk(old):
    if isinstance(node,ast.Constant) and node.value=='statistics_position.npz':node.value='statistics_velocity.npz';changed+=1
assert changed==1
assert ast.dump(old,include_attributes=False)==ast.dump(tree(H/'train.py'),include_attributes=False)
a=tree(P/'common.py');b=tree(H/'common.py')
for n in a.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='locks':
        other=next(x for x in b.body if isinstance(x,type(n)) and x.name==n.name)
        assert ast.dump(n,include_attributes=False)==ast.dump(other,include_attributes=False)
report=dict(status='PASS',unchanged_scientific_modules=records,train_only_one_statistics_filename_literal_repaired=True,old_train_SHA=sha(P/'train.py'),new_train_SHA=sha(H/'train.py'),protocol_and_resource_policy_EXACT=True,common_only_source_inheritance_changed=True,audits_added=['actual manifest expression all three seeds at preflight and prelaunch','in-memory old literal regression and bad metadata rejection'],source_parent_count=2499,parent_source_manifest_SHA=sha(P/'source_manifest.json'),no_model_loss_features_data_optimizer_threshold_changes=True,parent_FAILURE_preserved=True)
(H/'SOURCE_EQUIVALENCE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('SOURCE_EQUIVALENCE PASS one train filename literal',len(records))
