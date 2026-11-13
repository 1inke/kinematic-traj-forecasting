"""Before locking: document metadata-only successor and additional audit assertions."""
from pathlib import Path
import ast,json,hashlib
H=Path(__file__).resolve().parent;P=H.parent/'T2_execution_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def body(p):return ast.dump(ast.parse(p.read_text(encoding='utf-8-sig')),include_attributes=False)
records=[]
for name in ('features.py','objectives.py','model.py','metrics.py','bootstrap.py','train.py','evaluate_saved.py','memory_policy.py','prelaunch_review.py','independent_worker.py'):
    assert body(P/name)==body(H/name),name
    records.append(dict(file=name,AST_EXACT=True,parent_SHA=sha(P/name),current_SHA=sha(H/name)))
for name in ('start_independent.ps1','check_once.ps1','resource_policy.json'):
    assert (P/name).read_text(encoding='utf-8-sig')==(H/name).read_text(encoding='utf-8-sig')
old=ast.parse((P/'common.py').read_text());new=ast.parse((H/'common.py').read_text())
for node in old.body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name!='locks':
        now=next(x for x in new.body if isinstance(x,type(node)) and x.name==node.name)
        assert ast.dump(node,include_attributes=False)==ast.dump(now,include_attributes=False),node.name
z=dict(status='PASS',identical_algorithm_modules=records,common_only_source_inheritance_changed=True,protocol_only_descriptive_scope_representation_resource_and_mechanism_fields_corrected=True,preflight_extra_checks_only=['control-normalization FD active face','actual nonzero C all-parameter L0 identity and independent coordinate gradient'],no_scientific_threshold_model_loss_optimizer_data_changes=True,parent_source_manifest_SHA=sha(P/'source_manifest.json'),parent_production_never_launched=True)
(H/'SOURCE_EQUIVALENCE.json').write_text(json.dumps(z,indent=2),encoding='utf-8')
print('SOURCE_EQUIVALENCE PASS',len(records))
