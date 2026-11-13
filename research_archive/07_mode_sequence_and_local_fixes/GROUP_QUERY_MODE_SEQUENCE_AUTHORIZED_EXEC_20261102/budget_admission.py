"""Explicit later human authority, not a silent bypass of original budget result."""
from pathlib import Path
import json
def validate_values(pf,benchmark,authority):
    if pf.get('status')!='PASS_NUMERICAL_PREFLIGHT_REUSED_WITH_EXPLICIT_BUDGET_AUTHORIZATION':raise ValueError('NUMERICAL_PREFLIGHT_NOT_ADMITTED')
    if authority.get('status')!='HUMAN_AUTHORIZED_RUN_BEYOND_ORIGINAL_30MIN' or authority.get('allow_original_budget_override') is not True:raise ValueError('NO_CURRENT_BUDGET_AUTHORITY')
    if authority.get('finite_stage')!='Q' or authority.get('scientific_changes') is not False or authority.get('remove_or_reduce_audits') is not False:raise ValueError('AUTHORITY_SCOPE')
    if benchmark.get('conservative_full_seconds',0)<=0:raise ValueError('MISSING_ACTUAL_ESTIMATE')
    if pf.get('numerical_GPU')!='PASS' or pf.get('independent_saved')!='PASS' or pf.get('exact_resume')!='PASS':raise ValueError('MISSING_ACTUAL_NUMERICAL_EVIDENCE')
    return True
def validate(root):
    root=Path(root)
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    return validate_values(read(root/'PREFLIGHT.json'),read(root/'BENCHMARK.json'),read(root/'authorization.json'))
