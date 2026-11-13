from core import *
def main():
    parent=ROOT/'qoder_work/K20_COLLABORATIVE_REFINEMENT_EXEC_20261027';old=read_json(parent/'source_manifest.json');assert digest(old['files'])==old['digest']
    for n,h in old['files'].items():assert sha256(Path(n) if Path(n).is_absolute() else ROOT/n)==h,n
    records=[]
    for p in parent.glob('*.py'):
        q=HERE/p.name
        if p.name not in ('core.py','worker.py'):assert sha256(p)==sha256(q);records.append(dict(module=p.name,byte_EXACT=True))
        else:
            a=ast.parse(p.read_text(encoding='utf-8'));b=ast.parse(q.read_text(encoding='utf-8'));func=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};aa,bb=func(a),func(b);allowed={'guard','build_source'} if p.name=='core.py' else {'child'};changed=[k for k in aa if aa[k]!=bb[k]];assert set(changed)==allowed;records.append(dict(module=p.name,only_changed_functions=changed,all_other_functions_AST_EXACT=True))
    assert sha256(parent/'protocol.json')==sha256(HERE/'protocol.json');old_policy=read_json(parent/'RESOURCE_STAGE_PLAN.json');new_policy=read_json(HERE/'RESOURCE_STAGE_PLAN.json')
    for key in ('factor','startup','runtime','CPU_estimate_bytes','no_silent_inflate_or_OS_pagefile_userkill'):assert old_policy[key]==new_policy[key]
    estimate=read_json(HERE/'MEASURED_MEMORY_ESTIMATE.json');assert new_policy['GPU_estimate_bytes']==sum(new_policy['GPU_inventory_bytes'].values())==estimate['new_estimate_bytes'];peak=read_json(parent/'PREFLIGHT_GPU_RESOURCE.json');assert max(max(x['snapshot']['peak_private'],x['snapshot']['peak_rss']) for x in peak['observations'])==estimate['measured_peak_bytes']
    for n,h in read_json(HERE/'prepared/ready.json')['outputs'].items():assert sha256(HERE/'prepared'/n)==h and sha256(parent/'prepared'/n)==h
    write_json(HERE/'SOURCE_EQUIVALENCE.json',dict(status='PASS',parent_source=old['digest'],parent_count=len(old['files']),records=records,scientific_protocol_JSON_EXACT=True,resource_formula_factor_CPU_EXACT=True,GPU_estimate_explicit_actual_peak_revision=estimate,prepared_all_SHA_EXACT=True,all_parent_locked_sources_unchanged=True,formal_parent_updates=0,no_completed_model_retraining=True,only_resource_timing_estimate_and_namespace=True));print('PASS SOURCE EQUIVALENCE',flush=True)
if __name__=='__main__':main()

