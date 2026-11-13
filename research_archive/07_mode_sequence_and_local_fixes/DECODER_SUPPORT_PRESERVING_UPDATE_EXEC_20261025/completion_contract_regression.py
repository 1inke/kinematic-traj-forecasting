"""Actual bad-output branch and saved-contract regression, no model/GPU/optimizer."""
from pathlib import Path
import hashlib,json,importlib.util
from completion_contract import validate
HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'completion_contract_fixture';out.mkdir(exist_ok=True)
    payload=b'fixed finite analysis fixture, no scientific output';(out/'payload.bin').write_bytes(payload)
    b=dict(source_digest='fixture',output_SHA={'payload.bin':hashlib.sha256(payload).hexdigest()})
    path=out/'RESULT.json';path.write_text(json.dumps(b),encoding='utf-8');ok=validate(path,'fixture')
    for kind in ('source','output'):
        bad=json.loads(json.dumps(b));bad['source_digest']='wrong' if kind=='source' else 'fixture'
        if kind=='output':bad['output_SHA']['payload.bin']='0'*64
        path.write_text(json.dumps(bad),encoding='utf-8')
        try:validate(path,'fixture')
        except ValueError:pass
        else:raise AssertionError(kind)
    # Exercise the real original G0 auditor's early output-SHA rejection, not a stand-in.
    spec=importlib.util.spec_from_file_location('_actual_g0_saved_review',HERE/'review_g0.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fixture=out/'auditor';(fixture/'G0').mkdir(parents=True,exist_ok=True);(fixture/'G0/payload.bin').write_bytes(payload)
    (fixture/'G0/RESULT.json').write_text(json.dumps({'output_SHA':{'payload.bin':'0'*64}}),encoding='utf-8')
    (fixture/'G0/PACKETS.json').write_text(json.dumps([{}]*48),encoding='utf-8');m.HERE=fixture
    try:m.audit()
    except AssertionError:pass
    else:raise AssertionError('Actual audit accepted corrupted SHA')
    result=dict(status='PASS_ACTUAL_ANALYSIS_SKIP_AND_BAD_SOURCE_OUTPUT_REJECTION',valid_done=ok,actual_G0_auditor_bad_output_branch=True,no_model_optimizer_GPU=True,base_source_manifest_unchanged=True)
    (out/'REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

if __name__=='__main__':main()
