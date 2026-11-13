"""Retain failed v2 adapter/schema evidence and lock new explicit adapter."""
import shutil
from common import *
from preflight_v2 import verify_v2

def main():
    verify_sources();verify_v2();folder=HERE/'failures/preflight2_legacy42_schema';folder.mkdir(parents=True,exist_ok=True)
    for name in ['preflight_v2.py','IMPLEMENTATION_TRANSITION_V2.json','reconcile_v3.py']:
        if not (folder/name).exists():shutil.copy2(HERE/name,folder/name)
    write_json(HERE/'IMPLEMENTATION_TRANSITION_V3.json',dict(reason='seed42 old JSON lacks native_good_count; check derived good-rescue+lost identity explicitly, never claim field was stored',original_sources_unchanged=True,scientific_protocol_sha_unchanged=sha256(HERE/'protocol.json'),files={p.relative_to(ROOT).as_posix():sha256(p) for p in [HERE/'reconcile_v3.py',HERE/'preflight_v3.py',HERE/'run_frozen_v3.py',HERE/'preserve_transition_v3.py']},historical_missing_optimizer_RNG_not_recreated=True,no_scientific_threshold_changed=True))
    print('v3 legacy schema adapter locked; original files and all numeric thresholds unchanged.')

if __name__=='__main__':main()
