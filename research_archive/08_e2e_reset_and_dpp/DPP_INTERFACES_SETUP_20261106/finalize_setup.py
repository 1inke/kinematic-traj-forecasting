"""Seal this interface delivery only, without touching the running Stage A."""
from interface_common import *
from stage_b_readiness import inspect
import ast,time

def main():
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    latest=max((ROOT/'tests').glob('*.json'),key=lambda p:int(p.stem));tests=read(latest)
    if tests['status']!='PASS_ENGINEERING_FIXTURES_ONLY':raise ValueError('LATEST_TEST_NOT_PASS')
    check_files(tests['source_files'])
    if set(tests['source_files'])!=set(seal(ROOT.glob('*.py'))):raise ValueError('TEST_SOURCE_SET_STALE')
    check_files(read(STAGE_A/'source_manifest.json')['files'])
    result=inspect();result.update(at=time.strftime('%Y-%m-%d %H:%M:%S'),engineering_tests=tests['tests'],engineering_receipt=str(latest),official_asset_files=len(read(ROOT/'YNET_DOWNLOAD.json')['files']),download_bytes=read(ROOT/'YNET_DOWNLOAD.json')['network_bytes'],actual_current_generator_test=read(ROOT/'current_h3_check/RESULT.json')['status'],actual_current_pipeline_test=read(ROOT/'CURRENT_PIPELINE_CHECK.json')['status'],old_stage_A_sources_unchanged=True,source_paper='https://arxiv.org/html/1907.04967#S4.SS1',CVAE_human_condition='Only pretrain CVAE if current generator interface fails; current interface passed',no_claim_of_coverage_gain_or_formal_resume_preflight=True)
    write(ROOT/'SETUP_STATUS.json',result)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='FINAL_INTERFACE_SEAL.json']
    write(ROOT/'FINAL_INTERFACE_SEAL.json',dict(status='PASS_ENGINEERING_INTERFACES_REAL_CURRENT_MODEL_MAP_ASSET_DELIVERY_NOT_ALL_PRODUCTION_READY',files=seal(files),original_Stage_A_source_manifest_sha=sha(STAGE_A/'source_manifest.json'),no_training_or_old_source_mutation=True))
    print('INTERFACE_DELIVERY_SEALED',len(files),'CVAE_NOT_NEEDED_MAPS_PENDING')

if __name__=='__main__':main()
