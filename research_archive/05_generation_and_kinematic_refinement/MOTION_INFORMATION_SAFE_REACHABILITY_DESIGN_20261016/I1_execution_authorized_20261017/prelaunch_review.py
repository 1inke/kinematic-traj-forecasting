import common as c
def main():
    pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS'
    files=c.read(c.H/'source_manifest.json')['files'];assert c.digest(files)==pre['source_digest'];c.verify(files)
    # Current-source saved smoke contracts and actual production manifest branches, no new training.
    from train import saved,manifest
    data=c.load()
    for smoke in pre['smoke']:
        row=c.read(smoke['done_file']);c.valid_done(row,smoke['source_digest'])
        z=saved(row);assert len(z['sample_id'])==128
    records=[manifest(data,a,s,pre['source_digest']) for a in ('S','H') for s in (42,43,44)]
    assert len({x['statistics_SHA'] for x in records})==1
    c.verify(files);c.write('PRELAUNCH_REVIEW.json',dict(status='PASS',source_digest=pre['source_digest'],source_count=len(files),actual_six_production_manifest_branches=records,all_saved_smoke_SHA_compatible=True,no_optimizer_or_model_forward=True,old_F16_not_loaded=True,resource=c.RESOURCE))
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_I1_PRELAUNCH');raise
