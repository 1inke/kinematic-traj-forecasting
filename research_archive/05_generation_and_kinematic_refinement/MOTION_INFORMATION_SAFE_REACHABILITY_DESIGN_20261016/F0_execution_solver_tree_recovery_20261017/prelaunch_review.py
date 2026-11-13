import common as c
from run import independent_attempt
def main():
    pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS';files=c.read(c.H/'source_manifest.json')['files']
    assert c.digest(files)==pre['source_digest'];c.verify(files)
    for p in sorted((c.H/'preflight_smoke').glob('w*_slot*.json')):
        row=c.read(p);row['files']={str(p):c.sha(p),str(p.with_suffix('.npz')):c.sha(p.with_suffix('.npz'))};independent_attempt(row)
    c.verify(files);c.write('PRELAUNCH_REVIEW.json',dict(status='PASS',source_digest=pre['source_digest'],source_count=len(files),actual_saved_attempt_branches_revalidated=True,no_forward_optimizer_or_new_solve=True,resource=c.RESOURCE))
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_F0_PRELAUNCH');raise
