import common as c,metrics as v
def main():
    pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS';files=c.read(c.H/'source_manifest.json')['files'];assert c.digest(files)==pre['source_digest'];c.verify(files)
    for cell in ('A','B','C','D'):
        p=c.H/'preflight_smoke'/cell;v.validate(c.npz(p/'paths.npz'),c.read(p/'metrics.json'),c.npz(p/'aux.npz'),'integrated_acceleration')
    c.verify(files);c.write('PRELAUNCH_REVIEW.json',dict(status='PASS',source_count=len(files),source_digest=pre['source_digest'],actual_all_four_saved_pipeline_revalidated=True,no_new_forward_backward_optimizer=True,resource=c.RESOURCE))
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_P1_PRELAUNCH');raise



