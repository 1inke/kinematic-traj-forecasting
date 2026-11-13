"""Read all48 already independently audited groups; verify scalar gate and source/output provenance. No model or arrays replay."""
import json,hashlib,math,time
from pathlib import Path
H=Path(__file__).resolve().parents[1]
def read(p):return json.loads((H/p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def verify(files):
    for p,h in files.items():assert sha(p)==h,('SHA_CHANGED',p)
def main():
    t=time.perf_counter();result=read('RESULT.json');decision=read('DECISION.json');jobs=read('JOBS.json');audit=read('ARITHMETIC_AUDIT.json');files=read('source_manifest.json')['files'];worker=read('worker_state.json');exit=read('exit_status.json')
    assert worker['status']==exit['status']=='SUCCESS' and result['status']=='PASS_DIAGNOSTIC_ONLY_ANALYTIC_ENVELOPE'
    assert audit['status']=='PASS' and audit['jobs']==4 and audit['groups']==48 and audit['maxerror']<=2e-12 and len(files)==1823
    outputs=dict(result['outputs_SHA']);assert len(outputs)==8;verify(files);verify(outputs);donefiles={};rows=[];summary=[]
    for split,r in jobs.items():
        donepath=H/'jobs'/f'{split}.done.json';done=json.loads(donepath.read_text());assert done['status']=='PASS' and done['source_digest']==result['source_digest'] and done['files']==r['files'];donefiles[str(donepath)]=sha(donepath)
        assert result['records'][split]==r and len(r['metrics'])==3
        for g in ('overall','stable','transition','change'):
            a,b,c=[r['metrics'][p][g] for p in ('native20','UB_bottom4','UB_all20')];n=a['n'];assert n==b['n']==c['n'] and a['good']<=b['good']<=c['good']
            assert a['lost']==b['lost']==c['lost']==0
            assert b['rescue']==b['good']-a['good'] and c['rescue']==c['good']-a['good']
            gb=b['good']-a['good'];ga=c['good']-a['good'];bm=100*gb/n;am=100*ga/n
            summary.append(dict(split=split,group=g,n=n,tracks=a['tracks'],native_good=a['good'],bottom4_good=b['good'],all20_good=c['good'],native_coverage_percent=100*a['good']/n,bottom4_coverage_percent=100*b['good']/n,all20_coverage_percent=100*c['good']/n,bottom4_gain_pp=bm,all20_gain_pp=am,bottom4_realized_relaxed_opportunity_fraction=None if ga==0 else gb/ga,unavailable_to_fixed_bottom4_count=c['good']-b['good']))
            for port,z in zip(('native20','UB_bottom4','UB_all20'),(a,b,c)):
                assert math.isclose(z['coverage'],z['good']/n,rel_tol=0,abs_tol=2e-15)
                rows.append(dict(split=split,group=g,port=port,**z))
        assert sum(r['metrics']['native20'][g]['n'] for g in ('stable','transition','change'))==r['metrics']['native20']['overall']['n']
    assert len(rows)==48
    fit=next(r for r in summary if r['split']=='FIT' and r['group']=='change');assert 100*(fit['bottom4_good']-fit['native_good'])<5*fit['n'] and 100*(fit['all20_good']-fit['native_good'])>=5*fit['n']
    assert decision['status']=='STOP_FIXED_BOTTOM4_OPPORTUNITY' and not decision['A1_preflight_triggered']
    verify(files);verify(outputs);verify(donefiles)
    report=dict(status='PASS_COMPLETION_SAVED_SCALAR_REVIEW',source_files=len(files),primary_outputs=len(outputs),done_contracts=len(donefiles),done_SHA=donefiles,groups=rows,summary=summary,exact_FIT_integer_gate_reconstructed=True,source_output_done_initial_final_SHA=True,scope='all48 saved-group counts/coverage/rescue-lost/nesting/FIT integergate+1823source8primary4doneSHA initialfinal. Full geometry/complete-link/five-motion independently reconstructed in production audit, NOT rerun here. No model/optimizer/newdata/CI.',elapsed_seconds=time.perf_counter()-t)
    (H/'completion_review/REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],summary=summary,oracle_motion_pass_groups={p:sum(r['motion_literal_1pct'] for r in rows if r['port']==p) for p in ('native20','UB_bottom4','UB_all20')},elapsed_seconds=report['elapsed_seconds']),ensure_ascii=False))
if __name__=='__main__':main()
