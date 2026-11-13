"""Additional registered-statistics/active-face/SHA-contract review; locked code unchanged."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c,model as m
import numpy as np,torch,ast,time,gc
def faces(out,raw):
    e=torch.linalg.vector_norm(out[0]-raw[:,None],dim=-1);w=(e.mean(-1)+.5*e[:,:,-1]).argmin(1)
    pair=torch.triu_indices(20,20,1,device=out[0].device);hinge=torch.linalg.vector_norm(out[0][:,pair[0]]-out[0][:,pair[1]],dim=-1).mean(-1)<.3
    normalization=torch.linalg.vector_norm(torch.tanh(out[1]).double(),dim=-1)>1
    return w,hinge,normalization
def main():
    start=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30;files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);data=c.load();st=c.npz(c.H/'statistics.npz');ix=data['split']['FIT'];p=data['p2'][ix].astype(float);n=data['native'][ix].reshape(-1,24).astype(np.float32).astype(float);mean=np.concatenate((p.mean(0),n.mean(0)));std=np.concatenate((p.std(0),n.std(0)));std=np.where(std<1e-6,1.,std);error=max(c.close(mean,st['mean'],1e-10),c.close(std,st['std'],1e-10));assert np.array_equal(data['sample_id'][ix],st['training_ids']);del p,n;gc.collect()
    face_records=[]
    for arm in m.ARMS:
        m.seed(162);net=m.Refiner(st).to(m.DEVICE).double()
        with torch.no_grad():net.head[-1].weight.normal_(0,.002);net.head[-1].bias.fill_(.01)
        p2,native=m.inputs(data,data['split']['FIT'][:8]);p2=p2.double();native=native.double();gt=torch.from_numpy(data['gt'][data['split']['FIT'][:8]]).to(m.DEVICE);center=[p.detach().clone() for p in net.parameters()];dirs=[torch.randn_like(p) for p in net.parameters()];norm=torch.sqrt(sum((x*x).sum() for x in dirs));dirs=[x/norm for x in dirs]
        ref=faces(net(p2,native,arm),gt);checks=[]
        for eps in (1e-4,1e-5):
            for sign in (-1,1):
                with torch.no_grad():
                    for p,p0,d in zip(net.parameters(),center,dirs):p.copy_(p0+sign*eps*d)
                    actual=faces(net(p2,native,arm),gt)
                assert all(torch.equal(a,b) for a,b in zip(ref,actual)),'FD_ACTIVE_FACE_CHANGED_UNDETERMINED'
                checks.append(dict(eps=eps,sign=sign,quality_winner_hinge_and_control_norm_faces_EXACT=True))
        cp=c.H/'resume_smoke'/arm/'resumed.pt';done=dict(status='PASS',source_digest=pre['source_digest'],files={str(cp):c.sha(cp)});assert c.valid_done(done,pre['source_digest']);bad=0
        for row in (dict(done,source_digest='wrong'),dict(done,files={str(cp):'0'*64})):
            try:c.valid_done(row,pre['source_digest'])
            except AssertionError:bad+=1
        assert bad==2;face_records.append(dict(arm=arm,faces=checks,completed_checkpoint_SHA_contract_accept_and_badsource_output_reject=True))
    assert any(isinstance(x,ast.If) and 'done.exists()' in ast.unparse(x.test) for x in ast.walk(ast.parse((c.H/'run.py').read_text(encoding='utf-8'))))
    c.verify(files);c.write('prelaunch_review/REVIEW.json',dict(status='PASS',source_count=len(files),source_digest=c.digest(files),source_initial_final_unchanged=True,FIT_only_full_mean_std_independent_maxerror=error,FD_face_checks=face_records,locked_science_modified=False,only_small_registered_double_fixture_forward_no_training_or_oldmodel_replay=True,review_script_SHA=c.sha(__file__),resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-start));print('PRELAUNCH_REVIEW PASS',error,len(files),flush=True)
if __name__=='__main__':main()
