"""CPU synthetic scalar/all-parameter equivalence, not student training."""
from common import *

def main():
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261022);module=model_module();rows=[]
    for nr,nn in [(3,4),(3,0),(0,4),(0,0)]:
        net=module.build('BASE').double();net.requires_grad_(True)
        # Fixed synthetic nonzero head to test nontrivial upstream gradients.
        with torch.no_grad():net.head.weight.copy_(torch.randn_like(net.head.weight)*.01)
        state=frozen_state(net);p=net(torch.randn(2,8,2,dtype=torch.float64))['paths'];R=torch.randn(2,nr,12,2,dtype=torch.float64);N=torch.randn(2,nn,12,2,dtype=torch.float64)
        def term(t):
            if t.shape[1]==0:return p.sum()*0,0
            e=torch.linalg.vector_norm(t[:,:,None]-p[:,None],dim=-1);q=e.mean(-1)+.5*e[:,:,:,-1];return q.min(-1).values.sum(),2*t.shape[1]
        sr,nr2=term(R);sn,nn2=term(N);den=max(nr2+nn2,1);union=(sr+sn)/den;decomposed=(nr2/den)*(sr/max(nr2,1))+(nn2/den)*(sn/max(nn2,1));params=list(net.parameters());g1=torch.autograd.grad(union,params,retain_graph=True);g2=torch.autograd.grad(decomposed,params);diff=max(float((a-b).abs().max()) for a,b in zip(g1,g2));scalar=float((union-decomposed).detach().abs())
        if max(diff,scalar)>2e-12 or any(not torch.isfinite(g).all() for g in g1+g2):raise RuntimeError('ALL_PARAMETER_ALGEBRA_IDENTITY')
        if any(not torch.equal(v.detach().cpu(),state[k]) for k,v in net.state_dict().items()):raise RuntimeError('FIXTURE_MODEL_UPDATED')
        rows.append(dict(R=nr2,N=nn2,scalar_abs_diff=scalar,all_parameter_gradient_maxdiff=diff,parameter_tensors=len(params),parameter_values=sum(p.numel() for p in params),no_optimizer_no_update=True))
    write_json(HERE/'ALGEBRA_ALL_PARAMETER_FIXTURE.json',dict(status='PASS',scope='synthetic freshly instantiated CPU network only, no saved student changed or retrained',source_model_sha256=sha256(MODEL_DIR/'models.py'),fixture_script_sha256=sha256(__file__),rows=rows))
    print('All-parameter identity fixture PASS;4 group-emptiness cases;0 optimizer steps.')

if __name__=='__main__':main()
