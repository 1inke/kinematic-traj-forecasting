from pathlib import Path
from types import SimpleNamespace
import copy,sys,unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from probe_model import Refiner,predict,score_head,make_basis,allocation,extra_loss
from probe_metrics import decide,metrics
from probe_train import Trainer,cpu_tree
from probe_cache import criterion_from

def fixture(n=8):
    torch.manual_seed(17)
    mode=torch.randn(n,20,64);relative=torch.randn(n,20,3,2);scale=torch.rand(n,2)+.5
    head=score_head();head.requires_grad_(False)
    z=head(torch.cat([mode,relative.flatten(-2)],-1)).squeeze(-1).detach()
    return dict(mode=mode,relative=relative,scale=scale,logits=z,y=torch.randn(n,12,2)),head
def criterion():
    r=dict(decoder=dict(WINNER_FDE_WEIGHT=.25,CLASSIFICATION_WEIGHT=.5,ENDPOINT_LOSS_WEIGHT=.25,SOFT_CLASSIFICATION=True,SOFT_TEMPERATURE=.5,HARD_CLASSIFICATION_WEIGHT=.25,DIVERSITY_WEIGHT=.05,DIVERSITY_MARGIN=.4,SHORT_HORIZON_WEIGHT=.2,SHORT_HORIZON_STEPS=2),loss=dict(START_WEIGHT=1.,END_WEIGHT=1.,WEIGHTED_PENALTY='none'),pred_time=4.8,frequency=2.5)
    return criterion_from(r,'cpu',{'regression.weights':torch.ones(12)})
class Tests(unittest.TestCase):
    def test_zero_init_and_parameter_pairing(self):
        cache,head=fixture();states=[]
        for arm in ('A1','A2','A3','A4','A5','A6'):
            torch.manual_seed(4);model=Refiner(arm);states.append(copy.deepcopy(model.state_dict()))
            p,z,s=predict(model,head,cache);torch.testing.assert_close(p,torch.einsum('tp,nkpc->nktc',make_basis(),cache['relative']),atol=0,rtol=0);torch.testing.assert_close(z,cache['logits'],atol=0,rtol=0);self.assertFalse(s.any())
        for state in states[1:]:
            for k in state:self.assertTrue(torch.equal(state[k],states[0][k]))
    def test_strict_self_has_no_cross_candidate_gradient(self):
        cache,head=fixture(1);m=Refiner('A1');torch.nn.init.normal_(m.anchor_out.weight,std=.03)
        cache['mode'].requires_grad_(True);cache['relative'].requires_grad_(True)
        p,z,s=predict(m,head,cache);p[:,0].sum().backward()
        self.assertEqual(float(cache['mode'].grad[:,1:].abs().max()),0)
        self.assertEqual(float(cache['relative'].grad[:,1:].abs().max()),0)
    def test_sparse_graph_and_signed_relations(self):
        cache,head=fixture(2);features,graph=Refiner('A5').relations(cache['relative'],cache['logits'])
        self.assertTrue((graph.sum(-1)==5).all());self.assertTrue(graph.diagonal(dim1=-2,dim2=-1).all())
        torch.testing.assert_close(features[:,0,1,:2],cache['relative'][:,1,-1]-cache['relative'][:,0,-1])
        torch.testing.assert_close(features[:,0,1,3],cache['logits'][:,1]-cache['logits'][:,0])
    def test_extra_loss_only_selected_path_gradient(self):
        p=torch.zeros(2,20,12,2);p[:,:,:,0]=2+torch.arange(20)[None,:,None]*.03;p.requires_grad_();y=torch.zeros(2,12,2);z=torch.arange(20).float().expand(2,20).clone().requires_grad_()
        value,info=extra_loss(p,y,z);self.assertTrue(info['active'].all());self.assertTrue((info['r']!=info['winner']).all());value.backward()
        nonzero=p.grad.abs().sum((-1,-2))>0;self.assertTrue((nonzero.sum(-1)==1).all());self.assertTrue(nonzero[torch.arange(2),info['r']].all());self.assertIsNone(z.grad)
        self.assertEqual(float(extra_loss(p.detach(),y,torch.zeros_like(z))[0]),0)
    def test_gate_is_not_min_fde_uncovered(self):
        p=np.ones((1,20,12,2))*5;p[0,0]=0;p[0,0,-1,0]=.5;p[0,1,:,0]=.3;p[0,1,:,1]=0
        w=metrics(p,np.zeros((1,12,2)),np.zeros((1,20)),np.ones(1,bool))
        self.assertTrue(w['allocation_gate'][0]);self.assertFalse(w['uncovered'][0])
    def test_future_not_in_prediction(self):
        cache,head=fixture();m=Refiner('A5');torch.nn.init.normal_(m.anchor_out.weight,std=.02)
        a=predict(m,head,cache);cache['y']=cache['y']+1000;b=predict(m,head,cache)
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
    def test_resume_exact(self):
        cache,head=fixture(9);crit=criterion();contract={'fixture':1}
        a=Trainer('A5',1,cache,head,crit,'cpu',epochs=2,batch=4);a.epoch_step();state=a.snapshot(contract);a.epoch_step()
        b=Trainer('A5',1,cache,head,crit,'cpu',epochs=2,batch=4);b.restore(state,contract);b.epoch_step()
        for k,v in a.model.state_dict().items():self.assertTrue(torch.equal(v,b.model.state_dict()[k]),k)
        self.assertEqual(a.history[-1]['order_hash'],b.history[-1]['order_hash']);self.assertEqual(a.history[-1]['loss'],b.history[-1]['loss'])
    def test_mechanism_and_protection_are_separate(self):
        def entry(fde,top1=1):return {'groups':{g:dict(minADE=fde,minFDE=fde,top1FDE=top1) for g in ('all','stable','decel_stop')}}
        base=entry(1);runs={a:{str(s):entry(v) for s in range(3)} for a,v in dict(A1=.98,A2=.98,A3=.98,A4=.97,A5=.94,A6=.96).items()}
        d=decide(base,runs);self.assertEqual(d['verdict'],'H_SUPPORTED');self.assertTrue(d['eligible_to_advance']);self.assertEqual(d['threshold'],.005)
        for row in runs['A5'].values():row['groups']['all']['top1FDE']=1.03
        d=decide(base,runs);self.assertEqual(d['verdict'],'H_SUPPORTED');self.assertEqual(d['protection_status'],'PROTECTION_FAILED');self.assertFalse(d['eligible_to_advance'])
        invalid={'A1':{str(s):entry(1.01) for s in range(3)}};self.assertEqual(decide(base,invalid)['verdict'],'PROXY_INVALID')

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
