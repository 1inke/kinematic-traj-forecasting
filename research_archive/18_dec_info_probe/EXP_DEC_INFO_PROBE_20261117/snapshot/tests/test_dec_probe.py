import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
import torch
from dec_model import InfoRefiner, risk_geometry, route, make_basis, predict
from dec_analysis import permutation, diagnose, decision
from dec_train import new_model, future_shuffle_check
from probe_model import score_head, Refiner

torch.set_num_threads(2)

def fixture():
    rng = torch.Generator().manual_seed(17)
    n = 4
    data = dict(mode=torch.randn(n, 20, 64, generator=rng), relative=torch.randn(n, 20, 3, 2, generator=rng),
                scale=torch.ones(n, 2), logits=torch.zeros(n, 20),
                tokens_self=torch.randn(n, 8, 96, generator=rng).half(),
                tokens_nbr=torch.randn(n, 8, 96, generator=rng).half(),
                nbr_pos=torch.randn(n, 8, 2, generator=rng),
                nbr_vel=torch.randn(n, 8, 2, generator=rng),
                self_vel=torch.randn(n, 2, generator=rng), y=torch.zeros(n, 12, 2),
                nbr_valid=torch.arange(8)[None] < torch.tensor([0, 1, 4, 8])[:, None])
    return data


class Probe(unittest.TestCase):
    def test_risk_against_scalar_reference(self):
        t = fixture()
        p = torch.einsum('tp,nkpc->nktc', make_basis(), t['relative'])
        idx, keep, f, dca = risk_geometry(p, t['nbr_pos'], t['nbr_vel'], t['nbr_valid'])
        for i in range(4):
            for k in range(20):
                expected = []
                for j in range(8):
                    d = [t['nbr_pos'][i,j].numpy()+t['nbr_vel'][i,j].numpy()*.4*(q+1)-p[i,k,q].numpy() for q in range(12)]
                    g = np.hypot(*np.array(d).T)
                    q = int(np.argmin(g))
                    d0 = float(t['nbr_pos'][i,j].norm())
                    np.testing.assert_allclose(f[i,k,j], [g[q], (q+1)/12, d0, *d[q]], atol=2e-6)
                    self.assertAlmostEqual(float(dca[i,k,j]), g[q], places=5)
                    if bool(t['nbr_valid'][i,j]) and g[q] < 3:
                        expected.append((-(np.exp(-.4*(q+1)/2.5-g[q]/1.25)+.1*np.exp(-d0/3)), j))
                self.assertEqual(idx[i,k][keep[i,k]].tolist(), [j for _, j in sorted(expected)[:3]])

    def test_ties_and_threshold(self):
        p = torch.zeros(1,20,12,2)
        pos = torch.tensor([[[3.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],[0.,0.],[9.,0.],[9.,0.]]])
        vel = torch.zeros_like(pos)
        val = torch.tensor([[True]*5+[False]*3])
        i, ok, f, _ = risk_geometry(p, pos, vel, val)
        self.assertEqual(i[0,0][ok[0,0]].tolist(), [1,2,3])
        self.assertAlmostEqual(float(f[0,0,1,1]), 1/12)
        _, ok, _ = route('C3', p, pos, vel, val & (torch.arange(8)[None]==0), torch.zeros(1,2))
        self.assertFalse(ok.any())

    def test_c4_common_and_c2_valid(self):
        t = fixture()
        for a, counts in [('C2',[0,1,4,8]),('C4',[0,1,3,3])]:
            i, ok, f = route(a, t['relative'], *(t[k] for k in ('nbr_pos','nbr_vel','nbr_valid','self_vel')))
            self.assertTrue(torch.equal(i, i[:, :1].expand_as(i)))
            self.assertEqual(ok.sum(-1)[:,0].tolist(), counts)
            self.assertTrue(torch.equal(f[:,0,:,2:4], (t['nbr_vel']-t['self_vel'][:,None]).gather(1,i[:,0,:,None].expand(-1,-1,2))))

    def test_initialization_and_capacity(self):
        a = new_model('A1', 0, 'cpu')
        w = new_model('A1L', 0, 'cpu')
        c = new_model('C1', 0, 'cpu')
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(c.state_dict()[k], v), k)
        self.assertTrue(torch.equal(a.ffn[0].weight, w.ffn[0].weight[:48]))
        self.assertTrue(torch.equal(a.ffn[2].weight, w.ffn[2].weight[:,:48]))
        self.assertLess(abs(sum(p.numel() for p in c.parameters())/sum(p.numel() for p in w.parameters())-1), .05)
        for arm in ('C2','C3','C4'):
            self.assertTrue(all(torch.equal(v, state) for v, state in zip(c.state_dict().values(), new_model(arm,0,'cpu').state_dict().values())))
        t = fixture()
        torch.manual_seed(0)
        old = Refiner('A1')
        self.assertTrue(torch.equal(a.stem(t['mode'],t['relative'],t['scale']), c.stem(t['mode'],t['relative'],t['scale'])))
        h = score_head()
        for x, y in zip(predict(old, h, t), predict(a, h, t)):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(all(torch.equal(v, a.state_dict()[k]) for k,v in old.state_dict().items()))

    def test_masked_inputs_and_future_isolation_after_perturbing_heads(self):
        t = fixture()
        h = score_head()
        changed = {k: v.clone() for k,v in t.items()}
        for k in ('tokens_nbr','nbr_pos','nbr_vel'):
            changed[k][~t['nbr_valid']] = 10
        for a in ('C1','C2','C3','C4'):
            model = new_model(a, 0, 'cpu')
            with torch.no_grad():
                model.anchor_out.weight.normal_(0,.1)
                model.score_out.weight.normal_(0,.1)
            x, y = predict(model,h,t), predict(model,h,changed)
            self.assertGreater(float(x[2].detach().abs().max()), 0)
            self.assertTrue(all(torch.equal(i,j) for i,j in zip(x,y)), a)
            self.assertEqual(future_shuffle_check(model,h,t)['status'], 'BITWISE_EQUAL')

    def test_shuffle_matched_counts_and_singletons(self):
        counts = np.array([0, 1, 2, 2, 2, 1, 0, 3])
        ix, audit = permutation(counts)
        np.testing.assert_array_equal(counts[ix], counts)
        self.assertEqual(audit['2']['moved'], 3)
        self.assertEqual(audit['3']['moved'], 0)
        np.testing.assert_array_equal(ix, permutation(counts)[0])

    def test_d0_excludes_gt_and_handles_empty(self):
        t = fixture()
        meta = dict(group=np.array([0,0,1,1]), labels=np.array(['stop','stop','turn','turn']))
        w,g = diagnose(dict(tensors=t,meta=meta), make_basis())
        self.assertFalse(w['social'][0])
        self.assertEqual(w['c3_count'][0].sum(),0)
        self.assertEqual(g['change']['n'],0)
        self.assertIsNone(g['change']['social_rate'])
        self.assertEqual(g['all']['c4_neighbor_count_histogram'],[20,20,0,40])

    def test_decision_and_separate_protection(self):
        names = ('all','social','non_social','stable','decel_stop')
        base = {n:dict(n=50,minFDE=1.,minADE=1.,top1FDE=1.) for n in names}
        def arm(v, social=None):
            return {n:dict(n=50,minFDE=v if n!='social' or social is None else social,minADE=v,top1FDE=1.) for n in names}
        def rows(clean,shuffled=None):
            return {str(i):dict(clean=clean,shuffled=shuffled,parameter_count=15000,
                                latency=dict(refiner_only=dict(median_ms=.5))) for i in range(3)}
        run = dict(A1=rows(arm(.9871)),A1L=rows(arm(.988)),C1=rows(arm(.985)),
                   C2=rows(arm(.981),arm(.985)),C3=rows(arm(.951,.925),arm(.985,.981)),
                   C4=rows(arm(.98),arm(.985)))
        r = decision(run,base,295474,{'1':dict(n=100,moved=100)})
        self.assertEqual(r['verdict'],'DESIGN_PROMISING')
        self.assertTrue(r['eligible_to_advance'])
        run['C3']['0']['clean']['all']['top1FDE']=1.1
        r=decision(run,base,295474,{'1':dict(n=100,moved=100)})
        self.assertEqual(r['verdict'],'DESIGN_PROMISING')
        self.assertEqual(r['protection_status'],'PROTECTION_FAILED')
        self.assertFalse(r['eligible_to_advance'])
        self.assertEqual(decision(dict(A1=rows(arm(.995))),base,295474,{})['verdict'],'PROXY_INVALID')

if __name__ == '__main__':
    unittest.main(verbosity=2)
