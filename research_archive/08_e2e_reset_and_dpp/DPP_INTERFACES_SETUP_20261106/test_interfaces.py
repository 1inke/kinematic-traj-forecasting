"""Engineering fixtures only, never pretraining or production scientific results."""
import ast,io,sys,time,unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from interface_common import *
from scene_interface import *
from dsf_interface import *
sys.path.insert(0,str(STAGE_A));from scoring import kinematic,dpp,scene_score
torch.set_num_threads(1)

class FixtureGenerator(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(3,24,bias=False);self.register_buffer('constant',torch.tensor(True))
    def encode_context(self,past):return past[:,-1]
    def decode(self,z,context,p0):return p0[:,None]+self.projection(z).reshape(-1,12,2)+context[:,None]*.01

CONTRACT=dict(latent_prior='standard_normal',output_unit='world_metres',dt_seconds=.4,T=12,latent_dim=3,context_dim=2)

class TestInterfaces(unittest.TestCase):
    def setUp(self):torch.manual_seed(1126)
    def pipeline(self):
        return DSFPipeline(FrozenGaussianGenerator(FixtureGenerator().double(),CONTRACT),DSFSampler(2,3).double())
    def arrays(self):
        a=np.tile(np.linspace(-.6,.8,50),(50,1));return dict(distance_metres=a,known=np.ones((50,50),bool),world_to_grid=np.diag([10.,10.,1.]),metres_per_cell=np.array(.1))
    def test_syntax(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    def test_no_gaussian_embedding_substitution(self):
        with self.assertRaises(ValueError):FrozenGaussianGenerator(FixtureGenerator(),CONTRACT|{'latent_prior':'mode_embedding'})
    def test_generator_freeze_but_latent_grad(self):
        m=self.pipeline();m.train();p=torch.randn(8,8,2,dtype=torch.double);before={k:v.clone() for k,v in m.generator.state_dict().items()};o=m(p)
        loss=-ec_from_weights(o['paths'],torch.ones((8,20),dtype=torch.double),.4).mean();loss.backward()
        self.assertGreater(sum(x.grad.abs().sum().item() for x in m.sampler.parameters()),0)
        self.assertTrue(all(x.grad is None and not x.requires_grad for x in m.generator.parameters()))
        for k,v in before.items():self.assertTrue(torch.equal(v,m.generator.state_dict()[k]))
        self.assertFalse(m.generator.generator.training)
    def test_bad_shape_units_and_time(self):
        for change in ({'dt_seconds':1.},{'output_unit':'pixels'},{'latent_dim':0}):
            with self.assertRaises(ValueError):FrozenGaussianGenerator(FixtureGenerator(),CONTRACT|change)
        with self.assertRaises(ValueError):self.pipeline()(torch.ones(2,20,2,dtype=torch.double))
    def test_quality_numpy_and_autograd(self):
        p=np.cumsum(np.random.default_rng(12).normal(size=(2,8,2))*.08,1);y=np.random.default_rng(21).normal(size=(2,20,12,2))*.1+p[:,-1,None,None]
        c=dict(vmax=.8,amax=2.,joinmax=.5,sigma_v=.9,sigma_a=3.,sigma_join=.6)
        paths=torch.tensor(y,requires_grad=True);q=kinematic_torch(torch.tensor(p),paths,c)
        np.testing.assert_allclose(q.detach(),kinematic(p,y,c)['qkin'],rtol=2e-12,atol=2e-12)
        actual=expected_cardinality(paths,q,.2);np.testing.assert_allclose(actual.detach(),dpp(y,q.detach().numpy(),.2,1e-6)['ec'],atol=2e-12,rtol=0)
        g=torch.autograd.grad(actual.sum(),paths)[0];self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)
    def test_fixed_double_finite_difference(self):
        m=self.pipeline();p=torch.randn(2,8,2,dtype=torch.double)*.1;param=m.sampler.net[-1].weight
        direction=torch.sin(torch.arange(param.numel(),dtype=torch.double)+1).reshape_as(param);direction/=direction.norm()
        def value():
            o=m(p);return ec_from_weights(o['paths'],torch.ones((2,20),dtype=torch.double),.4).sum()
        ref=float((torch.autograd.grad(value(),param)[0]*direction).sum());original=param.detach().clone()
        for eps in (1e-4,1e-5):
            with torch.no_grad():param.copy_(original+eps*direction)
            plus=float(value().detach())
            with torch.no_grad():param.copy_(original-eps*direction)
            minus=float(value().detach())
            self.assertLess(abs((plus-minus)/(2*eps)-ref),1e-6+1e-3*abs(ref))
        with torch.no_grad():param.copy_(original)
    def test_unknown_map_never_joint_one(self):
        z=torch.zeros(2,20,3)
        for arm in ('B1_scene','B3_joint'):
            with self.assertRaises(ValueError):quality_weights(arm,z,qkin=torch.ones(2,20))
        self.assertTrue(torch.equal(quality_weights('B4_uniform',z),torch.ones(2,20)))
    def test_original_radius_is_r_not_q(self):
        z=torch.zeros(1,20,3);z[0,0,0]=2
        r=quality_weights('B0_original',z,radius=1.);self.assertAlmostEqual(float(r[0,0]),np.exp(-3),places=7)
        q=torch.full((1,20),.25);r=quality_weights('B2_kin',z,qkin=q);self.assertAlmostEqual(float(r[0,0]**2),1e-6+(1-1e-6)*.25,places=7)
    def test_metric_resampling_and_unknown(self):
        labels=np.ones((61,61),np.uint8);labels[:,:20]=0;labels[25:28,25:28]=2
        a=build_metric_map(labels,np.diag([.1,.1,1]),{'0':'obstacle','1':'free','2':'unknown'})
        self.assertTrue(np.isnan(a['distance_metres'][26,26]));self.assertFalse(a['known'][26,26]);self.assertAlmostEqual(float(a['metres_per_cell']),.1)
        with self.assertRaises(ValueError):build_metric_map(labels,np.eye(3),{'0':'obstacle','1':'free'})
    def test_calibration_residual_and_degeneracy(self):
        xy=np.array([[0,0],[1,0],[0,1],[1,1],[2,1]],float);c=dict(image_xy_to_world_metres=np.eye(3).tolist(),holdout_image_xy=xy.tolist(),holdout_world_metres=xy.tolist(),independent_of_fit_and_future=True,point_source='synthetic known fixture')
        self.assertEqual(validate_calibration(c)[1]['rms_metres'],0)
        with self.assertRaises(ValueError):validate_calibration(c|{'holdout_world_metres':(xy+.5).tolist()})
        with self.assertRaises(ValueError):validate_calibration(c|{'holdout_image_xy':[[1,1]]*5})
    def test_map_registry_missing_and_wrong_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'registry.json';write(p,{'records':{}});self.assertIsNone(MapRegistry(p).load('missing'))
    def test_scene_value_gradient_unknown(self):
        a=self.arrays();p0=torch.tensor([1.15,1.25],dtype=torch.double)
        y=torch.stack([torch.linspace(1.23,2.37,12),torch.full((12,),1.25)],-1).double()[None].repeat(20,1,1).requires_grad_()
        qt,coverage=scene_torch(p0,y,a)
        ref=scene_score(p0.numpy(),y[0].detach().numpy(),a['distance_metres'],a['known'],a['world_to_grid'])
        self.assertAlmostEqual(float(qt[0].detach()),ref['qscene'],places=12);self.assertEqual(float(coverage[0]),1.)
        grad=torch.autograd.grad(qt.sum(),y)[0];self.assertGreater(float(grad.abs().sum()),0)
        a['known'][:]=False;q,f=scene_torch(p0,y,a);self.assertTrue(torch.isnan(q).all());self.assertEqual(float(f.sum()),0)
    def test_untrained_generator_admission_denied(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'contract.json';write(p,CONTRACT|{'status':'FRESH_RANDOM_FIXTURE'})
            with self.assertRaises(ValueError):load_generator(p,lambda c:FixtureGenerator())
    def test_gpu8_fixture_if_available(self):
        if not torch.cuda.is_available():self.skipTest('No CUDA')
        m=self.pipeline().cuda();p=torch.randn(8,8,2,dtype=torch.double,device='cuda');o=m(p);ec=ec_from_weights(o['paths'],torch.ones(8,20,device='cuda',dtype=torch.double),.4);ec.mean().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in m.sampler.parameters()));self.assertTrue(all(x.grad is None for x in m.generator.parameters()))

if __name__=='__main__':
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestInterfaces));print(stream.getvalue())
    write(ROOT/'tests'/f'{time.time_ns()}.json',dict(status='PASS_ENGINEERING_FIXTURES_ONLY' if result.wasSuccessful() else 'FAIL_ENGINEERING_FIXTURES_ONLY',tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),source_files=seal(ROOT.glob('*.py')),log=stream.getvalue(),no_real_model_pretraining_or_scientific_experiment=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
