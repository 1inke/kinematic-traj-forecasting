"""Synthetic-only tests. Never opens C0, model predictions or any checkpoint."""
import unittest,ast,io,contextlib,time
from pathlib import Path
import numpy as np
from common import RUN,write,sha,verify,authorize
from scoring import *
from independent import score_loop,spectral_ec_loop,quality_loop

class SyntheticTests(unittest.TestCase):
    @staticmethod
    def fixture():
        t=np.arange(1,13)[None,None,:,None]
        y=np.zeros((2,20,12,2));y[:,:,:,0]=t[...,0]*.1*np.arange(1,21)[None,:,None]/10
        y[:,:,:,1]=np.sin(t[...,0]/3)*np.arange(20)[None,:,None]/30
        p=np.zeros((2,8,2));p[:,:,0]=np.arange(-7,1)[None,:]*.1
        return p,y
    def test_independent_scores_validity(self):
        p,y=self.fixture();c=dict(vmax=1.,amax=2.,joinmax=.8,sigma_v=.3,sigma_a=.7,sigma_join=.4)
        a=kinematic(p,y,c);b,v=score_loop(p,y,c)
        np.testing.assert_allclose(a['qkin'],b,rtol=0,atol=2e-12);np.testing.assert_array_equal(a['kin_valid'],v)
    def test_raw_penalty_independence(self):
        p,y=self.fixture();c=dict(vmax=.2,amax=.3,joinmax=.1,sigma_v=.3,sigma_a=.7,sigma_join=.4)
        actual=kinematic(p,y,c);q,v,raw=score_loop(p,y,c,return_details=True)
        for key,value in raw.items():np.testing.assert_allclose(actual[key],value,rtol=2e-12,atol=2e-12)
    def test_spectral_marginal_independence(self):
        _,y=self.fixture();q=np.linspace(.2,.8,40).reshape(2,20)
        a=dpp(y,q,.7,1e-6);ec,m=spectral_ec_loop(y,q,.7,1e-6,return_marginal=True)
        np.testing.assert_allclose(a['marginal'],m,rtol=0,atol=2e-12)
    def test_copies_rankone_no_jitter(self):
        y=np.zeros((1,20,12,2));a=dpp(y,np.ones((1,20)),1,1e-6)
        self.assertAlmostEqual(a['ec'][0],20/21,places=12)
        b=dpp(y,np.zeros((1,20)),1,1e-6)
        self.assertAlmostEqual(b['ec'][0],20e-6/(1+20e-6),places=12)
    def test_separated_uniform(self):
        y=np.zeros((1,20,12,2));y[0,:,:,0]=np.arange(20)[:,None]*100
        self.assertAlmostEqual(dpp(y,np.ones((1,20)),1,1e-6)['ec'][0],10.,places=12)
    def test_spectral_and_monotonic(self):
        p,y=self.fixture();q=np.linspace(.1,.9,40).reshape(2,20)
        a=dpp(y,q,.7,1e-6);b=spectral_ec_loop(y,q,.7,1e-6)
        np.testing.assert_allclose(a['ec'],b,rtol=0,atol=2e-12)
        self.assertTrue(np.all(a['ec']<=dpp(y,np.ones_like(q),.7,1e-6)['ec']+2e-12))
    def test_logquality_fixed_FD(self):
        _,y=self.fixture();q=np.linspace(.2,.8,40).reshape(2,20);epsq=1e-6;k=.7
        S=np.exp(-k*pair_squared(y));qt=epsq+(1-epsq)*q;r=np.sqrt(qt);L=S*r[:,:,None]*r[:,None]
        inv=np.linalg.solve(np.eye(20)+L,np.broadcast_to(np.eye(20),L.shape))
        grad=np.diagonal(L@inv@inv,axis1=1,axis2=2)*(1-epsq)*q/qt
        for direction in range(2):
            d=np.sin(np.arange(40)+direction+1).reshape(2,20);d=d/np.linalg.norm(d)
            target=float((grad*d).sum())
            for epsilon in (1e-4,1e-5):
                plus=dpp(y,q*np.exp(epsilon*d),k,epsq)['ec'].sum()
                minus=dpp(y,q*np.exp(-epsilon*d),k,epsq)['ec'].sum()
                self.assertLess(abs((plus-minus)/(2*epsilon)-target),1e-6+1e-3*abs(target))
    def test_permutation(self):
        p,y=self.fixture();q=np.linspace(.1,.9,40).reshape(2,20);order=np.arange(19,-1,-1)
        a=dpp(y,q,.7,1e-6);b=dpp(y[:,order],q[:,order],.7,1e-6)
        np.testing.assert_allclose(a['ec'],b['ec'],rtol=0,atol=2e-12)
        np.testing.assert_allclose(a['marginal'][:,order],b['marginal'],rtol=0,atol=2e-12)
    def test_sqrt_quality_and_radius(self):
        z=np.array([[0.,0.],[np.sqrt(2),0.]])
        np.testing.assert_allclose(original_radius_quality(z,1),[1,np.exp(-1)],rtol=0,atol=1e-15)
        y=np.zeros((1,20,12,2));q=np.full((1,20),.25)
        # sqrt quality makes rank-one eigenvalue20*.25, NOT20*.25**2.
        e=dpp(y,q,1,1e-12)['ec'][0];qt=1e-12+(1-1e-12)*.25
        self.assertAlmostEqual(e,20*qt/(1+20*qt),places=12)
    def test_valid_diversity_empty(self):
        _,y=self.fixture();v=np.zeros((2,20),bool);v[1,:2]=True
        a,f,n=asd_fsd(y,v);self.assertTrue(np.isnan(a[0]) and np.isnan(f[0]));self.assertEqual(n[0],0);self.assertEqual(n[1],1)
    def test_scene_unknown_and_segment_crossing(self):
        free=np.ones((101,101),bool);free[:,45:55]=False;known=np.ones_like(free)
        sdf=sdf_from_mask(free,known,.1);H=np.diag([10.,10.,1.])
        y=np.tile([8.,5.],(12,1))
        crossing=scene_score(np.array([2.,5.]),y,sdf,known,H)
        self.assertTrue(crossing['violation']);self.assertGreater(crossing['known_fraction'],0)
        unknown=scene_score(np.array([2.,5.]),y,sdf,np.zeros_like(known),H)
        self.assertIsNone(unknown['qscene']);self.assertIsNone(unknown['violation'])
        self.assertEqual(unknown['known_fraction'],0)
        with self.assertRaises(ValueError):sdf_from_mask(free,np.zeros_like(known),.1)
    def test_large_turn_not_angle_penalty(self):
        p,y=self.fixture();y[:,:,:,0]=0;y[:,:,:,1]=np.arange(1,13)[None,None,:]*.1
        c=dict(vmax=100.,amax=100.,joinmax=100.,sigma_v=1.,sigma_a=1.,sigma_join=1.)
        a=kinematic(p,y,c)
        np.testing.assert_array_equal(a['qkin'],np.ones((2,20)))
        np.testing.assert_allclose(turning(p,y),90,atol=1e-12)
    def test_bad_inputs(self):
        p,y=self.fixture()
        with self.assertRaises(ValueError):dpp(y,np.ones((2,19)),1,1e-6)
        with self.assertRaises(ValueError):dpp(y,np.full((2,20),-1.),1,1e-6)
        y[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):states(p,y)
    def test_default_entry_and_execute_denied(self):
        import run_stage_a,prepare,preflight,evaluate,review
        before=set(RUN.rglob('*'))
        with contextlib.redirect_stdout(io.StringIO()):
            for module in (run_stage_a,prepare,preflight,evaluate,review):module.main(False)
        self.assertEqual(before,set(RUN.rglob('*')))
        with self.assertRaises(PermissionError):run_stage_a.main(True)
    def test_all_python_syntax(self):
        for path in RUN.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue())
    report=RUN/'synthetic_tests'/f'{time.time_ns()}.json'
    write(report,dict(status='PASS_SYNTHETIC_ONLY' if r.wasSuccessful() else 'FAIL_SYNTHETIC_ONLY',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),no_real_dataset_or_checkpoint_opened=True,no_experiment_started=True,source_files={str(p):sha(p) for p in RUN.glob('*.py')},log=stream.getvalue()))
    print('SYNTHETIC_RECEIPT',report)
    raise SystemExit(0 if r.wasSuccessful() else 1)
