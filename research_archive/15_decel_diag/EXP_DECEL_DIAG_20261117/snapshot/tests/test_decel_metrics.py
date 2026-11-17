import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from decel_metrics import compute,calibration,decide
from run_decel_diag import align


def fixture(n=4):
    p=np.zeros((n,8,2));p[:,:,0]=np.arange(8)*.4
    y=np.zeros((n,12,2));y[:,:,0]=2.8+np.arange(1,13)*.2
    x=np.broadcast_to(y[:,None],(n,20,12,2)).copy()
    x[:,1:,:,0]+=1
    z=np.tile(np.arange(20),(n,1)).astype(float)
    return p,y,x,z


class MetricsTests(unittest.TestCase):
    def test_known_geometry_and_rank(self):
        w=compute(*fixture())
        np.testing.assert_allclose(w['r_g'],.5)
        np.testing.assert_allclose(w['best_prog_error_t12'],0)
        np.testing.assert_equal(w['best_rank_t12'],20)
        np.testing.assert_equal(w['top1_overshoot_t12'],True)
        np.testing.assert_equal(w['envelope_covered_t12'],True)
        self.assertEqual(decide(w,np.repeat('deceleration',4))['main_class'],'B')

    def test_rotation_translation_invariance(self):
        p,y,x,z=fixture();theta=.73
        r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        a=compute(p,y,x,z);b=compute(p@r.T+17,y@r.T+17,x@r.T+17,z)
        for k in ('r_g','candidate_r','P_dec','top1_prog_error_t12','best_lat_error_t12'):
            np.testing.assert_allclose(a[k],b[k],atol=1e-12)

    def test_future_cannot_change_history_or_predictions(self):
        p,y,x,z=fixture();a=compute(p,y,x,z);b=compute(p,y+10,x,z)
        for k in ('history_ratio','history_bin','candidate_r','P_dec','top1_index'):
            np.testing.assert_equal(a[k],b[k])

    def test_calibration_edges_and_no_selection_on_future(self):
        w={'P_dec':np.array([0,.2,.4,.6,.8,1.]),'true_deceleration':np.array([0,1,0,1,0,1],bool)}
        b=calibration(w,np.ones(6,bool));self.assertEqual(sum(x['n'] for x in b),6)
        self.assertEqual(b[-1]['n'],2)
        self.assertEqual(b[0]['actual_deceleration_rate'],0)

    def test_mean_direction_threshold_and_ties(self):
        p,y,x,z=fixture();p[0]*=.02;z[:]=0
        w=compute(p,y,x,z)
        self.assertFalse(w['direction_valid'][0]);self.assertEqual(w['top1_index'][1],0)
        self.assertEqual(w['best_rank_t12'][1],1)
        self.assertEqual(decide(w,np.repeat('steady',4))['label'],'NOT_ESTIMABLE')

    def test_decision_thresholds(self):
        p,y,x,z=fixture(n=10);w=compute(p,y,x,z);labels=np.repeat('deceleration',10)
        w['all_overshoot_t12'][:]=False;w['longitudinal_reachable_t12'][:]=True
        w['all_overshoot_t12'][:3]=True
        self.assertEqual(decide(w,labels)['main_class'],'A+B')
        w['all_overshoot_t12'][:]=False;w['top1_overshoot_t12'][:]=False
        self.assertEqual(decide(w,labels)['main_class'],'NEITHER')
        w['longitudinal_reachable_t12'][:5]=False
        self.assertEqual(decide(w,labels)['main_class'],'A')

    def test_alignment_and_failure(self):
        p,y,x,z=fixture();data=dict(ids=np.arange(4),past=p[:,None],future=y,group=np.zeros(4))
        order=np.array([3,1,0,2]);pred=dict(ids=order,paths=x[order],logits=z[order],future=y[order],past=p[order])
        xx,zz,n=align(pred,data);np.testing.assert_equal(xx,x);self.assertEqual(n,3)
        pred['future'][0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'FUTURE'):align(pred,data)
        pred['ids'][0]=pred['ids'][1]
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):align(pred,data)


if __name__=='__main__':unittest.main()
