"""Certified metric-grid maps; no future/trajectory heatmaps are accepted as maps."""
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from interface_common import read,write,check_files,sha

def project(x,H):
    a=np.c_[np.asarray(x,np.float64).reshape(-1,2),np.ones(np.asarray(x).size//2)]@np.asarray(H,np.float64).T
    if not np.isfinite(a).all() or (a[:,2]==0).any():raise ValueError('HOMOGRAPHY_HORIZON')
    return a[:,:2]/a[:,2:]

def validate_calibration(c):
    H=np.asarray(c['image_xy_to_world_metres'],np.float64)
    image=np.asarray(c['holdout_image_xy'],np.float64);world=np.asarray(c['holdout_world_metres'],np.float64)
    if H.shape!=(3,3) or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-15:raise ValueError('SINGULAR_H')
    if image.ndim!=2 or image.shape[1]!=2 or len(image)<5 or world.shape!=image.shape:raise ValueError('FIVE_INDEPENDENT_CALIBRATION_POINTS_REQUIRED')
    if not c.get('independent_of_fit_and_future',False) or not c.get('point_source'):raise ValueError('CONTROL_POINT_PROVENANCE')
    if len(np.unique(image,axis=0))<5 or np.linalg.matrix_rank(image-image.mean(0))<2:raise ValueError('DEGENERATE_CONTROL_POINTS')
    err=np.linalg.norm(project(image,H)-world,axis=1)
    if np.sqrt(np.mean(err**2))>.10 or err.max()>.20:raise ValueError('CALIBRATION_RESIDUAL')
    return H,dict(rms_metres=float(np.sqrt(np.mean(err**2))),max_metres=float(err.max()),N=len(err))

def build_metric_map(labels,H_image_to_world,class_policy,resolution=.10):
    """Nearest class resampling onto an isotropic world grid before metric EDT.

    Unlike pixel EDT times one scale, this handles perspective correctly. An
    unknown-distance bound excludes cells where an unobserved class could be
    closer than the closest observed opposite-class boundary.
    """
    labels=np.asarray(labels)
    if labels.ndim!=2 or not 0<resolution<=.10:raise ValueError('MAP_SHAPE_OR_RESOLUTION')
    policy={int(k):v for k,v in class_policy.items()}
    if any(v not in ('free','obstacle','unknown') for v in policy.values()):raise ValueError('CLASS_POLICY')
    if not set(np.unique(labels)).issubset(policy):raise ValueError('UNDOCUMENTED_SEMANTIC_CLASS')
    h,w=labels.shape
    corner_h=np.c_[np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],float),np.ones(4)]@np.asarray(H_image_to_world,np.float64).T
    if not np.isfinite(corner_h).all() or not ((corner_h[:,2]>0).all() or (corner_h[:,2]<0).all()):raise ValueError('HOMOGRAPHY_HORIZON_CROSSES_IMAGE')
    bounds=project([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],H_image_to_world)
    origin=bounds.min(0);shape=np.ceil((bounds.max(0)-origin)/resolution).astype(int)+1
    if int(np.prod(shape))>4_000_000:raise ValueError('METRIC_GRID_TOO_LARGE_REVIEW_BOUNDS')
    rows,cols=np.indices((shape[1],shape[0]));xy=origin+np.stack([cols,rows],-1).reshape(-1,2)*resolution
    pixel=project(xy,np.linalg.inv(H_image_to_world));x=np.rint(pixel[:,0]).astype(int);y=np.rint(pixel[:,1]).astype(int)
    inside=(x>=0)&(y>=0)&(x<w)&(y<h);kind=np.zeros(len(x),np.int8)
    for value,meaning in policy.items():
        mask=np.zeros(len(x),bool);mask[inside]=labels[y[inside],x[inside]]==value
        kind[mask]={'unknown':0,'free':1,'obstacle':-1}[meaning]
    kind=kind.reshape(rows.shape);known=kind!=0;free=kind==1;obstacle=kind==-1
    if not free.any() or not obstacle.any():raise ValueError('NO_OBSERVED_BOUNDARY')
    to_obs=distance_transform_edt(~obstacle)*resolution;to_free=distance_transform_edt(~free)*resolution
    # Padding treats outside the image/grid as unknown, never as obstacle.
    to_unknown=distance_transform_edt(np.pad(known,1,constant_values=False))[1:-1,1:-1]*resolution
    opposite=np.where(free,to_obs,to_free)
    reliable=known&(opposite<to_unknown)
    distance=np.where(free,to_obs,-to_free);distance[~reliable]=np.nan
    world_to_grid=np.array([[1/resolution,0,-origin[0]/resolution],[0,1/resolution,-origin[1]/resolution],[0,0,1.]])
    return dict(distance_metres=distance,known=reliable,world_to_grid=world_to_grid,semantic_kind=kind,metres_per_cell=np.array(resolution))

class MapRegistry:
    def __init__(self,path):
        self.path=Path(path);self.registry=read(path);self.cache={}
    def load(self,recording):
        record=self.registry.get('records',{}).get(recording)
        if record is None:return None
        if record.get('status')!='ADMITTED':return None
        check_files(record['evidence_files']);file=Path(record['metric_sdf_npz'])
        if sha(file)!=record['metric_sdf_sha256']:raise ValueError('MAP_SHA')
        if record.get('recording_id')!=recording or record.get('unit')!='metres' or not record.get('no_future_labels'):
            raise ValueError('MAP_RECORDING_OR_UNIT_OR_FUTURE')
        if recording not in self.cache:
            with np.load(file,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            if a['distance_metres'].shape!=a['known'].shape or a['known'].dtype!=np.bool_:raise ValueError('SDF_SHAPE_MASK')
            if not np.isfinite(a['distance_metres'][a['known']]).all():raise ValueError('KNOWN_SDF_NONFINITE')
            if float(a['metres_per_cell'])>.10 or float(a['metres_per_cell'])<=0:raise ValueError('MAP_RESOLUTION')
            G=a['world_to_grid'];step=float(a['metres_per_cell'])
            if G.shape!=(3,3) or not np.allclose(G[:2,:2],np.eye(2)/step,rtol=0,atol=1e-10) or not np.array_equal(G[2],[0,0,1]):raise ValueError('NOT_METRIC_CARTESIAN_GRID')
            self.cache[recording]=a
        return self.cache[recording]
    def score(self,ids,past,paths):
        import sys
        from interface_common import STAGE_A
        sys.path.insert(0,str(STAGE_A));from scoring import scene_score
        ids=np.asarray(ids);past=np.asarray(past);paths=np.asarray(paths)
        if paths.shape!=(len(ids),20,12,2) or past.shape!=(len(ids),8,2):raise ValueError('SCENE_POPULATION_SHAPE')
        q=np.full((len(ids),20),np.nan);fraction=np.zeros_like(q);violation=np.full(q.shape,-1,np.int8)
        for i,identity in enumerate(ids):
            a=self.load(str(identity).split('|')[0])
            if a is None:continue
            for k in range(20):
                v=scene_score(past[i,-1],paths[i,k],a['distance_metres'],a['known'],a['world_to_grid'])
                fraction[i,k]=v['known_fraction']
                if v['qscene'] is not None:q[i,k]=v['qscene'];violation[i,k]=int(v['violation'])
        return dict(qscene=q,known_fraction=fraction,scene_violation=violation)

def admit(annotation_json,output_dir):
    from PIL import Image
    src=Path(annotation_json);c=read(src);check_files(c['evidence_files'])
    for key in ('semantic_class_documentation','pedestrian_rule_documentation','recording_match_evidence','source_permission','static_annotation_no_future_evidence'):
        file=c.get(key)
        if not file or str(Path(file).resolve()) not in c['evidence_files']:raise ValueError('MISSING_EVIDENCE '+key)
    if not c.get('no_future_labels'):raise ValueError('FUTURE_BASED_MAP_REJECTED')
    H,cal=validate_calibration(c['calibration'])
    labels=np.asarray(Image.open(c['labels_png']))
    if sha(c['labels_png'])!=c['labels_sha256']:raise ValueError('LABEL_SHA')
    a=build_metric_map(labels,H,c['class_policy'],c.get('resolution_metres',.10))
    out=Path(output_dir);out.mkdir(exist_ok=False,parents=True);np.savez_compressed(out/'metric_sdf.npz',**a)
    record=dict(status='ADMITTED',recording_id=c['recording_id'],unit='metres',no_future_labels=True,calibration=cal,metric_sdf_npz=str((out/'metric_sdf.npz').resolve()),metric_sdf_sha256=sha(out/'metric_sdf.npz'),evidence_files=c['evidence_files']|{str(src.resolve()):sha(src)})
    write(out/'record.json',record);return record

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--annotation',required=True);p.add_argument('--output',required=True);a=p.parse_args();admit(a.annotation,a.output)
