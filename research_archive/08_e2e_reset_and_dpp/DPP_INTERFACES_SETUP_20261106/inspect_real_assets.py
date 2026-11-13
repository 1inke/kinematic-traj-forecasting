"""Read maps and OBSERVATIONS only. No held GT, model, training or calibration fit."""
from interface_common import *
from scene_interface import project
import numpy as np
from PIL import Image

def main():
    download=read(ROOT/'YNET_DOWNLOAD.json');check_files({r['path']:r['sha256'] for r in download['files']})
    aliases={'biwi_hotel':'hotel','crowds_zara01':'zara1','crowds_zara02':'zara2','crowds_zara03':'zara3','students001':'students001','students003':'students003','uni_examples':'uni_examples'}
    files=download['files'];records={}
    for recording,alias in aliases.items():
        hfile=next(r for r in files if r['archive_path'].endswith('/'+alias+'_H.txt'))
        image=next(r for r in files if r['archive_path'].endswith('/'+alias+'/oracle.png'))
        H=np.loadtxt(hfile['path']);swap=np.array([[0,1,0],[1,0,0],[0,0,1.]]) if alias=='hotel' else np.eye(3)
        Hxy=H@swap;labels=np.asarray(Image.open(image['path']));unique,counts=np.unique(labels,return_counts=True)
        rows={}
        for split in ('fit','dev'):
            file=ROOT.parent/'E2E_MOTION_RESEARCH_RESET_EXEC_20261102/C0'/f'{split}.npz'
            with np.load(file,allow_pickle=False) as f:
                # Only IDs and observed past are accessed, never future/group.
                ids=f['ids'];mask=np.array([str(i).split('|')[0]==recording for i in ids]);past=f['past'][mask,0]
            xy=project(past.reshape(-1,2),np.linalg.inv(Hxy));x=np.rint(xy[:,0]).astype(int);y=np.rint(xy[:,1]).astype(int)
            inside=(x>=0)&(y>=0)&(x<labels.shape[1])&(y<labels.shape[0]);values,number=np.unique(labels[y[inside],x[inside]],return_counts=True)
            rows[split]=dict(windows=int(mask.sum()),observed_points=len(x),projected_inside=int(inside.sum()),label_counts_on_observed_points={str(v):int(n) for v,n in zip(values,number)},warning='OBSERVATION_OVERLAY_IS_NOT_INDEPENDENT_CALIBRATION_OR_CLASS_POLICY_PROOF')
        records[recording]=dict(status='OFFICIAL_ASSETS_PRESENT_NOT_SCIENTIFICALLY_ADMITTED',recording_id=recording,official_scene_alias=alias,labels_png=image['path'],labels_sha256=image['sha256'],labels_histogram={str(v):int(n) for v,n in zip(unique,counts)},homography_txt=hfile['path'],image_xy_to_world_candidate=Hxy.tolist(),hotel_swap_from_official_image2world=True if alias=='hotel' else False,observed_only_overlay=rows,missing=['verified pedestrian semantic 0/1 class meaning and independent static annotation provenance','independent metric calibration/valid domain certificate for exact recording','source usage permission scope'],special_case='SINGLE_LABEL_NO_OBSERVED_BOUNDARY' if len(unique)<2 else None,no_future_read=True)
    write(ROOT/'REAL_ASSET_INSPECTION.json',dict(status='OFFICIAL_ASSETS_FOUND_SCIENTIFIC_ADMISSION_PENDING',records=records,download_receipt_sha=sha(ROOT/'YNET_DOWNLOAD.json'),no_GT_in_map_or_class_policy=True,no_existing_experiment_source_mutation=True))
    write(ROOT/'map_registry.json',dict(status='ASSET_PRESENT_ADMISSION_PENDING',records={k:dict(status=v['status'],recording_id=k,labels_png=v['labels_png'],labels_sha256=v['labels_sha256'],missing=v['missing'],class_policy=None,metric_sdf_npz=None) for k,v in records.items()}))
    print({k:v['observed_only_overlay'] for k,v in records.items()})

if __name__=='__main__':main()
