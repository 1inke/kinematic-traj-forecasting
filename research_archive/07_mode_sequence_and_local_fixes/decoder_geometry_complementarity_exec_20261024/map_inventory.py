"""Existing static-map admission only; no inferred maps or remote acquisition."""
from common import *
from collections import Counter
def inventory():
    root=Path('<LOCAL_DESKTOP>/DRSE');dataset=root/'ASTRA/datasets/eth_ucy'
    records=read(PREVIOUS/'direction_B_scene_prior/records_20261024.json')['records']
    numbers=Counter((r['sequence'],r['split']) for r in records)
    # These are candidates, not an assertion that campuses/recordings share maps.
    image_names={'biwi_hotel':('hotel',None),'students001':(None,None),'students003':('students003','students_003.jpg'),
                 'crowds_zara01':('zara01','crowds_zara01.jpg'),'crowds_zara02':('zara02','crowds_zara02.jpg'),'crowds_zara03':(None,None)}
    reg=[]
    for recording,(folder,reference) in image_names.items():
        candidates=[]
        if reference:
            p=root/'crowd-data/crowds/data'/reference
            if p.exists():candidates.append(p)
        if folder:
            candidates+=sorted((dataset/'imgs'/folder).glob('*.jpg'))[:1]
        hp=dataset/'homography'/('H_'+folder+'.txt') if folder else None
        H=np.loadtxt(hp) if hp and hp.exists() else None
        reg.append(dict(recording_id=recording,n_FIT=numbers[(recording,'FIT')],n_inner=numbers[(recording,'inner')],
            image_sha=[dict(path=str(p),sha256=sha(p)) for p in candidates],annotation_sha=None,
            source_and_license=dict(image_source='existing local dataset / crowd-data material; provenance candidate only',dataset_license_verified=False,
                homography_source=str(hp) if H is not None else None,homography_sha=sha(hp) if H is not None else None,
                repository_code_LICENSE='MIT .homography_source/LICENSE does not independently certify dataset image/annotation license'),
            map_date=None,world_unit='trajectory metres; map calibration NOT verified',H_image_to_world=None,H_world_to_image=None,
            existing_H_candidate=H.tolist() if H is not None else None,H_direction_verified=False,
            valid_polygon=None,unknown_mask=None,annotation_method=None,
            fitting_control_points=0,independent_holdout_points=0,holdout_RMS_metres=None,holdout_max_metres=None,effective_resolution_metres=None,
            admission='MAP_CALIBRATION_AND_ANNOTATION_UNRESOLVED',
            missing=['independent static tri-state annotation','6 fit and5 heldout identifiable ground control points','H direction/units verification','valid polygon and unknown mask','<=0.10m effective resolution','map provenance/license and map date'],
            complete_image_and_H_candidates=bool(candidates and H is not None)))
    scanned=[str(ROOT),str(root/'ASTRA/datasets/eth_ucy'),str(root/'ASTRA_ACCURACY_Q20/datasets/eth_ucy'),str(root/'crowd-data'),str(root/'eth_dataset'),str(root/'ewap_dataset_full'),str(root/'.homography_source')]
    rejected=[dict(path=str(ROOT/'SCENE_TOPOLOGY_SERIES/build_walkability_map.py'),reason='SDD different dataset and trajectory-derived weak positives/negatives; forbidden replacement'),
              dict(path=str(dataset/'labels'),reason='trajectory coordinate TXT, not static obstacle annotations'),
              dict(path=str(root/'.homography_source/LICENSE'),reason='code MIT not sufficient dataset image/label license proof')]
    archive=root/'ewap_dataset_full/ewap_dataset_full.tar';archive_candidates=[]
    if archive.exists():
        import tarfile
        with tarfile.open(archive) as tf:
            archive_candidates=[m.name for m in tf.getmembers() if m.isfile() and any(t in m.name.lower() for t in ('homograph','h.txt','reference','control','calib','license','mask'))]
    out=dict(status='STOP_MAP_ADMISSION_UNRESOLVED',records=reg,searched_roots=scanned,rejected_substitutes=rejected,
        archive_inspection=dict(path=str(archive),sha256=sha(archive) if archive.exists() else None,candidate_names=archive_candidates),
        annotation_or_control_point_assets_found=False,all_required_recordings_qualified=False,known_map_IDs=0,
        registered_IDs=len(records),unknown_map_IDs=len(records),unknown_IDs_by_recording={s:[r['sample_id'] for r in records if r['sequence']==s] for s in image_names},
        Q1_probe='NOT_EXECUTED_MAP_ADMISSION_FAILED',Q2='NOT_TRIGGERED_MAP_ADMISSION_FAILED',
        statement='Local searched assets lack required certificates; this is not proof no static map could ever be obtained. No downloads, trajectory label maps, H tuning or scene substitution.')
    write(HERE/'q1_map_inventory/MAP_INVENTORY.json',out);return out
if __name__=='__main__':inventory()
