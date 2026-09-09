import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from processing.rgb_recovery.dataset import Settings,inspect_dataset,paired_images,rigid
from processing.analysis_workflow import mapping,measure_landmarks


@pytest.fixture
def capture(tmp_path):
    root=tmp_path/'other_species_42';root.mkdir()
    (root/'rgb').mkdir();(root/'depth').mkdir()
    K=[[120,0,80],[0,120,60],[0,0,1]]
    (root/'kdc_intrinsics.txt').write_text(json.dumps(dict(K=K,dist=[0]*5,width=160,height=120)))
    positions={}
    texture=np.random.default_rng(9).integers(0,256,(120,200,3),dtype=np.uint8)
    for i in range(8):
        token=100+i*7;positions[str(token)]=i*.01
        cv2.imwrite(str(root/'rgb'/f'{token}.png'),texture[:,i*3:i*3+160])
        cv2.imwrite(str(root/'depth'/f'{token}.png'),np.full((120,160),500,np.uint16))
    (root/'session.json').write_text(json.dumps(dict(frame_positions=positions,depth_scale_units_per_m=1000)))
    return root


def test_profile_uses_actual_ids_resolution_and_scale(capture):
    profile=inspect_dataset(capture,Settings(camera_axis='x',max_frames=8))
    assert profile['width']==160 and profile['height']==120
    assert profile['depth_scale']==1000
    assert profile['reference'] in [100+i*7 for i in range(8)]
    assert len(profile['anchors'])>=3
    assert all(len(profile['neighbours'][str(k)])>=2 for k in profile['anchors'])


def test_scale_is_not_guessed(capture):
    s=json.loads((capture/'session.json').read_text());s.pop('depth_scale_units_per_m')
    (capture/'session.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Depth units are missing'):
        inspect_dataset(capture,Settings())


def test_equal_counts_wrong_frame_ids_rejected(capture):
    (capture/'depth/100.png').rename(capture/'depth/999.png')
    with pytest.raises(ValueError,match='match exactly'):paired_images(capture)


def test_wrong_resolution_rejected(capture):
    intr=json.loads((capture/'kdc_intrinsics.txt').read_text());intr['width']=320
    (capture/'kdc_intrinsics.txt').write_text(json.dumps(intr))
    with pytest.raises(ValueError,match='Calibration resolution'):inspect_dataset(capture,Settings())


def test_zero_motion_rejected(capture):
    s=json.loads((capture/'session.json').read_text());s['frame_positions']={k:0 for k in s['frame_positions']}
    (capture/'session.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Insufficient camera translation'):inspect_dataset(capture,Settings())


def test_nonrigid_pose_rejected():
    t=np.eye(4);t[0,0]=-1
    with pytest.raises(ValueError,match='rigid'):rigid(t)


def test_blank_texture_requires_explicit_motion(capture):
    for file in (capture/'rgb').glob('*.png'):cv2.imwrite(str(file),np.zeros((120,160,3),np.uint8))
    with pytest.raises(ValueError,match='direction is ambiguous'):inspect_dataset(capture,Settings())


def test_unique_explicit_specimen_mapping():
    assert mapping('1:2 2:1')=={1:2,2:1}
    for text in ('','1:1 2:1','1:1 1:2','0:1'):
        with pytest.raises(ValueError):mapping(text)


def test_landmarks_calibrated_units_and_separate_rows(capture,tmp_path):
    config=tmp_path/'leaves.json'
    config.write_text(json.dumps(dict(depth_scale_units_per_m=1000,leaves=[dict(plant_id='plant_a',leaf_id=1,frame=100,points=[[50,60],[74,60],[60,50],[60,62]],manual_length_mm=100,manual_width_mm=50)])))
    output=tmp_path/'out';output.mkdir()
    rows=measure_landmarks(capture,config,output)
    assert [r['trait'] for r in rows]==['leaf_length','leaf_width']
    assert [r['rgbd_projected_mm'] for r in rows]==pytest.approx([100,50])
    assert all(r['absolute_percent_error']<1e-10 for r in rows)


def test_missing_endpoint_uses_recorded_local_depth_with_evidence(capture,tmp_path):
    image=cv2.imread(str(capture/'depth/100.png'),-1);image[60,50]=0;cv2.imwrite(str(capture/'depth/100.png'),image)
    config=tmp_path/'leaves.json';config.write_text(json.dumps(dict(depth_scale_units_per_m=1000,leaves=[dict(plant_id=1,leaf_id=1,frame=100,points=[[50,60],[74,60],[60,50],[60,62]],manual_length_mm=100,manual_width_mm=50)])))
    rows=measure_landmarks(capture,config,tmp_path)
    assert rows[0]['depth_search_radius_px']==3
    assert rows[0]['rgbd_projected_mm']==pytest.approx(100)


def test_no_observed_depth_rejects_leaf(capture,tmp_path):
    for file in (capture/'depth').glob('*.png'):cv2.imwrite(str(file),np.zeros((120,160),np.uint16))
    config=tmp_path/'leaves.json';config.write_text(json.dumps(dict(depth_scale_units_per_m=1000,leaves=[dict(plant_id=1,leaf_id=1,frame=100,points=[[50,60],[74,60],[60,50],[60,62]],manual_length_mm=100,manual_width_mm=50)])))
    with pytest.raises(ValueError,match='No adjacent frame'):measure_landmarks(capture,config,tmp_path)


def test_absent_encoder_selects_existing_sensor_route(capture):
    session=json.loads((capture/'session.json').read_text())
    session.pop('frame_positions')
    (capture/'session.json').write_text(json.dumps(session))
    cfg=inspect_dataset(capture,Settings(max_frames=8))
    assert cfg['method']=='sensor'
    assert len(cfg['frames'])==8


def test_raised_neutral_board_is_not_mistaken_for_plant(capture,tmp_path):
    from processing.rgb_recovery.helpers import read_frame
    im=np.full((120,160,3),100,np.uint8)
    depth=np.full((120,160),1200,np.uint16)
    im[10:110,15:145]=220;depth[10:110,15:145]=800
    im[40:80,60:100]=[30,140,30];depth[40:80,60:100]=650
    cv2.imwrite(str(capture/'rgb/100.png'),im)
    cv2.imwrite(str(capture/'depth/100.png'),depth)
    cfg=dict(frames=[dict(frame=100,rgb=str(capture/'rgb/100.png'),depth=str(capture/'depth/100.png'))],depth_scale=1000,near=.1,far=1.5,foreground='auto',tolerance=.004)
    (tmp_path/'profile.json').write_text(json.dumps(cfg))
    K=np.array([[120.,0,80],[0,120,60],[0,0,1]])
    maps=cv2.initUndistortRectifyMap(K,np.zeros(5),None,K,(160,120),cv2.CV_32FC1)
    result=read_frame(tmp_path,100,K,maps)
    assert .7<result['foreground_limit']<.8
    assert result['mask'][60,80]
    assert not result['mask'][30,35]
    assert not result['mask'][5,5]


def test_sensor_adapter_runs_existing_icp(capture,tmp_path):
    from processing.rgb_recovery.sensor import run
    session=json.loads((capture/'session.json').read_text());session.pop('frame_positions')
    (capture/'session.json').write_text(json.dumps(session))
    rgb=cv2.imread(str(capture/'rgb/100.png'))
    for path in (capture/'rgb').glob('*.png'):cv2.imwrite(str(path),rgb)
    cfg=inspect_dataset(capture,Settings(max_frames=8))
    run(cfg,tmp_path/'sensor')
    result=json.loads((tmp_path/'sensor/result/summary.json').read_text())
    assert result['points']>100
    assert result['minimum_support_views'] is None
    assert (tmp_path/'sensor/result/plant_upright.ply').is_file()
