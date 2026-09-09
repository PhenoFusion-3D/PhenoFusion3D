"""Validate capture inputs before any expensive reconstruction or output writes."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass
class Settings:
    depth_scale: float = 0.0  # zero requires explicit session metadata
    near: float = 0.0
    far: float = 0.0
    max_frames: int = 44
    voxel: float = 0.0
    camera_axis: str = "auto"
    initial_poses: str = ""
    foreground: str = "auto"
    method: str = "auto"


def paired_images(root):
    root = Path(root)
    def collect(folder, prefix):
        result = {}
        for path in folder.glob('*.png'):
            match = re.fullmatch(rf'(?:{prefix}_)?(\d+)\.png', path.name, re.I)
            if match:
                token = int(match.group(1))
                if token in result:
                    raise ValueError(f'Duplicate {prefix} frame ID: {token}')
                result[token] = path
        return result
    rgb = collect(root / 'rgb' if (root / 'rgb').is_dir() else root, 'rgb')
    depth = collect(root / 'depth' if (root / 'depth').is_dir() else root, 'depth')
    # Flat folders must use prefixes; plain numbered files require separate folders.
    if not (root / 'rgb').is_dir():
        rgb = {k:v for k,v in rgb.items() if v.name.lower().startswith('rgb_')}
        depth = {k:v for k,v in depth.items() if v.name.lower().startswith('depth_')}
    if not rgb or rgb.keys() != depth.keys():
        raise ValueError('RGB/depth frame IDs must match exactly in paired folders or prefixed files.')
    return [(k, rgb[k], depth[k]) for k in sorted(rgb)]


def rigid(matrix):
    value = np.asarray(matrix, dtype=float)
    if (value.shape != (4,4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0,0,0,1])
            or not np.allclose(value[:3,:3].T @ value[:3,:3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3,:3]), 1, atol=1e-5)):
        raise ValueError('Camera poses must be finite rigid camera-to-reference 4x4 transforms.')
    return value


def inspect_dataset(root, settings):
    root = Path(root).resolve()
    if settings.max_frames < 8 or settings.max_frames > 64:
        raise ValueError('Select between 8 and 64 camera views.')
    if settings.foreground not in ('auto', 'depth', 'colour'):
        raise ValueError('Foreground must be auto, depth or colour.')
    pairs = paired_images(root)
    if len(pairs) < 8:
        raise ValueError('RGB recovery needs at least eight overlapping views with camera translation.')
    intrinsic_path = root / 'kdc_intrinsics.txt'
    intr = json.loads(intrinsic_path.read_text())
    K = np.asarray(intr['K'], dtype=float)
    if K.shape != (3,3) or not np.isfinite(K).all() or min(K[0,0], K[1,1]) <= 0 or not np.allclose(K[2], [0,0,1]):
        raise ValueError('Invalid camera intrinsics; supply the saved RGB calibration.')
    sample = cv2.imread(str(pairs[0][1]))
    if sample is None:
        raise ValueError('Cannot decode the first RGB frame.')
    height, width = sample.shape[:2]
    if (intr.get('width', width), intr.get('height', height)) != (width,height):
        raise ValueError('Calibration resolution differs from RGB images; recalibrate or use matching images.')
    dist = np.asarray(intr.get('dist', [0]*5), dtype=float)
    if dist.size not in (4,5,8,12,14) or not np.isfinite(dist).all():
        raise ValueError('Invalid distortion coefficients.')
    session_path = root / 'session.json'
    session = json.loads(session_path.read_text()) if session_path.is_file() else {}
    scale = settings.depth_scale or session.get('depth_scale_units_per_m', 0)
    if not scale and session.get('depth_scale_m'):
        scale = 1 / float(session['depth_scale_m'])
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Depth units are missing. Enter camera-reported raw units per metre; scale will not be guessed.')
    lookup = {k:(rgb,dep) for k,rgb,dep in pairs}
    use_sensor = settings.method=='sensor' or (settings.method=='auto' and not settings.initial_poses and not session.get('frame_positions'))
    if use_sensor:
        indices=np.unique(np.linspace(0,len(pairs)-1,min(settings.max_frames,len(pairs)),dtype=int))
        ids=[pairs[i][0] for i in indices];ref=ids[0]
        transforms={k:np.eye(4) for k in ids}
        pose_source='sensor-depth ICP; no external camera poses'
    elif settings.initial_poses:
        data = json.loads(Path(settings.initial_poses).read_text())
        rows = [r for r in data['frames'] if r.get('accepted', True)]
        ids = [int(r['frame']) for r in rows]
        if len(set(ids)) != len(ids) or any(k not in lookup for k in ids):
            raise ValueError('Pose frame IDs must be unique and match image file IDs.')
        if len(ids) < 8:
            raise ValueError('At least eight accepted camera poses are required.')
        ref = int(data['reference'])
        if ref not in ids:
            raise ValueError('Reference camera is missing from accepted poses.')
        transforms = {int(r['frame']):rigid(r['transform']) for r in rows}
        origin = np.linalg.inv(transforms[ref])
        transforms = {k:origin @ t for k,t in transforms.items()}
        pose_source = 'supplied camera-to-reference poses; metric scale supplied by operator'
    else:
        positions = session.get('frame_positions', {})
        try:
            locations = np.array([float(positions[str(k)]) for k,_,_ in pairs])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Missing encoder positions. Supply metric camera poses for this dataset.') from error
        if not np.isfinite(locations).all():
            raise ValueError('Encoder positions contain non-finite values.')
        diff = np.diff(locations)
        if not (np.all(diff >= -1e-5) or np.all(diff <= 1e-5)):
            raise ValueError('Multiple or reversing passes require supplied camera poses.')
        if np.ptp(locations) < .005:
            raise ValueError('Insufficient camera translation for stereo; record a moving overlapping pass.')
        axis = settings.camera_axis
        if axis == 'auto':
            # Determine direction from robust image correspondences across
            # several nearby camera positions. Refuse ambiguous vertical/rotating
            # motion rather than silently assigning a wrong camera axis.
            estimates=[]
            sift=cv2.SIFT_create(nfeatures=1500)
            stride=max(1,len(pairs)//40)
            for i in np.linspace(0,len(pairs)-stride-1,7,dtype=int):
                j=i+stride
                if abs(locations[j]-locations[i])<.002:continue
                a=cv2.imread(str(pairs[i][1]),0);b=cv2.imread(str(pairs[j][1]),0)
                ka,da=sift.detectAndCompute(a,None);kb,db=sift.detectAndCompute(b,None)
                if da is None or db is None:continue
                matches=[m for pair in cv2.BFMatcher().knnMatch(da,db,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.65*n.distance]
                if len(matches)<20:continue
                shifts=np.array([np.array(kb[m.trainIdx].pt)-ka[m.queryIdx].pt for m in matches])
                shift=np.median(shifts,axis=0)
                if abs(shift[0])>2*max(abs(shift[1]),1):
                    estimates.append(-np.sign(shift[0]/(locations[j]-locations[i])))
            if len(estimates)<2 or abs(np.mean(estimates))<.75:
                raise ValueError('Camera motion direction is ambiguous. Select the known camera X direction or supply calibrated poses.')
            axis='x' if np.mean(estimates)>0 else '-x'
        if axis not in ('x', '-x'):
            raise ValueError('Automatic gantry initialization supports horizontal camera X motion only; use supplied poses otherwise.')
        lo,hi=0,len(pairs)-1
        if settings.foreground=='auto':
            scores=[]
            for index in np.unique(np.linspace(0,len(pairs)-1,min(80,len(pairs)),dtype=int)):
                im=cv2.imread(str(pairs[index][1]));small=cv2.resize(im,(320,180))
                hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
                # Broad natural foliage colours. Blue cables/gantry paint are
                # excluded as seeds; pale leaves can still enter the ROI hull.
                seed=((hsv[:,:,1]>40)&(hsv[:,:,2]>20)&((hsv[:,:,0]<105)|(hsv[:,:,0]>125))).astype(np.uint8)
                seed=cv2.morphologyEx(seed,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
                count,lab,stats,centers=cv2.connectedComponentsWithStats(cv2.dilate(seed,np.ones((7,7),np.uint8)),8)
                score=0.
                if count>1:
                    central=np.exp(-2*np.sum(((centers[1:]-[160,90])/[160,90])**2,axis=1))
                    score=float(np.max(stats[1:,cv2.CC_STAT_AREA]*central))
                scores.append((int(index),score))
            peak=max(score for _,score in scores)
            if peak<150:
                raise ValueError('No reliable foliage window found. For pale/atypical foliage use depth mode with an explicit depth interval or supplied camera poses.')
            useful=[index for index,score in scores if score>=max(150,peak*.18)]
            pad=max(1,len(pairs)//80)
            lo=max(0,min(useful)-pad);hi=min(len(pairs)-1,max(useful)+pad)
        indices = sorted(set(int(np.argmin(abs(locations - x))) for x in np.linspace(locations[lo], locations[hi], settings.max_frames)))
        ids = [pairs[i][0] for i in indices]
        if len(ids) < 8:
            raise ValueError('Too few distinct camera positions.')
        ref = ids[len(ids)//2]
        transforms = {}
        for k in ids:
            T = np.eye(4)
            T[0,3] = (float(positions[str(k)]) - float(positions[str(ref)])) * (-1 if axis == '-x' else 1)
            transforms[k] = T
        pose_source = f'encoder initialization; {"image-estimated" if settings.camera_axis=="auto" else "operator-selected"} camera {axis} direction'
    depths = [];foliage_depths=[]
    digest = hashlib.sha256(intrinsic_path.read_bytes())
    for k in ids:
        rgb, dep = lookup[k]
        im = cv2.imread(str(rgb)); z = cv2.imread(str(dep), -1)
        if im is None or z is None or im.shape[:2] != (height,width) or z.shape != (height,width) or z.dtype != np.uint16:
            raise ValueError(f'Frame {k}: expected matching RGB and aligned uint16 depth at calibration resolution.')
        valid = z[(z>0)&(z<65535)][::64] / scale
        if len(valid):
            depths.append(valid)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        colour=(hsv[:,:,1]>40)&(hsv[:,:,2]>20)&((hsv[:,:,0]<105)|(hsv[:,:,0]>125))&(z>0)&(z<65535)
        if np.count_nonzero(colour)>z.size*.004:
            foliage_depths.append(z[colour][::32]/scale)
        digest.update(rgb.read_bytes()); digest.update(dep.read_bytes())
    if not depths:
        raise ValueError('Selected frames have no valid depth for scale/range assessment.')
    values = np.concatenate(depths)
    near = settings.near or max(.01, float(np.percentile(values, 1)) * .8)
    far = settings.far or float(np.percentile(values, 99)) * 1.05
    if not np.isfinite([near,far]).all() or not 0 < near < far:
        raise ValueError('Depth interval must be finite, positive and ordered.')
    median = float(np.median(np.concatenate(foliage_depths))) if foliage_depths and settings.foreground=='auto' else float(np.median(values))
    voxel = settings.voxel or max(.0003, median / 800)
    if not np.isfinite(voxel) or not .0001 <= voxel <= .02:
        raise ValueError('Voxel spacing must be between 0.1 and 20 mm.')
    # Select stereo neighbours by physical baseline and expected pixel disparity,
    # never by capture-specific frame offsets.
    neighbours = {}
    for k in ids:
        options = []
        for j in ids:
            if j == k:
                continue
            rel = np.linalg.inv(transforms[j]) @ transforms[k]
            t = rel[:3,3]
            if abs(t[0]) < 2 * max(abs(t[1]),abs(t[2]),1e-9):
                continue
            disparity = K[0,0] * abs(t[0]) / median
            if 3 <= disparity <= min(width*.3, 320):
                options.append((abs(disparity - 55),j))
        neighbours[str(k)] = [j for _,j in sorted(options)[:6]]
    anchors = [k for k in ids[::2] if len(neighbours[str(k)]) >= 2]
    if len(anchors) < 3 and not use_sensor:
        raise ValueError('Insufficient overlapping horizontal stereo baselines. Use closer-spaced translated views or suitable camera poses.')
    return dict(dataset=str(root), width=width,height=height,K=K.tolist(),dist=dist.tolist(),
                depth_scale=scale,near=near,far=far,voxel=voxel,trunc=max(voxel*5,.001),
                tolerance=max(voxel*6,.0015),stereo_tolerance=max(voxel*10,.0025),reference=ref,anchors=anchors,neighbours=neighbours,
                frames=[dict(frame=k,rgb=str(lookup[k][0]),depth=str(lookup[k][1]),accepted=True,transform=transforms[k].tolist()) for k in ids],
                method='sensor' if use_sensor else 'rgb',foreground=settings.foreground,pose_source=pose_source,settings=asdict(settings),input_sha256=digest.hexdigest(),
                warnings=['Hidden surfaces cannot be recovered without views that observe them.',
                          'Depth interval includes scene background; review foreground masks before traits.',
                          'Camera -Z defines display up for overhead scans; confirm height axis for other capture orientations.',
                          'Internal residuals do not establish physical trait accuracy.'])
