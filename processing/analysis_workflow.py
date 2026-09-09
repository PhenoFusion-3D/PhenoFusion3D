"""Auditable offline trait jobs used by the desktop analysis dialog."""
import argparse
import csv
from dataclasses import asdict
import html
import json
import math
from pathlib import Path


def mapping(text):
    result = {}
    for item in text.split():
        ref, model = map(int, item.split(':'))
        if ref < 1 or model < 1 or ref in result or model in result.values():
            raise ValueError('Use unique positive specimen pairs, e.g. 1:2 2:1.')
        result[ref] = model
    if not result:
        raise ValueError('Confirm specimen correspondence explicitly, e.g. 1:1.')
    return result


def fresh_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('Choose a new empty output directory to preserve previous results.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manual_report(rows, output, title='Manual versus software measurements'):
    body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(row.get(key,"")))}</td>' for key in rows[0])+'</tr>' for row in rows) if rows else ''
    header = ''.join(f'<th>{html.escape(key.replace("_"," "))}</th>' for key in rows[0]) if rows else ''
    (output/'manual_report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>body{{font:16px system-ui;margin:30px;color:#15352a}}table{{border-collapse:collapse}}td,th{{padding:12px;border:1px solid #c9dace;text-align:left}}.table{{overflow:auto}}</style><h1>{title}</h1><p>Only explicitly matched measurements are compared. Image-derived leaf chords are not full 3D leaf surface area or curved leaf length. Missing measurements are not zero.</p><div class="table"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div></html>''',encoding='utf-8')


def measure_landmarks(dataset, config, output):
    """Measure confirmed leaf endpoints using aligned depth and undistorted rays.

    Local depth sampling and bidirectional image matching can recover a missing
    endpoint observation. Their radii and correspondence scores are reported;
    they are estimates from observed neighbouring pixels, not exact point depth.
    """
    import cv2
    import numpy as np
    from .rgb_recovery.dataset import paired_images
    root = Path(dataset)
    data = json.loads(Path(config).read_text())
    if 'leaves' not in data and 'plants' in data:
        # Accept the previously reviewed guided-landmark file without requiring
        # its owners to re-enter every specimen and endpoint.
        data['leaves']=[dict(plant_id=plant['manual_plant_id'],leaf_id=leaf['leaf_id'],frame=plant['frame_index'],
                            points=leaf['length_pixels']+leaf['width_pixels'],manual_length_mm=leaf['manual_length_mm'],manual_width_mm=leaf['manual_width_mm'],support_depth_m=plant.get('support_depth_raw',float('inf'))/float(data['depth_scale_units_per_m']))
                        for plant in data['plants'] for leaf in plant['leaves']]
    scale = float(data['depth_scale_units_per_m'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Supply camera depth units per metre.')
    intr = json.loads((root/'kdc_intrinsics.txt').read_text())
    K = np.array(intr['K'],dtype=float)
    dist = np.array(intr.get('dist',[0]*5),dtype=float)
    pairs = {k:(rgb,dep) for k,rgb,dep in paired_images(root)}
    rows = []; identities=set()
    for leaf in data['leaves']:
        identity=(str(leaf['plant_id']),str(leaf['leaf_id']))
        if identity in identities:
            raise ValueError('Duplicate plant/leaf identity; use distinct IDs or a separate repeated-measurement run.')
        identities.add(identity)
        rgb_path, depth_path=pairs[int(leaf['frame'])]
        im=cv2.imread(str(rgb_path));depth=cv2.imread(str(depth_path),-1)
        if im is None or depth is None or depth.shape!=im.shape[:2]:
            raise ValueError('RGB and aligned depth must have the same dimensions.')
        if (intr.get('width'),intr.get('height'))!=(im.shape[1],im.shape[0]):
            raise ValueError('Image size must match calibration.')
        uv=np.array(leaf['points'],dtype=float)
        if uv.shape!=(4,2) or not np.isfinite(uv).all():
            raise ValueError('Each leaf needs tip, base, left edge and right edge coordinates.')
        pixels=np.rint(uv).astype(int)
        if ((pixels<0).any() or (pixels[:,0]>=depth.shape[1]).any() or (pixels[:,1]>=depth.shape[0]).any()):
            raise ValueError('Leaf endpoint lies outside the image.')
        from .leaf_tracking import recover_observation
        measured_frame,uv,z,radii,correlations=recover_observation(pairs,int(leaf['frame']),uv,scale,leaf.get('support_depth_m'))
        rays=cv2.undistortPoints(np.ascontiguousarray(uv,dtype=np.float64).reshape(-1,1,2),K,dist).reshape(-1,2)
        xyz=np.c_[rays*z[:,None],z]
        for trait,i,j in [('length',0,1),('width',2,3)]:
            manual=float(leaf[f'manual_{trait}_mm'])
            if not math.isfinite(manual) or manual<=0:
                raise ValueError('Manual leaf dimensions must be finite positive millimetres.')
            # Retain both definitions from the guided measurement workflow.
            chord=float(np.linalg.norm(xyz[i]-xyz[j])*1000)
            projected=float(np.linalg.norm(rays[i]-rays[j])*np.mean(z[[i,j]])*1000)
            rows.append(dict(plant_id=identity[0],leaf_id=identity[1],frame=int(leaf['frame']),measurement_frame=measured_frame,depth_search_radius_px=max(radii),minimum_tracking_correlation=min(correlations),trait=f'leaf_{trait}',manual_mm=manual,
                             rgbd_projected_mm=projected,rgbd_3d_chord_mm=chord,signed_error_mm=projected-manual,
                             absolute_percent_error=abs(projected-manual)/manual*100,source='reviewed RGB-D endpoints; not reconstructed leaf segmentation'))
    if not rows:
        raise ValueError('No annotated leaves were supplied.')
    (output/'matched_leaf_comparison.json').write_text(json.dumps(rows,indent=2))
    with (output/'matched_leaf_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    write_manual_report(rows,output,'Matched leaf length and width')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['traits','references','compare','leaves'])
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--reference',default='');parser.add_argument('--manual',default='');parser.add_argument('--mapping',default='')
    parser.add_argument('--config',default='');parser.add_argument('--depth-scale',type=float,default=0)
    parser.add_argument('--plants',type=int,default=1);parser.add_argument('--axis',choices=['x','y','z'],default='z')
    args=parser.parse_args()
    output=fresh_output(args.output)
    if args.mode=='traits':
        from .pointcloud_post import extract_traits
        result=extract_traits(args.input,output/'plant_1',height_axis=args.axis)
        (output/'measurement_status.json').write_text(json.dumps({'status':'unvalidated model descriptors','height_base':'minimum point along selected axis; confirm physical plant base','height_axis':args.axis,'pot_excluded':'operator must confirm','traits':asdict(result)},indent=2))
        print(f'RESULT: {output / "plant_1/traits.json"}',flush=True)
    elif args.mode=='references':
        from .reference_traits import extract_dataset_reference_traits
        if not math.isfinite(args.depth_scale) or args.depth_scale<=0:
            raise ValueError('Enter camera-reported raw units per metre for image-derived references.')
        extract_dataset_reference_traits(args.input,output,expected_plants=args.plants,depth_scale=args.depth_scale)
        print(f'RESULT: {output / "reference_traits.json"}',flush=True)
    elif args.mode=='compare':
        from .trait_validation import compare_reference_to_3d,compare_manual_to_3d
        pairs=mapping(args.mapping)
        if not args.reference and not args.manual:
            raise ValueError('Choose an image reference JSON and/or completed physical measurement CSV.')
        available={int(p.parent.name.split('_')[1]) for p in Path(args.input).glob('plant_*/traits.json')}
        if not set(pairs.values())<=available:
            raise ValueError('Mapped model specimen IDs do not all have traits.json files.')
        if args.reference:
            reference=json.loads(Path(args.reference).read_text())
            if not {int(r['plant_id']) for r in reference['plants']}<=pairs.keys():
                raise ValueError('Explicit mapping is required for every reference specimen.')
            compare_reference_to_3d(args.reference,args.input,output,plant_mapping=pairs)
        if args.manual:
            with Path(args.manual).open(encoding='utf-8-sig',newline='') as handle:
                manual=list(csv.DictReader(handle))
            if not {int(r['plant_id']) for r in manual}<=pairs.keys():
                raise ValueError('Explicit mapping is required for every physical specimen.')
            rows=compare_manual_to_3d(args.manual,args.input,output,plant_mapping=pairs)
            if not rows:raise ValueError('The manual CSV contains no populated measurements.')
            write_manual_report(rows,output)
        print(f'RESULT: {output / ("manual_report.html" if args.manual else "validation_report.html")}',flush=True)
    else:
        measure_landmarks(args.input,args.config,output)
        print(f'RESULT: {output / "manual_report.html"}',flush=True)

if __name__=='__main__':
    main()
