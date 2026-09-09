"""Offline command entrypoint also used by the GUI's isolated process."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import traceback

from .dataset import Settings, inspect_dataset


def run(dataset, output, settings, inspect_only=False):
    output = Path(output).resolve()
    dataset = Path(dataset).resolve()
    if output == dataset or output in dataset.parents:
        raise ValueError('Use a new output folder, not the dataset itself or a parent directory.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output folder is not empty. Choose a new run folder to preserve previous results.')
    print('Checking calibration, frame IDs, depth units and camera baselines...', flush=True)
    cfg = inspect_dataset(dataset, settings)
    output.mkdir(parents=True, exist_ok=True)
    (output/'profile.json').write_text(json.dumps(cfg,indent=2))
    (output/'initial_poses.json').write_text(json.dumps(cfg,indent=2))
    print(f"Preflight: {len(cfg['frames'])} poses, {len(cfg['anchors'])} dense views, depth {cfg['near']:.3f}–{cfg['far']:.3f} m",flush=True)
    if inspect_only:
        return output/'profile.json'
    state = {'status':'running','completed_stages':[], 'dataset':str(dataset)}
    try:
        # Lazy imports keep optional processing dependencies out of app startup.
        from . import bundle, stereo, fusion, viewer, sensor
        steps = [
            ('camera_bundle',bundle.run,dict(dataset=output,poses=output/'initial_poses.json',output=output/'bundle')),
            ('rgb_stereo',stereo.run,dict(dataset=output,poses=output/'bundle/poses.json',output=output/'rgb',anchors=cfg['anchors'],pair_images=False,refine_pairs=False)),
            ('icp_fusion',fusion.run,dict(dataset=output,poses=output/'bundle/poses.json',depths=output/'rgb',output=output/'result',anchors=cfg['anchors'],voxel=cfg['voxel'],trunc=cfg['trunc'],min_views=3)),
        ]
        if cfg['method']=='sensor':
            steps=[('sensor_icp',lambda _:sensor.run(cfg,output),{})]
        for name, function, args in steps:
            if name=='icp_fusion':
                records=json.loads((output/'rgb/diagnostics.json').read_text())
                args['anchors']=[r['frame'] for r in records]
            (output/'run_status.json').write_text(json.dumps(dict(state,current_stage=name),indent=2))
            print(f'Stage: {name}',flush=True)
            function(SimpleNamespace(**args))
            state['completed_stages'].append(name)
        summary = json.loads((output/'result/summary.json').read_text())
        photo = next(r['rgb'] for r in cfg['frames'] if r['frame']==cfg['reference'])
        viewer.run(SimpleNamespace(cloud=output/'result/plant_rgb_icp.ply',summary=output/'result/summary.json',photo=photo,output=output/'result/index.html'))
        report = '# Reconstruction evidence\n\n' + '\n\n'.join(summary.get('warnings',cfg['warnings']))
        report += '\n\nThis candidate needs foreground and orientation review before trait extraction. The pot/background may be included.\n\n'
        report += f"Reconstructed points: {summary['points']:,}. Method: {summary.get('depth_source','RGB stereo')}.\n\n"
        report += ('Minimum supporting views: '+str(summary['minimum_support_views'])+'\n\n') if summary['minimum_support_views'] is not None else 'Sensor-depth route: independent multi-view support counts are unavailable.\n\n'
        report += '[Open model](index.html) · [Upright PLY](plant_upright.ply) · [Diagnostics](icp_diagnostics.json)\n'
        (output/'result/RECONSTRUCTION_REPORT.md').write_text(report)
        state['status']='complete_candidate'
        print(f"RESULT: {output/'result/index.html'}",flush=True)
        return output/'result/index.html'
    except BaseException as error:
        state.update(status='failed',error=str(error))
        raise
    finally:
        (output/'run_status.json').write_text(json.dumps(state,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset');parser.add_argument('--output',required=True)
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--depth-scale',type=float,default=0)
    parser.add_argument('--near',type=float,default=0);parser.add_argument('--far',type=float,default=0)
    parser.add_argument('--max-frames',type=int,default=44);parser.add_argument('--voxel',type=float,default=0)
    parser.add_argument('--camera-axis',choices=['auto','x','-x'],default='auto')
    parser.add_argument('--initial-poses',default='')
    parser.add_argument('--foreground',choices=['auto','depth','colour'],default='auto')
    parser.add_argument('--method',choices=['auto','rgb','sensor'],default='auto')
    args=parser.parse_args()
    run(args.dataset,args.output,Settings(**{key:getattr(args,key) for key in Settings.__dataclass_fields__}),args.inspect_only)

if __name__=='__main__':
    # Bound CPU consumption without changing the lab GUI or capture runtime.
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
    main()
