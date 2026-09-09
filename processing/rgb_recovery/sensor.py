"""Use the unchanged lab ICP implementation when RGB pose priors are absent."""
import json
from pathlib import Path
import numpy as np
import open3d as o3d

from processing.reconstructor import Reconstructor


def run(cfg, output):
    out=Path(output)/'result';out.mkdir(parents=True,exist_ok=True)
    records=[]
    def progress(i,total,pcd,fitness,rmse,status):
        print(f'Sensor ICP {i+1}/{total}: {status}, fitness={fitness:.3f}, residual={rmse:.5f} m',flush=True)
        records.append(dict(frame=cfg['frames'][i]['frame'],status=status,fitness=float(fitness),rmse_m=float(rmse)))
    recon=Reconstructor(pairs=[(r['rgb'],r['depth']) for r in cfg['frames']],K=np.array(cfg['K']),dist=cfg['dist'],
                        depth_scale=cfg['depth_scale'],depth_trunc=cfg['far'],voxel_size=max(.001,cfg['voxel']*3),
                        max_iter=80,depth_min_mm=cfg['near']*cfg['depth_scale'],min_fitness=.3,max_rmse=max(.005,cfg['tolerance']*3),
                        on_frame=progress)
    cloud,success,fail=recon.run()
    if len(success)<max(3,len(cfg['frames'])//2) or len(cloud.points)<100:
        raise ValueError('Too few sensor-depth frames aligned reliably. Record more overlapping views or provide calibrated camera poses.')
    if not np.isfinite(np.asarray(cloud.points)).all():raise ValueError('Non-finite reconstructed coordinates.')
    o3d.io.write_point_cloud(str(out/'plant_rgb_icp.ply'),cloud)
    transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=float(np.max(np.asarray(cloud.points)[:,2]))
    upright=o3d.geometry.PointCloud(cloud);upright.transform(transform);o3d.io.write_point_cloud(str(out/'plant_upright.ply'),upright)
    summary=dict(points=len(cloud.points),frames=[r['frame'] for r in cfg['frames']],depth_source='sensor depth with lab ICP',
                 minimum_support_views=None,upright_transform=transform.tolist(),warnings=cfg['warnings']+[
                     'Sensor-depth fallback: no RGB stereo support-vote validation. Inspect stretched edges and missing surfaces.',
                     'Sampling may be too sparse for some recordings; increase camera views if alignment fails.'],
                 pose_source=cfg['pose_source'],status='Candidate; scene cleanup and physical validation required')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'icp_diagnostics.json').write_text(json.dumps(records,indent=2))
