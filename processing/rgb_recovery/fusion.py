"""ICP registration and visibility-checked surface fusion of RGB stereo maps."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from .helpers import xyz_image,cloud,upright_preview,profile,remove_small_components,read_frame


def foreground_plane(cfg,frames,root,K):
    """Use central overlapping views to separate a raised support from the floor.

    This is an overhead-capture assumption, used only in automatic foreground
    mode. The plane is derived from observed support depths and metric poses.
    """
    if cfg['foreground']!='auto':return None
    reference=min(frames,key=lambda fr:abs(fr['frame']-cfg['reference']))
    nearby=sorted(frames,key=lambda fr:np.linalg.norm(fr['T'][:3,3]-reference['T'][:3,3]))[:7]
    maps=cv2.initUndistortRectifyMap(K,np.array(cfg['dist']),None,K,(cfg['width'],cfg['height']),cv2.CV_32FC1)
    normal=np.median([fr['T'][:3,2] for fr in nearby],axis=0)
    normal/=np.linalg.norm(normal)
    limits=[]
    for fr in nearby:
        obs=read_frame(root,fr['frame'],K,maps)
        limits.append(float(normal@(fr['T'][:3,3]+fr['T'][:3,2]*obs['foreground_limit'])))
    return dict(normal=normal.tolist(),maximum=float(np.median(limits)),source_frames=[fr['frame'] for fr in nearby],assumption='Common overhead support plane from central observed views; review mask before traits')

def project_visibility(xyz,frames,K,tolerance=.004,chunk_size=100000):
    """Bound temporary memory while preserving every independent view vote."""
    if chunk_size<1:raise ValueError('Visibility chunk size must be positive')
    if len({fr['frame'] for fr in frames})!=len(frames):
        raise ValueError('Each supporting view must have a distinct frame ID')
    support=np.zeros(len(xyz),np.uint16);conflict=np.zeros(len(xyz),np.uint16)
    colour=np.zeros((len(xyz),3),np.uint8)
    for start in range(0,len(xyz),chunk_size):
        end=min(start+chunk_size,len(xyz))
        support[start:end],conflict[start:end],colour[start:end]=_visibility_batch(xyz[start:end],frames,K,tolerance)
    return support,conflict,colour


def _visibility_batch(xyz,frames,K,tolerance=.004):
    """Require independent depth agreement; ignore legitimately occluded views."""
    ids=[f['frame'] for f in frames]
    if len(ids)!=len(set(ids)):
        raise ValueError('Each supporting view must have a distinct frame ID')
    support=np.zeros(len(xyz),np.uint16);conflict=np.zeros(len(xyz),np.uint16)
    color=np.zeros((len(xyz),3),np.uint8);best=np.zeros(len(xyz),np.float32)
    for frame in frames:
        T=np.linalg.inv(frame['T']);p=xyz@T[:3,:3].T+T[:3,3]
        uv=np.rint(p[:,:2]/p[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]).astype(int)
        x,y=uv.T;z=frame['z'];h,w=z.shape
        inside=(x>=1)&(x<w-1)&(y>=1)&(y<h-1)&(p[:,2]>0)
        ii=np.flatnonzero(inside);xx,yy=x[ii],y[ii]
        delta=np.full(len(ii),np.inf);minimum=np.full(len(ii),np.inf);vote_weight=np.zeros(len(ii))
        for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            obs=z[yy+dy,xx+dx];valid=np.isfinite(obs)&(frame['votes'][yy+dy,xx+dx]>=2)
            plant_valid=valid&frame['mask'][yy+dy,xx+dx]
            delta=np.minimum(delta,np.where(plant_valid,abs(obs-p[ii,2]),np.inf))
            vote_weight=np.maximum(vote_weight,np.where(plant_valid&(abs(obs-p[ii,2])<tolerance),frame['votes'][yy+dy,xx+dx],0))
            minimum=np.minimum(minimum,np.where(valid,obs,np.inf))
        agree=delta<tolerance;free=np.isfinite(minimum)&(p[ii,2]<minimum-.01)
        support[ii]+=agree.astype(np.uint16);conflict[ii]+=free.astype(np.uint16)
        weight=np.where(agree,vote_weight/(1+delta/.002)/p[ii,2],0)
        choose=weight>best[ii];jj=ii[choose];best[jj]=weight[choose];color[jj]=frame['rgb'][yy[choose],xx[choose],::-1]
    return support,conflict,color

def run(args):
    cv2.setNumThreads(4);root,source,out=Path(args.dataset),Path(args.depths),Path(args.output);out.mkdir(exist_ok=True,parents=True)
    cfg=profile(root);K=np.array(cfg['K']);tol=cfg['tolerance']
    pose_report=json.loads(Path(args.poses).read_text())
    poses={r['frame']:np.array(r['transform']) for r in pose_report['frames'] if r['accepted']}
    frames=[]
    for f in args.anchors:
        a=np.load(source/f'depth_{f}.npz');z=a['depth'];mask=a['mask'];xyz=xyz_image(z,K)[mask]@poses[f][:3,:3].T+poses[f][:3,3]
        rgb=a['rgb'][mask][:,::-1];p=cloud(xyz,rgb).voxel_down_sample(cfg['voxel']*2.5)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['voxel']*8,max_nn=30))
        frames.append(dict(frame=f,z=z,votes=a['votes'],mask=mask,rgb=a['rgb'],T=poses[f].copy(),pcd=p))
    diag=[]
    for fr in frames:
        neighbours=sorted([g for g in frames if g['frame']!=fr['frame']],key=lambda g:np.linalg.norm(g['T'][:3,3]-fr['T'][:3,3]))[:4]
        targets=[g['pcd'] for g in neighbours]
        target=o3d.geometry.PointCloud()
        for t in targets:target+=t
        target=target.voxel_down_sample(cfg['voxel']*2.5);target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['voxel']*8,max_nn=30))
        before=o3d.pipelines.registration.evaluate_registration(fr['pcd'],target,tol,np.eye(4))
        try:
            r=o3d.pipelines.registration.registration_colored_icp(fr['pcd'],target,tol,np.eye(4),criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35))
            method='coloured ICP'
        except RuntimeError:
            # Low texture or disconnected components may leave no colour
            # correspondences. Try the geometric estimator with the same gate.
            try:
                r=o3d.pipelines.registration.registration_icp(fr['pcd'],target,tol,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35))
                method='point-to-plane fallback'
            except RuntimeError:
                r=before;method='rejected: no usable correspondences'
        xyz=np.asarray(fr['pcd'].points);displacement=float(np.median(np.linalg.norm(xyz@r.transformation[:3,:3].T+r.transformation[:3,3]-xyz,axis=1)))
        angle=float(np.linalg.norm(cv2.Rodrigues(r.transformation[:3,:3])[0])*180/np.pi)
        accepted=r.fitness>.65 and displacement<tol*.5 and angle<.5 and not method.startswith('rejected')
        if accepted:fr['T']=r.transformation@fr['T']
        fr['usable']=accepted or (before.fitness>.65 and before.inlier_rmse<tol*.6)
        record=dict(frame=fr['frame'],icp_accepted=accepted,fitness_before=before.fitness,fitness_after=r.fitness,rmse_before_m=before.inlier_rmse,rmse_after_m=r.inlier_rmse,median_displacement_m=displacement,rotation_degrees=angle,transform=fr['T'].tolist())
        record['method']=method;record['integrated']=fr['usable'];diag.append(record);print(record,flush=True)
    if sum(r['icp_accepted'] for r in diag)<max(2,len(diag)//2):raise ValueError('Most coloured ICP corrections were rejected; inspect registration before fusion')
    (out/'icp_diagnostics.json').write_text(json.dumps(diag,indent=2))
    frames=[fr for fr in frames if fr['usable']]
    if len(frames)<3:raise ValueError('Fewer than three overlapping camera surfaces survived registration.')
    plane=foreground_plane(cfg,frames,root,K)
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=args.voxel,sdf_trunc=args.trunc,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    pin=o3d.camera.PinholeCameraIntrinsic(cfg['width'],cfg['height'],K[0,0],K[1,1],K[0,2],K[1,2])
    for fr in frames:
        valid=np.isfinite(fr['z'])&(fr['votes']>=2)&(fr['z']>cfg['near'])&(fr['z']<cfg['far'])
        valid&=cv2.dilate(fr['mask'].astype(np.uint8),np.ones((51,51),np.uint8))>0
        # Stereo-derived background is included to provide free-space evidence.
        z=np.where(valid,fr['z'],0).astype(np.float32)
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(fr['rgb'][:,:,::-1])),o3d.geometry.Image(z),depth_scale=1,depth_trunc=cfg['far'],convert_rgb_to_intensity=False)
        volume.integrate(rgbd,pin,np.linalg.inv(fr['T']));print('Integrated',fr['frame'],flush=True)
    surf=volume.extract_point_cloud();del volume
    xyz=np.asarray(surf.points);pre=len(xyz)
    print(f'Checking independent visibility for {pre:,} surface points in bounded batches',flush=True)
    keep=np.isfinite(xyz).all(axis=1)
    if plane is not None:keep&=xyz@np.array(plane['normal'])<plane['maximum']
    surf=surf.select_by_index(np.flatnonzero(keep));xyz=np.asarray(surf.points)
    support,conflict,rgb=project_visibility(xyz,frames,K,tolerance=tol)
    keep=(support>=args.min_views)&(conflict<=np.maximum(1,support*.3))
    xyz,rgb,support,conflict=xyz[keep],rgb[keep],support[keep],conflict[keep]
    if len(xyz)<30:raise ValueError('Fusion has fewer than 30 supported points; no usable result')
    del surf
    print(f'Cleaning {len(xyz):,} supported points',flush=True)
    p=cloud(xyz,rgb);p,indices=p.remove_statistical_outlier(20,1.8)
    support,conflict=support[indices],conflict[indices]
    p,support,conflict=remove_small_components(p,support,conflict,cfg['voxel'])
    o3d.io.write_point_cloud(str(out/'plant_rgb_icp.ply'),p)
    upright=o3d.geometry.PointCloud(p);upright_transform=np.diag([1.,-1.,-1.,1.]);upright_transform[2,3]=float(np.max(np.asarray(p.points)[:,2]))
    upright.transform(upright_transform);o3d.io.write_point_cloud(str(out/'plant_upright.ply'),upright)
    np.savez_compressed(out/'point_evidence.npz',support_views=support,contradicting_views=conflict)
    xyz=np.asarray(p.points);rgb=np.uint8(np.asarray(p.colors)*255)
    for yaw,e in [(0,25),(90,25),(180,25),(270,25),(0,90)]:upright_preview(xyz,rgb,out/f'view_{yaw}_{e}.png',yaw,e)
    summary=dict(points=len(xyz),extent_m=np.ptp(xyz,axis=0).tolist(),frames=args.anchors,icp_accepted=sum(r['icp_accepted'] for r in diag),median_support_views=float(np.median(support)),minimum_support_views=int(support.min()),volume_points_before_crop=pre,voxel_m=args.voxel,truncation_m=args.trunc,depth_source='RGB multi-baseline stereo, joint bundle-adjusted camera poses, followed by colored ICP and TSDF fusion',status='Candidate; visual and physical trait validation required')
    summary['frames']=[fr['frame'] for fr in frames]
    summary['automatic_foreground_plane']=plane
    summary['camera_refinement']={key:pose_report.get(key) for key in ('status','message','median_reprojection_px','p90_reprojection_px','points','observations')}
    summary['upright_transform']=upright_transform.tolist();summary['warnings']=cfg['warnings'];summary['pose_source']=cfg['pose_source'];summary['input_sha256']=cfg['input_sha256'];summary['foreground']=cfg['foreground']
    if pose_report.get('status')==0:
        summary['warnings'].append('Camera refinement reached its iteration limit. Reprojection residual gates passed, but numerical convergence was not established.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset');p.add_argument('--depths',required=True);p.add_argument('--poses',required=True);p.add_argument('--output',required=True);p.add_argument('--anchors',type=int,nargs='+',required=True);p.add_argument('--voxel',type=float,default=.0007);p.add_argument('--trunc',type=float,default=.003);p.add_argument('--min-views',type=int,default=3);run(p.parse_args())
