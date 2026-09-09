"""Multi-baseline RGB stereo evidence for gantry plant scans."""
import argparse,json,warnings,hashlib
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from scipy.optimize import least_squares
from .helpers import read_frame,cloud,upright_preview,profile

def refine_pair(a,b,K,relative):
    sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.015)
    ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None)
    kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    good=[m for pairs in cv2.BFMatcher().knnMatch(da,db,k=2) if len(pairs)==2 for m,n in [pairs] if m.distance<.7*n.distance]
    ua=np.array([ka[m.queryIdx].pt for m in good]);ub=np.array([kb[m.trainIdx].pt for m in good])
    _,inlier=cv2.findFundamentalMat(ua,ub,cv2.USAC_MAGSAC,.6,.999,10000)
    if inlier is None:raise ValueError('No epipolar inliers')
    ua,ub=ua[inlier.ravel()>0],ub[inlier.ravel()>0]
    xa=np.c_[(ua-[K[0,2],K[1,2]])/[K[0,0],K[1,1]],np.ones(len(ua))]
    xb=np.c_[(ub-[K[0,2],K[1,2]])/[K[0,0],K[1,1]],np.ones(len(ub))]
    rv=cv2.Rodrigues(relative[:3,:3])[0].ravel();t=relative[:3,3];x0=np.r_[rv,t];base=np.linalg.norm(t)
    def err(x,prior=True):
        R=cv2.Rodrigues(x[:3])[0];tx,ty,tz=x[3:];skew=np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]])
        E=skew@R;la=xa@E.T;lb=xb@E
        r=np.sum(xb*la,axis=1)/np.sqrt(la[:,0]**2+la[:,1]**2+lb[:,0]**2+lb[:,1]**2)*K[0,0]
        if prior:r=np.r_[r,(x[:3]-x0[:3])/.01,(x[3:]-x0[3:])/.01,(np.linalg.norm(x[3:])-base)/.0001]
        return r
    fit=least_squares(err,x0,loss='soft_l1',f_scale=.5,max_nfev=100)
    T=np.eye(4);T[:3,:3]=cv2.Rodrigues(fit.x[:3])[0];T[:3,3]=fit.x[3:]*base/np.linalg.norm(fit.x[3:])
    return T,dict(matches=len(good),epipolar_inliers=len(ua),epipolar_before_px=float(np.median(abs(err(x0,False)))),epipolar_after_px=float(np.median(abs(err(fit.x,False)))))

def stereo(a,b,K,rel,out=None,refine=False,near=.3,far=.9):
    h,w=a.shape[:2]
    if abs(rel[0,3])<2*max(abs(rel[1,3]),abs(rel[2,3]),1e-9):raise ValueError("Stereo requires a mainly horizontal baseline")
    if refine:rel,stats=refine_pair(a,b,K,rel)
    else:stats={'joint_camera_poses': True}
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K,None,K,None,(w,h),rel[:3,:3],rel[:3,3],alpha=0)
    m1=cv2.initUndistortRectifyMap(K,None,R1,P1,(w,h),cv2.CV_32FC1)
    m2=cv2.initUndistortRectifyMap(K,None,R2,P2,(w,h),cv2.CV_32FC1)
    left=cv2.remap(a,*m1,cv2.INTER_LINEAR);right=cv2.remap(b,*m2,cv2.INTER_LINEAR)
    nd=int(np.ceil((abs(P2[0,3])/near+32)/16))*16;nd=min(((w//2)//16)*16,max(16,nd))
    if nd<16:raise ValueError("Images are too small for stereo")
    md=0 if P2[0,3]<0 else -nd
    def matcher(minimum):return cv2.StereoSGBM_create(minDisparity=minimum,numDisparities=nd,blockSize=3,P1=8*3*9,P2=32*3*9,uniquenessRatio=8,speckleWindowSize=40,speckleRange=1,disp12MaxDiff=1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp=matcher(md).compute(left,right).astype(np.float32)/16
    rd=matcher(-md-nd).compute(right,left).astype(np.float32)/16
    y,x=np.indices((h,w),dtype=np.float32);rx=x-disp
    back=cv2.remap(rd,rx,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-10000)
    valid=(disp>md)&(rx>=0)&(rx<w)&(abs(disp+back)<.8)
    with np.errstate(invalid='ignore'):
        xyz=cv2.reprojectImageTo3D(disp,Q)@R1
    valid&=np.isfinite(xyz).all(axis=2)&(xyz[:,:,2]>near)&(xyz[:,:,2]<far)
    p=xyz[valid];uv=np.rint(p[:,:2]/p[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]).astype(int)
    inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv,p=uv[inside],p[inside]
    # Nearest surface wins if rectified pixels map to the same reference pixel.
    depth=np.full((h,w),np.inf,np.float32);np.minimum.at(depth,(uv[:,1],uv[:,0]),p[:,2]);depth[~np.isfinite(depth)]=np.nan
    stats['valid_depth_pixels']=int(np.isfinite(depth).sum())
    if out is not None:cv2.imwrite(str(out),np.concatenate([left,right],axis=1))
    return depth,stats


def run(args):
    cv2.setNumThreads(4)
    root,out=Path(args.dataset),Path(args.output);out.mkdir(exist_ok=True,parents=True)
    cfg=profile(root);K=np.array(cfg['K']);maps=cv2.initUndistortRectifyMap(K,np.array(cfg['dist']),None,K,(cfg['width'],cfg['height']),cv2.CV_32FC1)
    rows=json.loads(Path(args.poses).read_text())['frames'];poses={r['frame']:np.array(r['transform']) for r in rows if r['accepted']}
    signature=hashlib.sha256(Path(args.poses).read_bytes()+(root/'profile.json').read_bytes()+str(root.resolve()).encode()+str(args.refine_pairs).encode()+b'rgb-stereo-v2').hexdigest()
    pts=[];cols=[];records=[]
    for f in args.anchors:
        d=read_frame(root,f,K,maps);stack=[];pairs=[]
        if np.count_nonzero(d['roi'])<30:
            print(f'{f}: rejected; no foreground above support',flush=True)
            continue
        cache=out/f'depth_{f}.npz'
        cached_record=out/f'anchor_{f}_diagnostics.json'
        if cache.exists() and cached_record.exists() and json.loads(cached_record.read_text()).get('signature')==signature:
            saved=np.load(cache);z,votes,mask=saved['depth'],saved['votes'],saved['mask']
            y,x=np.nonzero(mask);v=z[y,x];p=np.c_[(x-K[0,2])*v/K[0,0],(y-K[1,2])*v/K[1,1],v]
            pts.append(p@poses[f][:3,:3].T+poses[f][:3,3]);cols.append(d['im'][mask][:,::-1]);print(f'{f}: cached',flush=True)
            records.append(json.loads(cached_record.read_text()))
            continue
        for g in cfg['neighbours'][str(f)]:
            b=read_frame(root,g,K,maps)
            dep,s=stereo(d['im'],b['im'],K,np.linalg.inv(poses[g])@poses[f],out/f'pair_{f}_{g}.jpg' if args.pair_images else None,args.refine_pairs,cfg['near'],cfg['far'])
            stack.append(dep);pairs.append(dict(frame=g,**s));print(f'{f}/{g}: {s}',flush=True)
        stack=np.array(stack)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            med=np.nanmedian(stack,axis=0);consistent=abs(stack-med)<cfg['stereo_tolerance']
            z=np.nanmedian(np.where(consistent,stack,np.nan),axis=0)
        votes=consistent.sum(axis=0)
        mask=d['roi']&(z>cfg['near'])&(z<d['foreground_limit'])&(votes>=2)
        if np.count_nonzero(mask)<30:
            print(f'{f}: rejected; too little repeated stereo agreement',flush=True)
            continue
        y,x=np.nonzero(mask);v=z[y,x];p=np.c_[(x-K[0,2])*v/K[0,0],(y-K[1,2])*v/K[1,1],v]
        p=p@poses[f][:3,:3].T+poses[f][:3,3];rgb=d['im'][mask][:,::-1]
        pts.append(p);cols.append(rgb)
        np.savez_compressed(out/f'depth_{f}.npz',depth=z,votes=votes,mask=mask,sensor=d['z'],rgb=d['im'],stack=stack)
        overlay=(d['im']*.2).astype(np.uint8);overlay[mask]=d['im'][mask];cv2.imwrite(str(out/f'mask_{f}.png'),overlay)
        for yaw in [0,180]:upright_preview(p,rgb,out/f'anchor_{f}_{yaw}.png',yaw)
        record=dict(frame=f,points=len(p),pairs=pairs,signature=signature)
        cached_record.write_text(json.dumps(record,indent=2));records.append(record)
        (out/'diagnostics.json').write_text(json.dumps(records,indent=2))
    if len(records)<3:raise ValueError('Fewer than three dense views passed foreground/stereo checks. Review the masks, depth range and camera poses.')
    p=cloud(np.concatenate(pts),np.concatenate(cols)).voxel_down_sample(cfg['voxel'])
    (out/'diagnostics.json').write_text(json.dumps(records,indent=2))
    o3d.io.write_point_cloud(str(out/'rgb_multiview.ply'),p)
    for yaw in [0,90,180,270]:upright_preview(np.asarray(p.points),np.uint8(np.asarray(p.colors)*255),out/f'view_{yaw}.png',yaw)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset');p.add_argument('--poses',required=True);p.add_argument('--output',required=True);p.add_argument('--anchors',nargs='+',type=int,required=True);p.add_argument('--pair-images',action='store_true');p.add_argument('--refine-pairs',action='store_true');run(p.parse_args())
