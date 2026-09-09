"""Measured, distortion-aware multi-view plant reconstruction.

RGB feature registration initializes each pose independently; coloured ICP
refines it. Fusion retains 3D surfaces from all accepted views, not only the
reference depth sheet. Outputs are separate from all previous reconstructions.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


def cloud(xyz, rgb):
    p=o3d.geometry.PointCloud()
    p.points=o3d.utility.Vector3dVector(xyz)
    p.colors=o3d.utility.Vector3dVector(rgb.astype(float)/255)
    return p


def xyz_image(depth,K):
    y,x=np.indices(depth.shape)
    return np.stack([(x-K[0,2])*depth/K[0,0],(y-K[1,2])*depth/K[1,1],depth],axis=-1)


def upright_preview(xyz,rgb,path,yaw=0,elevation=25):
    """Equal-scale orthographic view, camera -Z displayed as plant up."""
    p=xyz.astype(float)*[1,-1,-1];p-=np.median(p,axis=0)
    a,b=np.deg2rad([yaw,elevation])
    right=np.array([np.cos(a),np.sin(a),0])
    up=np.array([-np.sin(a)*np.sin(b),np.cos(a)*np.sin(b),np.cos(b)])
    toward=np.cross(right,up)
    u,v,d=p@right,p@up,p@toward
    lo=np.percentile(np.c_[u,v],.05,axis=0);hi=np.percentile(np.c_[u,v],99.95,axis=0)
    s=min(1150/max(hi[0]-lo[0],1e-6),850/max(hi[1]-lo[1],1e-6))
    x=np.rint((u-(lo[0]+hi[0])/2)*s+600).astype(int)
    y=np.rint(450-(v-(lo[1]+hi[1])/2)*s).astype(int)
    keep=(x>=0)&(x<1200)&(y>=0)&(y<900);idx=np.argsort(d);idx=idx[keep[idx]]
    im=np.full((900,1200,3),245,np.uint8)
    # Resolve the nearest sample at each pixel; splat footprint is display only.
    zbuf=np.full((900,1200),-np.inf)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        ii=idx[(x[idx]+dx<1200)&(y[idx]+dy<900)]
        px,py=x[ii]+dx,y[ii]+dy
        np.maximum.at(zbuf,(py,px),d[ii])
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        ii=idx[(x[idx]+dx<1200)&(y[idx]+dy<900)]
        px,py=x[ii]+dx,y[ii]+dy
        visible=d[ii]>=zbuf[py,px]-1e-9
        im[py[visible],px[visible]]=rgb[ii[visible],::-1]
    cv2.imwrite(str(path),im)


def profile(root):
    return json.loads((Path(root)/'profile.json').read_text())

def read_frame(root,f,K,maps):
    cfg=profile(root)
    row=next(r for r in cfg['frames'] if r['frame']==f)
    im=cv2.imread(row['rgb']); raw=cv2.imread(row['depth'],-1)
    if im is None or raw is None:raise ValueError(f'Cannot decode frame {f}')
    im=cv2.remap(im,*maps,cv2.INTER_LINEAR)
    z=cv2.remap(raw,*maps,cv2.INTER_NEAREST).astype(np.float32)/cfg['depth_scale']
    valid=(z>cfg['near'])&(z<cfg['far'])
    # Depth foreground does not require green or saturated leaves. The operator
    # may constrain the interval to exclude the supporting table/background.
    roi=valid.copy();foreground_limit=cfg['far']
    if cfg['foreground']=='auto':
        # In an overhead capture, the support/background is usually visible at
        # the border. Segment above it without imposing a foliage hue.
        border=np.r_[z[:max(1,z.shape[0]//12)].ravel(),z[-max(1,z.shape[0]//12):].ravel(),z[:,:max(1,z.shape[1]//12)].ravel(),z[:,-max(1,z.shape[1]//12):].ravel()]
        border=border[(border>cfg['near'])&(border<cfg['far'])]
        if len(border)>100:
            support=float(np.percentile(border,70))
            hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            neutral=z[valid&(hsv[:,:,1]<35)&(hsv[:,:,2]>60)]
            # Image borders can see the distant floor beyond a raised table.
            # Use the dominant neutral-depth layer when a sufficiently strong
            # supporting surface is visible inside the frame.
            if len(neutral)>z.size*.02:
                width=max(cfg['tolerance']*3,.005)
                bins=np.floor(neutral/width).astype(int)
                counts=np.bincount(bins)
                peak=int(np.argmax(counts))
                close=neutral[abs(bins-peak)<=1]
                if counts[peak]>len(neutral)*.08 and len(close)>z.size*.02:
                    support=float(np.median(close))
            foreground_limit=support-max(cfg['tolerance']*3,support*.025)
            foreground=valid&(z<foreground_limit)
            hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            seed=foreground&(hsv[:,:,1]>40)&(hsv[:,:,2]>20)&((hsv[:,:,0]<105)|(hsv[:,:,0]>125))
            seed=cv2.morphologyEx(seed.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
            # Broad foliage hues seed the plant. Grey foliage falls back to depth
            # foreground; colour is never used to synthesize geometry.
            if seed.sum()>z.size*.004:
                connected=cv2.dilate(seed.astype(np.uint8),np.ones((max(3,z.shape[1]//40),)*2,np.uint8))
                count,lab,stats,centers=cv2.connectedComponentsWithStats(connected,8)
                scores=stats[1:,cv2.CC_STAT_AREA].astype(float)
                normalized=(centers[1:]-[z.shape[1]/2,z.shape[0]/2])/[z.shape[1]/2,z.shape[0]/2]
                scores*=np.exp(-np.sum(normalized**2,axis=1))
                chosen=1+int(np.argmax(scores))
                yy,xx=np.nonzero((lab==chosen)&seed)
                if len(xx)>=3:
                    foreground=np.zeros(z.shape,np.uint8)
                    cv2.fillConvexPoly(foreground,cv2.convexHull(np.c_[xx,yy].astype(np.int32)),1)
                    foreground=foreground.astype(bool)
            n,labels,stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),8)
            keep=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>max(30,z.size*.0005));keep=keep[keep!=0]
            foreground=np.isin(labels,keep)
            if foreground.sum()>z.size*.002:
                roi=cv2.dilate(foreground.astype(np.uint8),np.ones((15,15),np.uint8))>0
            else:
                roi=np.zeros_like(valid)
        else:
            roi=np.zeros_like(valid)
    if cfg['foreground']=='colour':
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        seed=(valid&(hsv[:,:,1]>40)&(hsv[:,:,2]>20)).astype(np.uint8)
        roi=cv2.dilate(seed,np.ones((21,21),np.uint8))>0
    return dict(frame=f,im=im,z=z,roi=roi,mask=valid&roi,foreground_limit=foreground_limit)

def remove_small_components(p, support, conflict, voxel):
    if len(p.points)<30:raise ValueError('Too few supported points; inspect depth, poses and masks.')
    labels=np.asarray(p.cluster_dbscan(max(voxel*12,.002),8))
    counts=np.bincount(labels[labels>=0])
    keep=np.zeros(len(labels),dtype=bool)
    good=labels>=0
    if len(counts):keep[good]=counts[labels[good]]>=max(30,int(len(labels)*.0005))
    if not keep.any():raise ValueError('No connected supported surface remains. Do not use this run for traits.')
    return p.select_by_index(np.flatnonzero(keep)),support[keep],conflict[keep]
