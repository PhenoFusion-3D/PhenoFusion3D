"""Joint RGB reprojection refinement of gantry camera poses and scene points."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .helpers import profile

def run(args):
 cv2.setNumThreads(4);root=Path(args.dataset);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 data=json.loads(Path(args.poses).read_text());rows=[r for r in data['frames'] if r['accepted']];frames=[r['frame'] for r in rows]
 cfg=profile(root);K=np.array(cfg['K']);maps=cv2.initUndistortRectifyMap(K,np.array(cfg['dist']),None,K,(cfg['width'],cfg['height']),cv2.CV_32FC1)
 poses=np.array([np.linalg.inv(np.array(r['transform'])) for r in rows]);ref=frames.index(data['reference'])
 sift=cv2.SIFT_create(nfeatures=6500,contrastThreshold=.015);features=[];parent={};members={}
 def find(x):
  if x not in parent:parent[x]=x;members[x]={x[0]}
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def join(a,b):
  ra,rb=find(a),find(b)
  if ra==rb or members[ra]&members[rb]:return
  parent[rb]=ra;members[ra]|=members.pop(rb)
 for f in frames:
  im=cv2.imread(next(r['rgb'] for r in cfg['frames'] if r['frame']==f));im=cv2.remap(im,*maps,cv2.INTER_LINEAR)
  k,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None);features.append((np.array([p.pt for p in k]),d))
 print('Features extracted',flush=True)
 matcher=cv2.BFMatcher()
 for i in range(len(frames)):
  for j in range(i+1,len(frames)):
   if j-i not in [1,2,4,7,11]:continue
   if features[i][1] is None or features[j][1] is None:continue
   good=[m for pair in matcher.knnMatch(features[i][1],features[j][1],k=2) if len(pair)==2 for m,n in [pair] if m.distance<.65*n.distance]
   if len(good)<20:continue
   ua=np.array([features[i][0][m.queryIdx] for m in good]);ub=np.array([features[j][0][m.trainIdx] for m in good])
   _,mask=cv2.findFundamentalMat(ua,ub,cv2.USAC_MAGSAC,.65,.999,10000)
   if mask is None:continue
   for m,keep in zip(good,mask.ravel()):
    if keep:join((i,m.queryIdx),(j,m.trainIdx))
  print('Matched camera',i,flush=True)
 groups={}
 for key in parent:groups.setdefault(find(key),[]).append(key)
 tracks=sorted([v for v in groups.values() if len(v)>=5],key=len,reverse=True)[:5500]
 points=[];obs=[]
 for tr in tracks:
  tr=sorted(tr);i,ki=tr[0];j,kj=tr[-1]
  p=cv2.triangulatePoints(K@poses[i,:3],K@poses[j,:3],features[i][0][ki].reshape(2,1),features[j][0][kj].reshape(2,1))[:,0];p=p[:3]/p[3]
  if not np.isfinite(p).all() or not cfg['near']*.5<p[2]<cfg['far']*1.5:continue
  good=[]
  for c,k in tr:
   q=poses[c,:3,:3]@p+poses[c,:3,3];uv=q[:2]/q[2]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
   if np.linalg.norm(uv-features[c][0][k])<4:good.append((c,k))
  if len(good)<4:continue
  idx=len(points);points.append(p)
  for c,k in good:obs.append((c,idx,*features[c][0][k]))
 if len(points)<30 or len(obs)<120:raise ValueError('Insufficient reliable image tracks. Check camera motion direction, overlap and texture.')
 obs=np.array(obs);ci=obs[:,0].astype(int);pi=obs[:,1].astype(int);uv=obs[:,2:]
 nc=len(frames);npnt=len(points);cameras=np.array([np.r_[cv2.Rodrigues(t[:3,:3])[0].ravel(),t[:3,3]] for t in poses])
 var=[i for i in range(nc) if i!=ref];cmap={i:n for n,i in enumerate(var)};base=np.array(points)
 initial=np.r_[cameras[var].ravel(),base.ravel()];nv=len(var)*6
 centers_prior=np.array([np.array(r['transform'])[:3,3] for r in rows]);prior_m=max(.002,cfg['far']*.005)
 def unpack(x):
  cams=cameras.copy();cams[var]=x[:nv].reshape(-1,6);p=x[nv:].reshape(-1,3)
  R=np.array([cv2.Rodrigues(c[:3])[0] for c in cams]);return cams,R,p
 def fun(x):
  c,R,p=unpack(x);q=np.einsum('nij,nj->ni',R[ci],p[pi])+c[ci,3:]
  pred=q[:,:2]/q[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
  centers=-np.einsum('nji,nj->ni',R,c[:,3:])
  # Encoder fixes metric scale; weak Y/Z/rotation priors prevent ill-conditioned
  # rail motion from turning into a curved camera path.
  prior=np.c_[(centers[var]-centers_prior[var])/prior_m,(c[var,:3]-cameras[var,:3])/.015]
  return np.r_[(pred-uv).ravel(),prior.ravel()]
 sparsity=lil_matrix((len(obs)*2+len(var)*6,len(initial)),dtype=int)
 for n,(c,p) in enumerate(zip(ci,pi)):
  if c!=ref:sparsity[2*n:2*n+2,cmap[c]*6:cmap[c]*6+6]=1
  sparsity[2*n:2*n+2,nv+p*3:nv+p*3+3]=1
 for i in range(len(var)):sparsity[len(obs)*2+i*6:len(obs)*2+(i+1)*6,i*6:(i+1)*6]=1
 print('BA points',npnt,'observations',len(obs),'initial median',np.median(np.linalg.norm(fun(initial)[:len(obs)*2].reshape(-1,2),axis=1)),flush=True)
 fit=least_squares(fun,initial,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=.5,x_scale='jac',ftol=1e-5,max_nfev=60,verbose=1)
 c,R,p=unpack(fit.x);err=np.linalg.norm(fun(fit.x)[:len(obs)*2].reshape(-1,2),axis=1)
 if not np.isfinite(err).all() or np.median(err)>1.5 or np.percentile(err,90)>4:raise ValueError('Camera refinement has excessive reprojection error; check calibration and motion.')
 for i,row in enumerate(rows):
  T=np.eye(4);T[:3,:3]=R[i];T[:3,3]=c[i,3:];row['transform']=np.linalg.inv(T).tolist();row['bundle_reprojection_px']=float(np.median(err[ci==i]))
 summary=dict(reference=frames[ref],frames=rows,points=npnt,observations=len(obs),median_reprojection_px=float(np.median(err)),p90_reprojection_px=float(np.percentile(err,90)),status=int(fit.status),message=fit.message)
 (out/'poses.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(out/'landmarks.npz',points=p,observations=obs,residual_px=err)
 print({k:v for k,v in summary.items() if k!='frames'},flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset');p.add_argument('--poses',required=True);p.add_argument('--output',required=True);run(p.parse_args())
