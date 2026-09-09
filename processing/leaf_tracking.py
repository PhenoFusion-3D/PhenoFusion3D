"""Local image correspondence from the working guided leaf workflow."""
import cv2
import numpy as np

def track_pixel(
    reference_gray: np.ndarray,
    candidate_gray: np.ndarray,
    pixel: list[int],
    frame_delta: int,
    template_radius: int = 11,
) -> tuple[list[int], float]:
    u, v = map(int, pixel)
    if u-template_radius<0 or v-template_radius<0 or u+template_radius>=reference_gray.shape[1] or v+template_radius>=reference_gray.shape[0]:
        raise RuntimeError('Landmark is too close to the image border for reliable tracking')
    template = reference_gray[
        v - template_radius : v + template_radius + 1,
        u - template_radius : u + template_radius + 1,
    ]
    horizontal = min(260, abs(frame_delta) * 4 + 35)
    vertical = 35
    x1 = max(0, u - horizontal - template_radius)
    x2 = min(candidate_gray.shape[1], u + horizontal + template_radius + 1)
    y1 = max(0, v - vertical - template_radius)
    y2 = min(candidate_gray.shape[0], v + vertical + template_radius + 1)
    search = candidate_gray[y1:y2, x1:x2]
    if (
        template.size == 0
        or search.shape[0] < template.shape[0]
        or search.shape[1] < template.shape[1]
    ):
        raise RuntimeError("Landmark template or search window is outside the frame")
    score = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, location = cv2.minMaxLoc(score)
    matched = [
        x1 + location[0] + template_radius,
        y1 + location[1] + template_radius,
    ]
    return matched, float(maximum)


def observed_depths(depth, pixels, scale, support):
    values=[];radii=[]
    for u,v in np.rint(pixels).astype(int):
        found=None
        for radius in (0,3,5,7,9,13,17,21,25):
            patch=depth[max(0,v-radius):v+radius+1,max(0,u-radius):u+radius+1].astype(float)/scale
            patch=patch[(patch>.01)&(patch<support-.002)&(patch<65535/scale)]
            if len(patch) and (radius==0 or len(patch)>=5):
                lo,hi=np.percentile(patch,[10,90])
                if hi-lo <= max(.008,float(np.median(patch))*.06):
                    found=float(np.median(patch));radii.append(radius);break
        if found is None:raise ValueError('No coherent observed foreground depth near a leaf endpoint')
        values.append(found)
    return np.array(values),radii


def recover_observation(pairs, reference_id, pixels, scale, support=None):
    ids=sorted(pairs);index=ids.index(reference_id)
    reference=cv2.imread(str(pairs[reference_id][0]),0)
    source_depth=cv2.imread(str(pairs[reference_id][1]),-1)
    if support is None:
        border=np.r_[source_depth[:20].ravel(),source_depth[-20:].ravel(),source_depth[:,:20].ravel(),source_depth[:,-20:].ravel()]
        border=border[(border>0)&(border<65535)]
        support=float(np.percentile(border,75))/scale if len(border) else float('inf')
        valid=source_depth[(source_depth>0)&(source_depth<65535)]/scale
        if len(valid) and np.ptp(np.percentile(valid,[5,95]))<max(.01,float(np.median(valid))*.03):
            support=float('inf')
    try:
        z,radii=observed_depths(source_depth,pixels,scale,support)
        return reference_id,np.asarray(pixels),z,radii,[1.]*4
    except ValueError:
        pass
    candidates=[]
    span=min(80,max(4,len(ids)//12));step=max(1,span//16)
    for j in range(max(0,index-span),min(len(ids),index+span+1),step):
        if j==index:continue
        frame=ids[j];gray=cv2.imread(str(pairs[frame][0]),0);depth=cv2.imread(str(pairs[frame][1]),-1)
        if gray is None or depth is None or gray.shape!=reference.shape:continue
        try:
            tracked=[];scores=[]
            for pixel in pixels:
                point,score=track_pixel(reference,gray,pixel,j-index)
                back,backscore=track_pixel(gray,reference,point,index-j)
                if min(score,backscore)<.55 or np.linalg.norm(np.array(back)-pixel)>2.5:
                    raise ValueError('Ambiguous image correspondence')
                tracked.append(point);scores.append(min(score,backscore))
            z,radii=observed_depths(depth,tracked,scale,support)
            quality=float(np.mean(scores)-.01*max(radii)-.0002*abs(j-index))
            candidates.append((quality,frame,np.array(tracked),z,radii,scores))
        except (ValueError,RuntimeError):continue
    if not candidates:raise ValueError('No adjacent frame retained all four reliably matched, depth-supported endpoints; mark another view.')
    _,frame,pixels,z,radii,scores=max(candidates,key=lambda row:row[0])
    return frame,pixels,z,radii,scores
