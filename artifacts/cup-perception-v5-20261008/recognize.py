"""New-camera reference-bank detection; never a robot command or grasp pose.

Only real-frame station registration plus independent local cup verification may
produce a 2D candidate. Camera identity, stale frames and depth are not inferred.
"""
import hashlib,json,time,argparse
from pathlib import Path
import cv2
import numpy as np
from collections import deque

ROOT=Path(__file__).resolve().parent

def project(points,matrix):
    return cv2.perspectiveTransform(np.array(points,np.float64)[None],matrix)[0]

def corners(box):
    a,b,c,d=box
    return np.float64([[a,b],[c,b],[c,d],[a,d]])

def ncc(a,b):
    a=a.astype(float).ravel();b=b.astype(float).ravel()
    a-=a.mean();b-=b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-6 else -1.

def edges(image):
    image=cv2.GaussianBlur(image,(3,3),0)
    return cv2.magnitude(cv2.Sobel(image,cv2.CV_32F,1,0),cv2.Sobel(image,cv2.CV_32F,0,1))

class StabilityGate:
    """Three distinct new frames; no stale/duplicate/invalid-frame persistence."""
    def __init__(self):
        self.points=deque(maxlen=3);self.last_id=None;self.last_time=None

    def update(self,result,frame_id,observed_time):
        result['stable_2d']=False
        if (not result.get('candidate_valid_2d') or type(frame_id) is not int
                or not isinstance(observed_time,(int,float)) or not np.isfinite(observed_time)
                or (self.last_id is not None and frame_id<=self.last_id)
                or (self.last_time is not None and not 0<observed_time-self.last_time<=2.)):
            self.points.clear();self.last_id=self.last_time=None
            result['stability_reason']='invalid_duplicate_or_discontinuous_frame'
            return result
        self.last_id,self.last_time=frame_id,observed_time
        self.points.append(result['visible_lower_rim_uv'])
        points=np.asarray(self.points)
        deviation=float(np.linalg.norm(points-np.median(points,axis=0),axis=1).max())
        result.update(stable_2d=len(points)==3 and deviation<=2.,
                      stability_samples=len(points),stability_deviation_px=deviation)
        return result

class Detector:
    def __init__(self,root=ROOT):
        cv2.setNumThreads(1);cv2.setRNGSeed(0)
        self.root=Path(root)
        self.manifest=json.loads((self.root/'reference/manifest.json').read_text())
        self.sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.02)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)
        self.references=[]
        for spec in self.manifest['references']:
            path=self.root/'reference'/f"{spec['frame']}.jpg"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
                raise ValueError('Reference checksum mismatch')
            image=cv2.imread(str(path),0)
            if image is None or list(image.shape[::-1])!=self.manifest['image_size_wh']:
                raise ValueError('Invalid reference image size')
            mask=np.zeros_like(image);x1,y1,x2,y2=spec['station_bbox'];mask[max(0,y1-120):y2,x1:x2]=255
            kp,desc=self.sift.detectAndCompute(image,mask)
            if desc is None or len(kp)<8:raise ValueError('Insufficient station reference features')
            self.references.append((spec,image,kp,desc))

    def register(self,reference,gray,kp,desc):
        spec,ref,rkp,rd=reference
        if desc is None or len(desc)<2:return None
        pairs=self.matcher.knnMatch(rd,desc,k=2)
        reverse={m.queryIdx:m.trainIdx for m in self.matcher.match(desc,rd)}
        good=[];seen_r=set();seen_l=set()
        for pair in pairs:
            if len(pair)!=2:continue
            m,n=pair;a=tuple(round(v) for v in rkp[m.queryIdx].pt);b=tuple(round(v) for v in kp[m.trainIdx].pt)
            if m.distance<.72*n.distance and reverse.get(m.trainIdx)==m.queryIdx and a not in seen_r and b not in seen_l:
                good.append(m);seen_r.add(a);seen_l.add(b)
        if len(good)<8:return None
        src=np.float64([rkp[m.queryIdx].pt for m in good]);dst=np.float64([kp[m.trainIdx].pt for m in good])
        if len(good)<12:return None
        h,keep=cv2.findHomography(src,dst,cv2.RANSAC,2,maxIters=2000,confidence=.995)
        if h is None or not np.isfinite(h).all():return None
        keep=keep.ravel().astype(bool)
        if keep.sum()<12 or keep.mean()<.55:return None
        x1,y1,x2,y2=spec['station_bbox']
        if np.any(np.ptp(src[keep],axis=0)/[x2-x1,y2-y1]<[.3,.25]):return None
        residual=np.linalg.norm(project(src[keep],h)-dst[keep],axis=1)
        poly=project(corners(spec['station_bbox']),h)
        scale=np.sqrt(abs(cv2.contourArea(poly.astype(np.float32)))/((x2-x1)*(y2-y1)))
        angle=np.degrees(np.arctan2(*(poly[1]-poly[0])[::-1]))
        if not .7<=scale<=1.4 or abs(angle)>25 or np.percentile(residual,95)>1.8:return None
        if not cv2.isContourConvex(poly.astype(np.float32)) or cv2.contourArea(poly.astype(np.float32),oriented=True)<=0:return None
        if np.any(poly<0) or np.any(poly[:,0]>=gray.shape[1]) or np.any(poly[:,1]>=gray.shape[0]):return None
        aligned=cv2.warpPerspective(gray,h,ref.shape[::-1],flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        score=ncc(ref[y1:y2,x1:x2],aligned[y1:y2,x1:x2])
        if score<.70:return None
        return h,aligned,{'inliers':int(keep.sum()),'ncc':score,'scale':float(scale),'rotation_deg':float(angle)}

    def match(self,reference,gray,kp,desc):
        spec,ref,*_=reference
        found=self.register(reference,gray,kp,desc)
        if found is None:return {'reference':spec['frame'],'reason':'station_not_verified'}
        h,aligned,station=found
        x1,y1,x2,y2=spec['cup_bbox'];margin=16
        template=ref[y1:y2,x1:x2]
        # Cup and machine front lie on different planes. Independently fit a
        # bounded local rotation/scale instead of assuming a single shared plane.
        verified=[];highest=-1.
        for angle in [-12,-6,0,6,12]:
            for scale in [.85,.925,1.,1.075,1.15]:
                local=cv2.getRotationMatrix2D(tuple(spec['rim_anchor']),angle,scale)
                adjusted=cv2.warpAffine(aligned,local,ref.shape[::-1],flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
                search=adjusted[y1-margin:y2+margin,x1-margin:x2+margin]
                scores=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
                _,score,_,(x,y)=cv2.minMaxLoc(scores);dx,dy=x-margin,y-margin
                highest=max(highest,score)
                if not np.isfinite(score) or score<.82:continue
                patch=adjusted[y1+dy:y2+dy,x1+dx:x2+dx]
                edge=ncc(edges(template),edges(patch))
                # Lower band independently excludes the printed upper body.
                rim=ncc(edges(template)[-12:],edges(patch)[-12:])
                if edge>=.65 and rim>=.65:
                    verified.append((min(score,edge,rim),score,edge,rim,dx,dy,local))
        result={'reference':spec['frame'],'station':station,'cup_ncc':highest,'reason':'cup_patch_or_rim_not_verified'}
        if not verified:return result
        _,score,edge,rim,dx,dy,local=max(verified,key=lambda v:v[0])
        result.update(cup_ncc=score,edge_ncc=edge,rim_edge_ncc=rim)
        shift=np.float64([[1,0,dx],[0,1,dy],[0,0,1]])
        cm=h@np.vstack((local,[0,0,1]))@shift;poly=project(corners(spec['cup_bbox']),cm)
        if np.any(poly<0) or np.any(poly[:,0]>=640) or np.any(poly[:,1]>=480):
            result['reason']='cup_out_of_frame';return result
        result.update(reason=None,quality=float(min(score,edge,rim)),
                      polygon=poly.tolist(),rim_anchor=project([spec['rim_anchor']],cm)[0].tolist(),
                      center=project([[(x1+x2)/2,(y1+y2)/2]],cm)[0].tolist())
        return result

    def predict(self,image,camera_serial):
        started=time.perf_counter()
        result=dict(ok=True,recognition_revision='reference-bank-20261008',camera_serial=camera_serial,
                    candidate_valid_2d=False,stable_2d=False,coordinate_frame='color_image_pixels',
                    visible_lower_rim_uv=None,visible_cup_center_uv=None,cup_bbox_xyxy_px=None,
                    camera_surface_xyz_m=None,robot_surface_xyz_m=None,grasp_pose=None,
                    robot_motion_ready=False,objects={},reasons=[])
        if camera_serial!=self.manifest['camera_serial']:
            result['reasons']=['wrong_camera_serial'];return result
        if image is None or image.dtype!=np.uint8 or image.shape!=(480,640,3):
            result['reasons']=['unexpected_image_format'];return result
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);kp,desc=self.sift.detectAndCompute(gray,None)
        attempts=[self.match(ref,gray,kp,desc) for ref in self.references]
        result['reference_attempts']=attempts
        valid=sorted([a for a in attempts if a['reason'] is None],key=lambda a:a['quality'],reverse=True)
        if not valid:result['reasons']=['station_or_actual_cup_not_verified']
        else:
            best=valid[0];anchor=np.array(best['rim_anchor'])
            disagree=[a for a in valid[1:] if a['quality']>=best['quality']-.08 and np.linalg.norm(np.array(a['rim_anchor'])-anchor)>8]
            if disagree:result['reasons']=['references_disagree_on_cup_location']
            else:
                p=np.array(best['polygon']);box=[*p.min(axis=0),*p.max(axis=0)]
                result.update(candidate_valid_2d=True,visible_lower_rim_uv=best['rim_anchor'],
                              visible_cup_center_uv=best['center'],cup_bbox_xyxy_px=list(map(float,box)),
                              selected_reference=best['reference'],annotation_uncertainty_px=5.,
                              anchor_semantics='Approximate visible lower rim, not cup-bottom center or grasp pose',
                              objects={'exposed_cup':{'detected':True,'polygon_px':best['polygon'],
                                       'method':'station_registered_independent_cup_and_rim'}})
        result['inference_ms']=round((time.perf_counter()-started)*1000,2)
        return result

def annotate(image,result):
    out=image.copy()
    if result['candidate_valid_2d']:
        p=np.rint(result['objects']['exposed_cup']['polygon_px']).astype(np.int32)
        cv2.polylines(out,[p],True,(0,220,255),2)
        cv2.drawMarker(out,tuple(np.rint(result['visible_lower_rim_uv']).astype(int)),(0,0,255),cv2.MARKER_CROSS,12,1)
    cv2.putText(out,'2D OUTLET CUP ONLY - NOT A GRASP POSE',(8,466),0,.45,(0,0,255),1)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--camera-serial',required=True)
    p.add_argument('--output',type=Path);args=p.parse_args()
    image=cv2.imread(str(args.image));r=Detector().predict(image,args.camera_serial)
    if args.output:cv2.imwrite(str(args.output),annotate(image,r))
    print(json.dumps(r,indent=2,allow_nan=False))
