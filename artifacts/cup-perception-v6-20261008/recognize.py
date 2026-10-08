"""New-camera reference-bank detection; never a robot command or grasp pose.

Only real-frame station registration plus independent local cup verification may
produce a 2D candidate. Camera identity, stale frames and depth are not inferred.
"""
import hashlib,json,time,argparse
from pathlib import Path
import cv2
import numpy as np
from collections import deque
from observed_rim import detect as detect_observed_rim

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
        self.implementation={'opencv':cv2.__version__,'numpy':np.__version__,
                             'recognize_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'rim_module_sha256':hashlib.sha256((ROOT/'observed_rim.py').read_bytes()).hexdigest(),
                             'reference_manifest_sha256':hashlib.sha256((self.root/'reference/manifest.json').read_bytes()).hexdigest()}
        self.sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.02)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)
        self.references=[]
        self.planar_references={}
        self.registration_diagnostics={}
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
            # Upper printed film/rollers and lower metal frame are not one
            # plane. If mixed-depth matching fails, independently match only
            # the lower frame rather than reducing RANSAC consensus thresholds.
            mask[:y1]=0
            planar_kp,planar_desc=self.sift.detectAndCompute(image,mask)
            self.planar_references[spec['frame']]=(spec,image,planar_kp,planar_desc)

    def register(self,reference,gray,kp,desc):
        mixed={'support':'machine_with_upper_features'}
        audits=[mixed]
        self.registration_diagnostics[reference[0]['frame']]=audits
        result=self.register_features(reference,gray,kp,desc,mixed)
        if result is not None:return result
        planar=self.planar_references[reference[0]['frame']]
        audit={'support':'lower_station_plane'};audits.append(audit)
        if planar[3] is None or len(planar[2])<12:
            audit.update(passed=False,failed_gate='insufficient_planar_reference_features',reference_features=len(planar[2]),required=12)
            return None
        result=self.register_features(planar,gray,kp,desc,audit)
        if result is not None:result[2]['registration_support']='lower_station_plane'
        return result

    def register_features(self,reference,gray,kp,desc,audit):
        spec,ref,rkp,rd=reference
        audit.update(reference_features=len(rkp),live_features=len(kp),passed=False,
                     thresholds={'min_inliers':12,'min_inlier_fraction':.55,'min_spread_xy':[.3,.25],
                                 'scale_range':[.7,1.4],'max_rotation_deg':25,'max_residual_p95_px':1.8,'min_ncc':.70})
        def reject(reason):
            audit['failed_gate']=reason
            return None
        if desc is None or len(desc)<2:return reject('insufficient_live_features')
        pairs=self.matcher.knnMatch(rd,desc,k=2)
        reverse={m.queryIdx:m.trainIdx for m in self.matcher.match(desc,rd)}
        good=[];seen_r=set();seen_l=set()
        for pair in pairs:
            if len(pair)!=2:continue
            m,n=pair;a=tuple(round(v) for v in rkp[m.queryIdx].pt);b=tuple(round(v) for v in kp[m.trainIdx].pt)
            if m.distance<.72*n.distance and reverse.get(m.trainIdx)==m.queryIdx and a not in seen_r and b not in seen_l:
                good.append(m);seen_r.add(a);seen_l.add(b)
        audit['unique_matches']=len(good)
        if len(good)<12:return reject('insufficient_unique_matches')
        src=np.float64([rkp[m.queryIdx].pt for m in good]);dst=np.float64([kp[m.trainIdx].pt for m in good])
        h,keep=cv2.findHomography(src,dst,cv2.RANSAC,2,maxIters=2000,confidence=.995)
        if h is None or not np.isfinite(h).all():return reject('invalid_homography')
        keep=keep.ravel().astype(bool)
        audit.update(inliers=int(keep.sum()),inlier_fraction=float(keep.mean()))
        if keep.sum()<12:return reject('insufficient_inliers')
        if keep.mean()<.55:return reject('low_inlier_fraction')
        x1,y1,x2,y2=spec['station_bbox']
        spread=np.ptp(src[keep],axis=0)/[x2-x1,y2-y1];audit['spread_xy']=spread.tolist()
        if np.any(spread<[.3,.25]):return reject('matches_too_localized')
        residual=np.linalg.norm(project(src[keep],h)-dst[keep],axis=1)
        poly=project(corners(spec['station_bbox']),h)
        if not np.isfinite(residual).all() or not np.isfinite(poly).all():return reject('nonfinite_projection')
        scale=np.sqrt(abs(cv2.contourArea(poly.astype(np.float32)))/((x2-x1)*(y2-y1)))
        angle=np.degrees(np.arctan2(*(poly[1]-poly[0])[::-1]))
        audit.update(scale=float(scale),rotation_deg=float(angle),residual_p95_px=float(np.percentile(residual,95)))
        if not .7<=scale<=1.4:return reject('scale_out_of_range')
        if abs(angle)>25:return reject('rotation_out_of_range')
        if np.percentile(residual,95)>1.8:return reject('high_reprojection_error')
        if not cv2.isContourConvex(poly.astype(np.float32)) or cv2.contourArea(poly.astype(np.float32),oriented=True)<=0:return reject('invalid_projected_shape')
        if np.any(poly<0) or np.any(poly[:,0]>=gray.shape[1]) or np.any(poly[:,1]>=gray.shape[0]):return reject('station_support_out_of_frame')
        aligned=cv2.warpPerspective(gray,h,ref.shape[::-1],flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        score=ncc(ref[y1:y2,x1:x2],aligned[y1:y2,x1:x2])
        audit['appearance_ncc']=score
        if score<.70:return reject('low_station_appearance_ncc')
        audit.update(passed=True,failed_gate=None)
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
        result.update(cup_ncc=score,edge_ncc=edge,rim_edge_ncc=rim,
                      local_shift_reference_px=[dx,dy])
        shift=np.float64([[1,0,dx],[0,1,dy],[0,0,1]])
        cm=h@np.vstack((local,[0,0,1]))@shift;poly=project(corners(spec['cup_bbox']),cm)
        if np.any(poly<0) or np.any(poly[:,0]>=640) or np.any(poly[:,1]>=480):
            result['reason']='cup_out_of_frame';return result
        result.update(reason=None,quality=float(min(score,edge,rim)),
                      polygon=poly.tolist(),rim_anchor=project([spec['rim_anchor']],cm)[0].tolist(),
                      center=project([[(x1+x2)/2,(y1+y2)/2]],cm)[0].tolist())
        return result

    def match_direct_rim(self,reference,gray,bgr,kp,desc):
        spec,ref,*_=reference
        found=self.register(reference,gray,kp,desc)
        if found is None:return None
        h,aligned,station=found
        expected=project([spec['rim_anchor']],h)[0]
        if not np.isfinite(expected).all():return None
        candidates=detect_observed_rim(bgr,expected,(spec['cup_bbox'][2]-spec['cup_bbox'][0])*station['scale'])
        if len(candidates)!=1:return None
        rim=candidates[0]
        poly=corners(rim['box'])
        if np.any(poly<0) or np.any(poly[:,0]>=640) or np.any(poly[:,1]>=480):return None
        x1,y1,x2,y2=rim['box']
        return dict(reference=spec['frame'],reason=None,station=station,method='original_image_curved_rim',
                    rim_geometry=rim,quality=rim['quality'],polygon=poly.tolist(),
                    rim_anchor=rim['anchor'],center=[(x1+x2)/2,(y1+y2)/2])

    def predict(self,image,camera_serial):
        started=time.perf_counter()
        result=dict(ok=True,recognition_revision='observed-rim-v6-20261008',camera_serial=camera_serial,
                    implementation=dict(self.implementation),
                    candidate_valid_2d=False,stable_2d=False,coordinate_frame='color_image_pixels',
                    visible_lower_rim_uv=None,visible_cup_center_uv=None,cup_bbox_xyxy_px=None,
                    camera_surface_xyz_m=None,robot_surface_xyz_m=None,grasp_pose=None,
                    robot_motion_ready=False,objects={},reasons=[])
        if camera_serial!=self.manifest['camera_serial']:
            result['reasons']=['wrong_camera_serial'];return result
        if image is None or image.dtype!=np.uint8 or image.shape!=(480,640,3):
            result['reasons']=['unexpected_image_format'];return result
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);kp,desc=self.sift.detectAndCompute(gray,None)
        self.registration_diagnostics={}
        result['input_diagnostics']={'image_size_wh':[640,480],'gray_mean':float(gray.mean()),
                                     'gray_std':float(gray.std()),'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                                     'decoded_image_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
        attempts=[self.match(ref,gray,kp,desc) for ref in self.references]
        result['reference_attempts']=attempts
        valid=sorted([a for a in attempts if a['reason'] is None],key=lambda a:a['quality'],reverse=True)
        if not valid:
            geometry=[self.match_direct_rim(ref,gray,image,kp,desc) for ref in self.references]
            valid=sorted([g for g in geometry if g is not None],key=lambda g:g['quality'],reverse=True)
            result['geometry_attempts']=valid
        if not valid:result['reasons']=['station_or_actual_cup_not_verified']
        else:
            best=valid[0];anchor=np.array(best['rim_anchor'])
            disagree=[a for a in valid[1:] if a['quality']>=best['quality']-.08 and np.linalg.norm(np.array(a['rim_anchor'])-anchor)>8]
            if disagree:result['reasons']=['references_disagree_on_cup_location']
            else:
                if best.get('method')!='original_image_curved_rim':
                    p=np.array(best['polygon'])
                    measured=detect_observed_rim(image,best['rim_anchor'],float(np.ptp(p[:,0])))
                    refinement_limit=min(16.,.35*float(np.ptp(p[:,0])))
                    if len(measured)==1 and np.linalg.norm(np.array(measured[0]['anchor'])-anchor)<=refinement_limit:
                        rim=measured[0];x1,y1,x2,y2=rim['box']
                        best=dict(best,method='verified_patch_with_observed_rim_refinement',
                                  rim_anchor=rim['anchor'],polygon=corners(rim['box']).tolist(),
                                  center=[(x1+x2)/2,(y1+y2)/2],rim_geometry=rim)
                p=np.array(best['polygon']);box=[*p.min(axis=0),*p.max(axis=0)]
                result.update(candidate_valid_2d=True,visible_lower_rim_uv=best['rim_anchor'],
                              visible_cup_center_uv=best['center'],cup_bbox_xyxy_px=list(map(float,box)),
                              selected_reference=best['reference'],annotation_uncertainty_px=5.,
                              anchor_semantics='Approximate visible lower rim, not cup-bottom center or grasp pose',
                              objects={'exposed_cup':{'detected':True,'polygon_px':best['polygon'],
                                       'method':best.get('method','station_registered_independent_cup_and_rim')}})
        result['inference_ms']=round((time.perf_counter()-started)*1000,2)
        result['registration_diagnostics']=self.registration_diagnostics
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
