"""Direct exposed-cup detection. No machine/reference/registration prerequisite.

Bounding-box centers are 2D observations, NOT measured cup rims or grasp poses.
Each worker should own its Detector. No cameras or robot devices are opened here.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
REVISION='v9-supplement-direct-exposed-cup-20261009'

class StabilityGate:
    """Three distinct increasing frames; reset on missing/ambiguous detections."""
    def __init__(self):
        self.points=deque(maxlen=3)
        self.last_id=self.last_time=None

    def update(self,result,frame_id,observed_time):
        result['stable_2d']=False
        point=result.get('cup_center_uv')
        valid_point=(isinstance(point,(list,tuple)) and len(point)==2
                     and all(isinstance(x,(int,float)) and np.isfinite(x) for x in point))
        if (not result.get('candidate_valid_2d') or not valid_point or type(frame_id) is not int
            or type(observed_time) not in (int,float) or not np.isfinite(observed_time)
            or (self.last_id is not None and frame_id<=self.last_id)
            or (self.last_time is not None and not 0<observed_time-self.last_time<=2.)):
            self.points.clear();self.last_id=self.last_time=None
            result['stability_reason']='invalid_duplicate_or_discontinuous_frame'
            return result
        self.last_id,self.last_time=frame_id,observed_time
        self.points.append(point)
        pts=np.asarray(self.points)
        deviation=float(np.linalg.norm(pts-np.median(pts,axis=0),axis=1).max())
        result.update(stable_2d=len(pts)==3 and deviation<=3.,
                      stability_samples=len(pts),stability_deviation_px=deviation,
                      stability_basis='detected_bbox_center_not_rim_or_grasp')
        return result

class Detector:
    def __init__(self,root=ROOT,weights=None,confidence=None,device='cpu'):
        self.root=Path(root)
        self.weights=Path(weights) if weights else self.root/'weights/best.pt'
        if not self.weights.is_file():raise FileNotFoundError(self.weights)
        weights_hash=hashlib.sha256(self.weights.read_bytes()).hexdigest()
        if confidence is None:
            config=json.loads((self.root/'runtime-config.json').read_text())
            if weights_hash not in [config['weights_sha256'],*config.get('compatible_weights_sha256',[])]:
                raise ValueError('Model does not match calibrated runtime-config.json')
            confidence=config['confidence_threshold']
        if not 0<confidence<1:raise ValueError('confidence must be in (0,1)')
        torch.set_num_threads(4)
        self.model=YOLO(str(self.weights),task='detect')
        if self.model.names!={0:'exposed_cup'}:
            raise ValueError(f'Wrong model classes: {self.model.names}')
        self.confidence=float(confidence);self.device=device
        self.implementation={'weights_sha256':weights_hash,
                             'recognize_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'torch':torch.__version__,'opencv':cv2.__version__}

    def predict(self,image,camera_serial=None):
        start=time.perf_counter()
        result=dict(ok=True,recognition_revision=REVISION,algorithm='direct_yolo_detection',
                    requires_station_registration=False,camera_serial=camera_serial,
                    candidate_valid_2d=False,stable_2d=False,detections=[],
                    cup_bbox_xyxy_px=None,cup_center_uv=None,visible_cup_center_uv=None,
                    coordinate_frame='color_image_pixels',visible_lower_rim_uv=None,
                    camera_surface_xyz_m=None,robot_surface_xyz_m=None,
                    grasp_pose=None,robot_motion_ready=False,objects={},reasons=[],
                    confidence_threshold=self.confidence,implementation=self.implementation,
                    coordinate_semantics='bbox center in original RGB pixels; not a measured rim or 3D grasp',
                    calibration_status='no_verified_aligned_depth_or_robot_extrinsics')
        if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[2]!=3 or min(image.shape[:2])<32):
            result['ok']=False
            result['reasons']=['invalid_bgr_uint8_image'];return result
        h,w=image.shape[:2]
        result['input_diagnostics']={'image_size_wh':[w,h],
            'decoded_bgr_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
        predicted=self.model.predict(source=image,imgsz=640,conf=self.confidence,
                                     iou=.45,classes=[0],max_det=10,device=self.device,
                                     rect=False,verbose=False)[0]
        for raw in predicted.boxes.data.cpu().numpy():
            x1,y1,x2,y2,score,cls=map(float,raw)
            box=np.array([x1,y1,x2,y2])
            if not np.isfinite(box).all() or not np.isfinite(score):continue
            box=np.clip(box,[0,0,0,0],[w,h,w,h])
            x1,y1,x2,y2=box.tolist()
            if x2<=x1 or y2<=y1:continue
            result['detections'].append(dict(label='exposed_cup',confidence=score,
                bbox_xyxy_px=box.tolist(),center_uv=[(x1+x2)/2,(y1+y2)/2]))
        if len(result['detections'])==1:
            d=result['detections'][0]
            result.update(candidate_valid_2d=True,cup_bbox_xyxy_px=d['bbox_xyxy_px'],
                          cup_center_uv=d['center_uv'],visible_cup_center_uv=d['center_uv'])
            x1,y1,x2,y2=d['bbox_xyxy_px']
            result['objects']['exposed_cup']={**d,'detected':True,
                'polygon_px':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],'method':'yolo11n_user_manual_labels'}
        else:
            result['reasons']=['no_exposed_cup_detected' if not result['detections'] else 'multiple_cup_candidates']
        result['inference_ms']=round((time.perf_counter()-start)*1000,2)
        return result

def annotate(image,result):
    out=image.copy()
    for d in result['detections']:
        x1,y1,x2,y2=np.rint(d['bbox_xyxy_px']).astype(int)
        color=(0,220,0) if result['candidate_valid_2d'] else (0,180,255)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f"exposed_cup {d['confidence']:.2f}",(max(0,x1),max(15,y1-5)),0,.45,color,1)
    cv2.putText(out,'DIRECT CUP / 2D BOX ONLY - NOT A GRASP POSE',(8,out.shape[0]-10),0,.43,(0,0,220),1)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('image',type=Path);p.add_argument('--weights',type=Path)
    p.add_argument('--camera-serial');p.add_argument('--confidence',type=float)
    p.add_argument('--output',type=Path);p.add_argument('--json-output',type=Path)
    a=p.parse_args();im=cv2.imread(str(a.image))
    if im is None:raise SystemExit(f'Cannot read image: {a.image}')
    detector=Detector(weights=a.weights,confidence=a.confidence)
    result=detector.predict(im,a.camera_serial)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    print(payload)
    if a.output:
        if a.output.resolve()==a.image.resolve():raise SystemExit('Refusing to overwrite source image')
        if not cv2.imwrite(str(a.output),annotate(im,result)):raise SystemExit('Cannot save preview')
    if a.json_output:a.json_output.write_text(payload)
