"""Direct independent cup + holder detection, with class-aware overlapping boxes.

No whole-machine registration; no robot commands; overlap is NOT placement proof.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
NAMES={0:'cup',1:'target'}

def box_iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    intersection=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    return intersection/(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-intersection+1e-9)

class Detector:
    def __init__(self,root=ROOT,weights=None,confidence=None,device='cpu'):
        root=Path(root);self.weights=Path(weights) if weights else root/'weights/best.pt'
        digest=hashlib.sha256(self.weights.read_bytes()).hexdigest()
        if confidence is None:
            config=json.loads((root/'runtime-config.json').read_text())
            if digest not in [config['weights_sha256'],*config.get('compatible_weights_sha256',[])]:
                raise ValueError('Weights and calibrated config do not match')
            thresholds=config['confidence_thresholds']
        elif isinstance(confidence,(int,float)):thresholds={name:float(confidence) for name in NAMES.values()}
        else:thresholds=confidence
        if set(thresholds)!=set(NAMES.values()) or not all(np.isfinite(v) and 0<v<1 for v in thresholds.values()):
            raise ValueError('Need finite cup and target confidence thresholds in (0,1)')
        self.thresholds=thresholds;self.device=device;torch.set_num_threads(4)
        self.model=YOLO(str(self.weights),task='detect')
        if self.model.names!=NAMES:raise ValueError(f'Wrong class mapping: {self.model.names}')
        self.implementation=dict(weights_sha256=digest,
            recognize_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            torch=torch.__version__,opencv=cv2.__version__)

    def predict(self,image,camera_serial=None):
        start=time.perf_counter()
        result=dict(ok=True,algorithm='task2-direct-cup-target-yolo11n-20261008',
            requires_station_registration=False,camera_serial=camera_serial,
            detections=[],objects={name:dict(detected=False,count=0,unique=False,
                bbox_xyxy_px=None,center_uv=None,candidates=[]) for name in NAMES.values()},
            cup_center_uv=None,target_center_uv=None,cup_bbox_xyxy_px=None,target_bbox_xyxy_px=None,
            both_unique_2d=False,stable_2d=False,cup_target_bbox_iou=None,
            placement_confirmed=None,grasp_pose=None,robot_motion_ready=False,
            coordinate_frame='original_color_image_pixels',
            coordinate_semantics='2D bounding-box centers, not actual cup/holder geometric centers or 3D grasp poses',
            confidence_thresholds=self.thresholds,implementation=self.implementation,reasons=[])
        if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[2]!=3 or min(image.shape[:2])<32):
            result.update(ok=False,reasons=['invalid_bgr_uint8_image']);return result
        h,w=image.shape[:2];result['image_size_wh']=[w,h]
        pred=self.model.predict(image,imgsz=640,rect=False,conf=min(self.thresholds.values()),
            iou=.45,agnostic_nms=False,classes=[0,1],max_det=30,device=self.device,verbose=False)[0]
        for raw in pred.boxes.data.cpu().numpy():
            if not np.isfinite(raw).all():continue
            x1,y1,x2,y2,confidence,cls=map(float,raw)
            if cls not in NAMES:continue
            label=NAMES[int(cls)]
            if confidence<self.thresholds[label]:continue
            box=np.clip([x1,y1,x2,y2],[0,0,0,0],[w,h,w,h]).tolist()
            x1,y1,x2,y2=box
            if x2<=x1 or y2<=y1:continue
            result['detections'].append(dict(class_id=int(cls),label=label,confidence=confidence,
                bbox_xyxy_px=box,center_uv=[(x1+x2)/2,(y1+y2)/2]))
        for label in NAMES.values():
            candidates=[d for d in result['detections'] if d['label']==label]
            obj=result['objects'][label];obj.update(count=len(candidates),detected=bool(candidates),
                unique=len(candidates)==1,candidates=candidates)
            if len(candidates)==1:
                chosen=candidates[0]
                obj.update(bbox_xyxy_px=chosen['bbox_xyxy_px'],center_uv=chosen['center_uv'])
                result[label+'_center_uv']=chosen['center_uv']
                result[label+'_bbox_xyxy_px']=chosen['bbox_xyxy_px']
            elif not candidates:result['reasons'].append(label+'_not_detected')
            else:result['reasons'].append('multiple_'+label+'_candidates')
        if all(o['unique'] for o in result['objects'].values()):
            result['both_unique_2d']=True
            result['cup_target_bbox_iou']=box_iou(result['cup_bbox_xyxy_px'],result['target_bbox_xyxy_px'])
        result['inference_ms']=round((time.perf_counter()-start)*1000,2)
        return result

def annotate(image,result):
    out=image.copy()
    for d in result['detections']:
        x1,y1,x2,y2=np.rint(d['bbox_xyxy_px']).astype(int)
        color=(0,150,255) if d['label']=='cup' else (255,220,0)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f"{d['label']} {d['confidence']:.2f}",(max(0,x1),max(12,y1-4)),0,.42,color,1)
    cv2.putText(out,'2D DETECTIONS ONLY / OVERLAP != PLACEMENT',(8,out.shape[0]-10),0,.43,(0,0,220),1)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--weights',type=Path)
    p.add_argument('--confidence',type=float);p.add_argument('--camera-serial')
    p.add_argument('--output',type=Path);p.add_argument('--json-output',type=Path)
    a=p.parse_args();im=cv2.imread(str(a.image))
    if im is None:raise SystemExit('Cannot read input image')
    if a.output and a.output.resolve()==a.image.resolve():raise SystemExit('Refusing to overwrite original image')
    result=Detector(weights=a.weights,confidence=a.confidence).predict(im,a.camera_serial)
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False);print(text)
    if a.output:
        if not cv2.imwrite(str(a.output),annotate(im,result)):raise SystemExit('Cannot save preview')
    if a.json_output:a.json_output.write_text(text)
