"""Standalone multi-instance cup_covered detector. No cup/target class reuse."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
NAMES={0:'cup_covered'}
REVISION='task2-cup-covered-yolo11n-v2-hardneg-20261009'

class Detector:
    def __init__(self,root=ROOT,weights=None,confidence=None,device='cpu'):
        root=Path(root);self.weights=Path(weights) if weights else root/'weights/best.pt'
        digest=hashlib.sha256(self.weights.read_bytes()).hexdigest()
        if confidence is None:
            config=json.loads((root/'runtime-config.json').read_text())
            if digest not in [config['weights_sha256'],*config.get('compatible_weights_sha256',[])]:
                raise ValueError('Weights do not match covered-cup threshold config')
            confidence=config['confidence_threshold']
        if not isinstance(confidence,(int,float)) or not np.isfinite(confidence) or not 0<confidence<1:
            raise ValueError('Confidence must be finite and in (0,1)')
        self.confidence=float(confidence);self.device=device;torch.set_num_threads(4)
        self.model=YOLO(str(self.weights),task='detect')
        if self.model.names!=NAMES:raise ValueError(f'Wrong class mapping: {self.model.names}; expected cup_covered only')
        self.implementation=dict(weights_sha256=digest,recognize_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 torch=torch.__version__,opencv=cv2.__version__)

    def predict(self,image,camera_serial=None):
        start=time.perf_counter()
        result=dict(ok=True,algorithm=REVISION,camera_serial=camera_serial,requires_station_registration=False,
            detections=[],objects={'cup_covered':dict(detected=False,count=0,unique=False,candidates=[])},
            covered_cup_centers_uv=[],single_covered_cup_center_uv=None,
            selected_grasp_candidate=None,grasp_pose=None,robot_motion_ready=False,stable_2d=False,
            seal_quality_verified=None,coordinate_frame='original_color_image_pixels',
            coordinate_semantics='Centers of user-labeled visible cup-cover regions, not full-body or 3D grasp centers',
            confidence_threshold=self.confidence,implementation=self.implementation,reasons=[])
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<32:
            result.update(ok=False,reasons=['invalid_bgr_uint8_image']);return result
        h,w=image.shape[:2];result['image_size_wh']=[w,h]
        pred=self.model.predict(image,imgsz=640,rect=False,conf=self.confidence,iou=.45,
                                classes=[0],max_det=30,device=self.device,verbose=False)[0]
        for raw in pred.boxes.data.cpu().numpy():
            if not np.isfinite(raw).all():continue
            x1,y1,x2,y2,confidence,cls=map(float,raw)
            if cls!=0 or confidence<self.confidence:continue
            box=np.clip([x1,y1,x2,y2],[0,0,0,0],[w,h,w,h]).tolist()
            x1,y1,x2,y2=box
            if x2<=x1 or y2<=y1:continue
            result['detections'].append(dict(class_id=0,label='cup_covered',confidence=confidence,
                bbox_xyxy_px=box,center_uv=[(x1+x2)/2,(y1+y2)/2]))
        ds=result['detections']
        result['objects']['cup_covered'].update(detected=bool(ds),count=len(ds),unique=len(ds)==1,candidates=ds)
        result['covered_cup_centers_uv']=[d['center_uv'] for d in ds]
        if len(ds)==1:result['single_covered_cup_center_uv']=ds[0]['center_uv']
        elif not ds:result['reasons'].append('cup_covered_not_detected')
        else:result['reasons'].append('multiple_cup_covered_candidates_no_grasp_selection')
        result['inference_ms']=round((time.perf_counter()-start)*1000,2)
        return result

def annotate(image,result):
    out=image.copy()
    for i,d in enumerate(result['detections'],1):
        x1,y1,x2,y2=np.rint(d['bbox_xyxy_px']).astype(int)
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,220,0),2)
        cv2.putText(out,f"cup_covered {i} {d['confidence']:.2f}",(max(0,x1),max(12,y1-4)),0,.4,(0,180,0),1)
    cv2.putText(out,'CUP_COVERED ONLY / 2D COVER REGIONS / NOT A GRASP POSE',(8,out.shape[0]-10),0,.38,(0,0,220),1)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--weights',type=Path)
    p.add_argument('--confidence',type=float);p.add_argument('--camera-serial');p.add_argument('--output',type=Path);p.add_argument('--json-output',type=Path)
    a=p.parse_args()
    for out in [a.output,a.json_output]:
        if out and out.resolve()==a.image.resolve():raise SystemExit('Refusing to overwrite input image')
    if a.output and a.json_output and a.output.resolve()==a.json_output.resolve():raise SystemExit('Preview and JSON need different paths')
    image=cv2.imread(str(a.image))
    if image is None:raise SystemExit('Cannot read input image')
    r=Detector(weights=a.weights,confidence=a.confidence).predict(image,a.camera_serial)
    text=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False);print(text)
    if a.output and not cv2.imwrite(str(a.output),annotate(image,r)):raise SystemExit('Cannot save preview')
    if a.json_output:a.json_output.write_text(text)
