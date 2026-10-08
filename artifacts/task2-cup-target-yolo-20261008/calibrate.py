"""Select independent class thresholds using training-derived frames only."""
import hashlib,json
import cv2
import numpy as np
from recognize import Detector,ROOT,NAMES
from metrics import score

detector=Detector(confidence=.01);rows=[]
for path in sorted((ROOT/'dataset/images/train').glob('*.jpg')):
    gt=[]
    for line in (ROOT/'dataset/labels/train'/(path.stem+'.txt')).read_text().splitlines():
        cls,x,y,w,h=map(float,line.split())
        gt.append(dict(label=NAMES[int(cls)],bbox_xyxy_px=(np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]).tolist()))
    result=detector.predict(cv2.imread(str(path)))
    rows.append(dict(filename=path.name,ground_truth=gt,detections=result['detections']))
thresholds={};scans={}
for label in NAMES.values():
    scan=[]
    for threshold in [.03,.05,.075,.1,.125,.15,.175,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8]:
        tp=fp=fn=0
        for r in rows:
            s=score([d for d in r['detections'] if d['confidence']>=threshold],r['ground_truth'])[label]
            tp+=s['tp'];fp+=s['fp'];fn+=s['fn']
        scan.append(dict(threshold=threshold,tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn)))
    best=max(s['f1'] for s in scan);ties=[s for s in scan if s['f1']==best]
    thresholds[label]=ties[(len(ties)-1)//2]['threshold'];scans[label]=scan
config=dict(confidence_thresholds=thresholds,weights_sha256=detector.implementation['weights_sha256'],
    compatible_weights_sha256=[hashlib.sha256((ROOT/'weights/best.onnx').read_bytes()).hexdigest()] if (ROOT/'weights/best.onnx').exists() else [],
    source='Training-only per-class F1 threshold selection; not probability calibration',agnostic_nms=False,nms_iou=.45)
(ROOT/'runtime-config.json').write_text(json.dumps(config,indent=2))
(ROOT/'threshold-calibration.json').write_text(json.dumps(dict(config=config,scans=scans,records=rows),indent=2))
print(json.dumps(dict(config=config,scans=scans),indent=2))
