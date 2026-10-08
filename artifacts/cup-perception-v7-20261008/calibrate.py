"""Choose confidence using TRAINING-derived cases only, never validation frames."""
import hashlib,json
import cv2
import numpy as np
from recognize import ROOT,Detector
from evaluate import iou

detector=Detector(confidence=.01)
records=[]
for path in sorted((ROOT/'dataset/images/train').glob('*.jpg')):
    label=(ROOT/'dataset/labels/train'/(path.stem+'.txt')).read_text().strip()
    gt=None
    if label:
        _,x,y,w,h=map(float,label.split())
        gt=(np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]).tolist()
    result=detector.predict(cv2.imread(str(path)))
    records.append(dict(image=path.name,ground_truth=gt,detections=result['detections']))
scores=[]
for threshold in [.05,.075,.1,.125,.15,.175,.2,.225,.25,.3,.35,.4,.45,.5]:
    tp=fp=fn=0
    for r in records:
        ds=[d for d in r['detections'] if d['confidence']>=threshold]
        matched=bool(r['ground_truth'] is not None and any(iou(d['bbox_xyxy_px'],r['ground_truth'])>=.5 for d in ds))
        tp+=int(matched);fp+=len(ds)-int(matched);fn+=int(r['ground_truth'] is not None and not matched)
    scores.append(dict(threshold=threshold,tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn)))
# On tied F1, use the midpoint of the contiguous best threshold range, avoiding
# a threshold at the edge of observed positive confidence. Not probability calibration.
best_f1=max(s['f1'] for s in scores)
best=[s for s in scores if s['f1']==best_f1]
selected=best[(len(best)-1)//2]
report=dict(source='100 training-derived images only (80 positive,20 synthetic negative)',
            selection='highest F1 at IoU>=0.5; middle tested threshold among ties',
            selected=selected,thresholds=scores,records=records,
            weights_sha256=detector.implementation['weights_sha256'])
(ROOT/'threshold-calibration.json').write_text(json.dumps(report,indent=2))
(ROOT/'runtime-config.json').write_text(json.dumps(dict(confidence_threshold=selected['threshold'],
    weights_sha256=report['weights_sha256'],
    compatible_weights_sha256=[hashlib.sha256((ROOT/'weights/best.onnx').read_bytes()).hexdigest()] if (ROOT/'weights/best.onnx').exists() else [],
    source='training-only threshold-calibration.json'),indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
