"""Select a threshold from training-derived examples only. No test access."""
import hashlib,json
import cv2
import numpy as np
from recognize import Detector,ROOT
from metrics import score

def main():
    detector=Detector(confidence=.01)
    records=[]
    for path in sorted((ROOT/'dataset/images/train').glob('*.jpg')):
        text=(ROOT/'dataset/labels/train'/(path.stem+'.txt')).read_text().strip()
        box=None
        if text:
            _,x,y,w,h=map(float,text.split())
            box=(np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]).tolist()
        records.append(dict(image=path.name,ground_truth=box,detections=detector.predict(cv2.imread(str(path)))['detections']))
    scores=[]
    for threshold in [.05,.075,.1,.125,.15,.175,.2,.225,.25,.3,.35,.4,.45,.5]:
        rows=[score([d for d in r['detections'] if d['confidence']>=threshold],r['ground_truth']) for r in records]
        tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
        scores.append(dict(threshold=threshold,tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn)))
    best=[s for s in scores if s['f1']==max(r['f1'] for r in scores)]
    chosen=best[(len(best)-1)//2]
    report=dict(source=f'{len(records)} training-derived images only; no validation or test images',
                selection='max F1 IoU>=0.5, middle tested threshold among ties; not probability calibration',
                selected=chosen,thresholds=scores,records=records,
                weights_sha256=detector.implementation['weights_sha256'])
    (ROOT/'threshold-calibration.json').write_text(json.dumps(report,indent=2))
    config=dict(confidence_threshold=chosen['threshold'],weights_sha256=report['weights_sha256'],
                compatible_weights_sha256=[hashlib.sha256((ROOT/'weights/best.onnx').read_bytes()).hexdigest()] if (ROOT/'weights/best.onnx').exists() else [],
                source='training-only threshold-calibration.json')
    (ROOT/'runtime-config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(chosen))

if __name__=='__main__':main()
