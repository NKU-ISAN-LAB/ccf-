"""Choose threshold solely on training-derived multi-target examples."""
import hashlib,json
import cv2,numpy as np
from recognize import Detector,ROOT
from metrics import score
if __name__=='__main__':
    detector=Detector(confidence=.001);records=[]
    for path in sorted((ROOT/'dataset/images/train').glob('*.jpg')):
        boxes=[]
        for line in (ROOT/'dataset/labels/train'/(path.stem+'.txt')).read_text().splitlines():
            _,x,y,w,h=map(float,line.split());boxes.append((np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]).tolist())
        records.append(dict(image=path.name,boxes=boxes,detections=detector.predict(cv2.imread(str(path)))['detections']))
    scores=[]
    for threshold in [.005,.01,.025,.05,.075,.1,.125,.15,.175,.2,.225,.25,.3,.35,.4,.45,.5]:
        rs=[score([d for d in r['detections'] if d['confidence']>=threshold],r['boxes']) for r in records]
        tp=sum(r['tp'] for r in rs);fp=sum(r['fp'] for r in rs);fn=sum(r['fn'] for r in rs)
        scores.append(dict(threshold=threshold,tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn)))
    best=[s for s in scores if s['f1']==max(v['f1'] for v in scores)];chosen=best[(len(best)-1)//2]
    expected=json.loads((ROOT/'checkpoint-selection.json').read_text())
    assert chosen['threshold']==expected['selected_threshold']
    report=dict(source=f'{len(records)} training-derived images only',selection='max multi-object F1 at IoU>=0.5, middle tested threshold among ties; no validation/test tuning',
                selected=chosen,thresholds=scores,records=records,weights_sha256=detector.implementation['weights_sha256'])
    (ROOT/'threshold-calibration.json').write_text(json.dumps(report,indent=2))
    config=dict(confidence_threshold=chosen['threshold'],weights_sha256=report['weights_sha256'],
                compatible_weights_sha256=[hashlib.sha256((ROOT/'weights/best.onnx').read_bytes()).hexdigest()],source='training-only threshold-calibration.json')
    (ROOT/'runtime-config.json').write_text(json.dumps(config,indent=2));print(json.dumps(chosen))
