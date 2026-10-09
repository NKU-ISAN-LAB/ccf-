"""Select deployable checkpoint using train thresholds and DEVELOPMENT validation only."""
import hashlib,json
import cv2,numpy as np
from recognize import ROOT,Detector
from metrics import score,summary
THRESHOLDS=[.005,.01,.025,.05,.075,.1,.125,.15,.175,.2,.225,.25,.3,.35,.4,.45,.5]
if __name__=='__main__':
    candidates=[]
    for name in ['best','last']:
        weights=ROOT/'runs/covered-cup/weights'/(name+'.pt')
        d=Detector(weights=weights,confidence=.001);train=[]
        for path in sorted((ROOT/'dataset/images/train').glob('*.jpg')):
            boxes=[]
            for line in (ROOT/'dataset/labels/train'/(path.stem+'.txt')).read_text().splitlines():
                _,x,y,w,h=map(float,line.split());boxes.append((np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]).tolist())
            train.append(dict(case=path.name,boxes=boxes,detections=d.predict(cv2.imread(str(path)))['detections']))
        thresholds=[]
        for t in THRESHOLDS:
            rows=[dict(case=r['case'],**score([x for x in r['detections'] if x['confidence']>=t],r['boxes'])) for r in train]
            s=summary(rows);s.update(threshold=t,f1=2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn']));thresholds.append(s)
        best=[s for s in thresholds if s['f1']==max(r['f1'] for r in thresholds)];chosen=best[(len(best)-1)//2]
        rows=[]
        for r in json.loads((ROOT/'annotation-audit.json').read_text())['records']:
            if r['split']!='val':continue
            ds=[x for x in d.predict(cv2.imread(r['image']))['detections'] if x['confidence']>=chosen['threshold']]
            rows.append(dict(case=r['frame'],**score(ds,r['boxes_xyxy']),detections=ds))
        val=summary(rows);val['f1']=2*val['tp']/max(1,2*val['tp']+val['fp']+val['fn'])
        for r in json.loads((ROOT/'negative-source-audit.json').read_text())['records']:
            if r['split']!='val':continue
            ds=[x for x in d.predict(cv2.imread(r['image']))['detections'] if x['confidence']>=chosen['threshold']]
            rows.append(dict(case=r['case'],**score(ds,[]),detections=ds))
        val=summary(rows);val['f1']=2*val['tp']/max(1,2*val['tp']+val['fp']+val['fn'])
        candidates.append(dict(name=name,weights=str(weights),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                               training_threshold=chosen,thresholds=thresholds,validation=val,validation_rows=rows,training_records=train))
    selected=max(candidates,key=lambda r:(r['validation']['f1'],r['validation']['all_correct_frames'],r['validation']['mean_matched_iou'] or 0))
    report=dict(selected=selected['name'],selected_weights=selected['weights'],selected_threshold=selected['training_threshold']['threshold'],
                policy='Each checkpoint threshold selected on training only. Choose maximal development-validation multi-instance F1, then all-correct frames, then mean matched IoU. No test frames read.',candidates=candidates)
    (ROOT/'checkpoint-selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({**{k:v for k,v in report.items() if k!='candidates'},'candidates':[{k:v for k,v in r.items() if k not in ['training_records','thresholds','validation_rows']} for r in candidates]},indent=2))
