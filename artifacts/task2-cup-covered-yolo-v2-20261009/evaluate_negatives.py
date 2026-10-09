"""Separate train/val/held-out negative cases; no threshold tuning here."""
import json
import cv2
from recognize import ROOT,Detector,annotate
if __name__=='__main__':
    records=json.loads((ROOT/'negative-source-audit.json').read_text())['records']
    detector=Detector();rows=[];out=ROOT/'negative-previews';out.mkdir(exist_ok=True)
    for r in records:
        im=cv2.imread(r['image']);p=detector.predict(im)
        rows.append(dict(case=r['case'],split=r['split'],source=r['source'],detections=p['detections'],passed=not p['detections']))
        cv2.imwrite(str(out/(r['case']+'.jpg')),annotate(im,p))
    report=dict(groups={s:dict(cases=sum(r['split']==s for r in rows),passed=sum(r['split']==s and r['passed'] for r in rows),
                                failures=[r['case'] for r in rows if r['split']==s and not r['passed']]) for s in ['train','val','negative_test']},records=rows,
                qualification='Only four visually checked source frames from different old episodes, each full image plus four rendered crops. Negative-test source episode 000031 never used in training/selection. Small diagnostic set, not comprehensive open-cup/holder rejection accuracy.')
    (ROOT/'other-object-checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['groups'],indent=2))
