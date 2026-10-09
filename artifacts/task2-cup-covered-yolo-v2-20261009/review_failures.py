"""Render preserved failures, without modifying labels, thresholds or weights."""
import json
import cv2,numpy as np
from recognize import ROOT,annotate
from evaluate import variants
if __name__=='__main__':
    audit=json.loads((ROOT/'annotation-audit.json').read_text())
    stress=json.loads((ROOT/'stress-report.json').read_text())
    for r in stress:
        if r['all_correct']:continue
        source=next(s for s in audit['records'] if s['frame']==r['case'].split('_')[0])
        image=cv2.imread(source['image'])
        kind,image,boxes=next(v for v in variants(image,source['boxes_xyxy']) if v[0]==r['kind'])
        preview=annotate(image,{'detections':r['detections']})
        for x1,y1,x2,y2 in np.rint(boxes).astype(int):cv2.rectangle(preview,(x1,y1),(x2,y2),(255,100,0),1)
        cv2.imwrite(str(ROOT/('failure_'+r['case']+'.jpg')),preview)
