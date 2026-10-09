import json
import cv2,numpy as np
from recognize import ROOT,Detector
if __name__=='__main__':
    audit=json.loads((ROOT/'annotation-audit.json').read_text());pt=Detector();onnx=Detector(weights=ROOT/'weights/best.onnx');rows=[]
    for r in audit['records']:
        image=cv2.imread(r['image']);a=pt.predict(image);b=onnx.predict(image)
        same=len(a['detections'])==len(b['detections']);coords=[];scores=[]
        if same:
            # Deterministic location order rather than backend-dependent tied-score order.
            ds=sorted(a['detections'],key=lambda d:tuple(d['center_uv']))
            es=sorted(b['detections'],key=lambda d:tuple(d['center_uv']))
            for d,e in zip(ds,es):
                coords.append(float(np.max(np.abs(np.array(d['bbox_xyxy_px'])-e['bbox_xyxy_px']))))
                scores.append(abs(d['confidence']-e['confidence']))
        rows.append(dict(frame=r['frame'],same_count=same,max_coordinate_difference_px=max(coords,default=0),
                         max_confidence_difference=max(scores,default=0),passed=same and max(coords,default=0)<.1 and max(scores,default=0)<.001))
    report=dict(frames=len(rows),passed=sum(r['passed'] for r in rows),pt=pt.implementation,onnx=onnx.implementation,records=rows)
    (ROOT/'export-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    raise SystemExit(0 if rows and all(r['passed'] for r in rows) else 1)
