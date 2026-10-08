"""Verify class-aware PyTorch/ONNX parity on every labeled original image."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from recognize import Detector,ROOT
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);a=p.parse_args()
    pt=Detector();onnx=Detector(weights=ROOT/'weights/best.onnx');rows=[]
    audit=json.loads((ROOT/'annotation-audit.json').read_text())
    for r in audit['records']:
        image=cv2.imread(str(a.images/r['filename']))
        x=pt.predict(image);y=onnx.predict(image)
        order=lambda d:(d['class_id'],-d['confidence'])
        ds=sorted(x['detections'],key=order);es=sorted(y['detections'],key=order)
        same=[d['class_id'] for d in ds]==[e['class_id'] for e in es]
        errors=[];scores=[]
        if same:
            for d,e in zip(ds,es):
                errors.append(float(np.max(np.abs(np.array(d['bbox_xyxy_px'])-e['bbox_xyxy_px']))))
                scores.append(abs(d['confidence']-e['confidence']))
        rows.append(dict(filename=r['filename'],same_class_sequence=same,
            max_box_difference_px=max(errors,default=0),max_confidence_difference=max(scores,default=0),
            passed=same and max(errors,default=0)<.1 and max(scores,default=0)<.001))
    report=dict(frames=len(rows),passed=sum(r['passed'] for r in rows),pt=pt.implementation,
        onnx=onnx.implementation,records=rows)
    (ROOT/'export-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    raise SystemExit(0 if rows and all(r['passed'] for r in rows) else 1)
