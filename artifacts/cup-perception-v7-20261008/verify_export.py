"""Compare packaged PyTorch and ONNX inference on unchanged original images."""
import argparse,json
import cv2,numpy as np
from recognize import Detector,ROOT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args()
    from pathlib import Path
    pt=Detector();onnx=Detector(weights=ROOT/'weights/best.onnx')
    rows=[]
    for path in sorted((Path(a.data)/'color').glob('*.jpg')):
        image=cv2.imread(str(path));x=pt.predict(image);y=onnx.predict(image)
        same=len(x['detections'])==len(y['detections'])
        errors=[];scores=[]
        if same:
            for d,e in zip(x['detections'],y['detections']):
                errors.append(float(np.max(np.abs(np.array(d['bbox_xyxy_px'])-e['bbox_xyxy_px']))))
                scores.append(abs(d['confidence']-e['confidence']))
        rows.append(dict(frame=path.stem,same_detection_count=same,
            max_box_coordinate_difference_px=max(errors,default=0),
            max_confidence_difference=max(scores,default=0),
            passed=same and max(errors,default=0)<.1 and max(scores,default=0)<.001))
    report=dict(frames=len(rows),passed=sum(r['passed'] for r in rows),
                pt=pt.implementation,onnx=onnx.implementation,records=rows)
    (ROOT/'export-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    raise SystemExit(0 if rows and all(r['passed'] for r in rows) else 1)
