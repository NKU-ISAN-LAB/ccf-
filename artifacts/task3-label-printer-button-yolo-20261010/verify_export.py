import json
from pathlib import Path
import cv2
import numpy as np
from recognize import Detector, box_iou

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    pt=Detector();onnx=Detector(weights=ROOT/'weights/best.onnx')
    records=json.loads((ROOT/'annotation-audit.json').read_text())['records']
    rows=[]
    for r in records:
        im=cv2.imread(r['source'])
        a=pt.predict(im)['detections'];b=onnx.predict(im)['detections']
        used=set();deltas=[];ok=len(a)==len(b)
        for d in a:
            matches=[(box_iou(d['bbox_xyxy_px'],e['bbox_xyxy_px']),j) for j,e in enumerate(b) if j not in used and d['class_id']==e['class_id']]
            iou,j=max(matches,default=(0,None))
            if j is None:
                ok=False;continue
            used.add(j)
            boxdelta=float(np.max(np.abs(np.array(d['bbox_xyxy_px'])-b[j]['bbox_xyxy_px'])))
            confdelta=abs(d['confidence']-b[j]['confidence'])
            ok=ok and boxdelta<1.5 and confdelta<.025
            deltas.append(dict(max_box_delta_px=boxdelta,confidence_delta=confdelta,iou=iou))
        rows.append(dict(filename=r['filename'],passed=bool(ok),deltas=deltas,pt_count=len(a),onnx_count=len(b)))
    report=dict(frames=len(rows),passed=sum(r['passed'] for r in rows),rows=rows,
                tolerance='same class/count, box coordinate delta <1.5px and confidence delta <0.025; original-frame end-to-end including crop proposal rounding')
    (ROOT/'onnx-parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if report['passed']!=report['frames']:
        raise SystemExit('PT/ONNX parity failed; inspect onnx-parity.json')
