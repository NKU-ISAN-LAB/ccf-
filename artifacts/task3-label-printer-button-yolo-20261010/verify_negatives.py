"""Additional read-only negative regression: separate cup-covered capture.

These scenes were manually reviewed for absence of the target label printer and
its buttons; not used for training, checkpoint or threshold selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from recognize import Detector, annotate

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/vnv/yizhi/yzb2026-competition-ws/CCF数据集-20261009-154730/color')
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-only',action='store_true')
    args=parser.parse_args()
    paths=sorted(SOURCE.glob('*.jpg'))
    assert len(paths)==20
    sheet=np.full((1088,1600,3),235,np.uint8)
    for i,path in enumerate(paths):
        im=cv2.imread(str(path));row,col=divmod(i,5)
        sheet[row*272:row*272+240,col*320:(col+1)*320]=cv2.resize(im,(320,240))
        cv2.putText(sheet,path.name,(col*320+5,row*272+258),0,.5,(0,0,0),1)
    cv2.imwrite(str(ROOT/'real-negative-review.jpg'),sheet)
    if not args.review_only:
        detector=Detector();rows=[]
        for path in paths:
            result=detector.predict(cv2.imread(str(path)))
            rows.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             passed=len(result['detections'])==0,result=result))
        report=dict(frames=len(rows),passed=sum(r['passed'] for r in rows),rows=rows,
                    meaning='No target label printer or its buttons. One separate physical scene with 20 related frames, not 20 independent environments. Not used for training/selection.',
                    weights_sha256=detector.implementation['weights_sha256'])
        (ROOT/'real-negative-verification.json').write_text(json.dumps(report,indent=2))
        failures=[r for r in rows if not r['passed']]
        if failures:
            canvas=np.full((len(failures)*510,640,3),235,np.uint8)
            for i,r in enumerate(failures):
                im=annotate(cv2.imread(r['source']),r['result'])
                canvas[i*510:i*510+480]=im
                cv2.putText(canvas,Path(r['source']).name+' NEGATIVE / FALSE PRINTER',(8,i*510+500),0,.5,(0,0,220),1)
            cv2.imwrite(str(ROOT/'negative-failures.jpg'),canvas)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
