"""Small visually reviewed open-cup/empty-holder probes, evaluation only."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from recognize import ROOT,Detector,annotate
if __name__=='__main__':
    audit=json.loads(Path('/home/vnv/task2-cup-target-yolo-20261008/annotation-audit.json').read_text())
    r=next(r for r in audit['records'] if r['filename']=='001_old_s02_episode_000000_f00045.jpg')
    path=Path(audit['source_image_directory'])/r['filename']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['source_image_sha256']
    image=cv2.imread(str(path));detector=Detector();rows=[]
    out=ROOT/'other-object-probes';out.mkdir(exist_ok=True)
    for b in r['boxes']:
        x1,y1,x2,y2=np.rint(b['bbox_xyxy_px']).astype(int)
        crop=image[max(0,y1-16):min(480,y2+16),max(0,x1-16):min(640,x2+16)]
        for scale in [1.,2.]:
            patch=cv2.resize(crop,None,fx=scale,fy=scale)
            h,w=patch.shape[:2];canvas=np.full_like(image,127)
            left=(640-w)//2;top=(480-h)//2;canvas[top:top+h,left:left+w]=patch
            result=detector.predict(canvas);name=b['label']+f'_scale{scale}'
            cv2.imwrite(str(out/(name+'.jpg')),annotate(canvas,result))
            rows.append(dict(case=name,original_label=b['label'],source=str(path),
                             source_crop_xyxy=[int(v) for v in [max(0,x1-16),max(0,y1-16),min(640,x2+16),min(480,y2+16)]],
                             detections=result['detections'],passed=not result['detections']))
    report=dict(cases=len(rows),passed=sum(r['passed'] for r in rows),records=rows,
                qualification='Regression of previous failure. Source episode is now part of training; NOT independent negative validation. Only one open cup and one empty holder at two scales.')
    (ROOT/'prior-open-cup-checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
