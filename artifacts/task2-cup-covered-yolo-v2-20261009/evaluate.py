"""Full-frame multi-instance evaluation; reserve final five frames for test."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from recognize import Detector,ROOT,annotate
from metrics import score,summary

def warp(image,boxes,matrix):
    if matrix.shape==(2,3):matrix=np.vstack([matrix,[0,0,1]])
    transformed=[]
    for x1,y1,x2,y2 in boxes:
        pts=cv2.perspectiveTransform(np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]),matrix.astype(float))[0]
        box=np.r_[pts.min(axis=0),pts.max(axis=0)]
        assert (box[:2]>=0).all() and (box[2:]<=[640,480]).all()
        transformed.append(box.tolist())
    return cv2.warpPerspective(image,matrix,(640,480),borderValue=(114,114,114)),transformed

def variants(image,boxes):
    for angle in [-15,15]:yield f'rotate_{angle}',*warp(image,boxes,cv2.getRotationMatrix2D((320,240),angle,1))
    for scale in [.75,1.15]:yield f'scale_{scale}',*warp(image,boxes,cv2.getRotationMatrix2D((320,240),0,scale))
    src=np.float32([[0,0],[639,0],[639,479],[0,479]])
    dst=np.float32([[40,20],[610,40],[625,440],[10,465]])
    yield 'perspective',*warp(image,boxes,cv2.getPerspectiveTransform(src,dst))
    yield 'dim',np.clip(image.astype(float)*.75+8,0,255).astype(np.uint8),boxes
    yield 'bright',np.clip(image.astype(float)*1.15+5,0,255).astype(np.uint8),boxes
    yield 'blur',cv2.GaussianBlur(image,(3,3),.8),boxes
    _,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,60])
    yield 'jpeg60',cv2.imdecode(encoded,cv2.IMREAD_COLOR),boxes
    isolated=np.full_like(image,127)
    for x1,y1,x2,y2 in np.rint(boxes).astype(int):
        xa,ya,xb,yb=max(0,x1-14),max(0,y1-14),min(640,x2+14),min(480,y2+14)
        isolated[ya:yb,xa:xb]=image[ya:yb,xa:xb]
    yield 'context_removed',isolated,boxes

if __name__=='__main__':
    audit=json.loads((ROOT/'annotation-audit.json').read_text());detector=Detector()
    rows=[];stress=[];negatives=[];previews=ROOT/'previews';previews.mkdir(exist_ok=True)
    for r in audit['records']:
        path=Path(r['image']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256']
        image=cv2.imread(str(path));boxes=r['boxes_xyxy'];result=detector.predict(image)
        row=dict(case=r['frame'],split=r['split'],**score(result['detections'],boxes),result=result)
        rows.append(row);im=annotate(image,result)
        for b in boxes:
            x1,y1,x2,y2=np.rint(b).astype(int);cv2.rectangle(im,(x1,y1),(x2,y2),(255,100,0),1)
        cv2.imwrite(str(previews/(r['frame']+'.jpg')),im)
        if r['split']=='test':
            for kind,im,gt in variants(image,boxes):
                ds=detector.predict(im)['detections']
                stress.append(dict(case=r['frame']+'_'+kind,kind=kind,boxes=gt,**score(ds,gt),detections=ds))
        absent=image.copy()
        for x1,y1,x2,y2 in np.rint(boxes).astype(int):
            absent[max(0,y1-5):min(480,y2+5),max(0,x1-5):min(640,x2+5)]=40
        ds=detector.predict(absent)['detections']
        negatives.append(dict(case=r['frame']+'_synthetic_absent',detections=ds,passed=not ds))
    for name,im in [('black',np.zeros((480,640,3),np.uint8)),('white',np.full((480,640,3),255,np.uint8)),
                    ('noise',np.random.default_rng(309).integers(0,256,(480,640,3),dtype=np.uint8))]:
        ds=detector.predict(im)['detections'];negatives.append(dict(case=name,detections=ds,passed=not ds))
    report=dict(classes={'0':'cup_covered'},confidence_threshold=detector.confidence,implementation=detector.implementation,
        original_replay=summary(rows),train=summary([r for r in rows if r['split']=='train']),
        validation=summary([r for r in rows if r['split']=='val']),reserved_test=summary([r for r in rows if r['split']=='test']),
        transformed_reserved_test=summary(stress),by_transform={k:summary([r for r in stress if r['kind']==k]) for k in sorted({r['kind'] for r in stress})},
        synthetic_negatives=dict(frames=len(negatives),passed=sum(r['passed'] for r in negatives),failures=[r for r in negatives if not r['passed']]),
        qualification='20 same-session frames; 0016-0020 excluded from train/checkpoint/threshold, but already evaluated in the initial model and now reused as positive regression. NOT independent field accuracy. Real-source negatives are separately audited and evaluated. Class identifies cover appearance, not sealing quality or 3D grasp.')
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2))
    (ROOT/'replay.json').write_text(json.dumps(rows,indent=2));(ROOT/'stress-report.json').write_text(json.dumps(stress,indent=2))
    (ROOT/'negative-checks.json').write_text(json.dumps(negatives,indent=2))
    sheet=np.full((1056,1600,3),235,np.uint8)
    for i,r in enumerate(rows):
        im=cv2.imread(str(previews/(r['case']+'.jpg')));row,col=divmod(i,5)
        sheet[row*264:row*264+240,col*320:(col+1)*320]=cv2.resize(im,(320,240))
        label=f"{r['case']} {r['split']} TP={r['tp']} FP={r['fp']} FN={r['fn']}"
        cv2.putText(sheet,label,(col*320+4,row*264+257),0,.39,(0,0,0),1)
    cv2.imwrite(str(ROOT/'full-frame-grid.jpg'),sheet)
    print(json.dumps({k:v for k,v in report.items() if k!='by_transform'},indent=2))
