"""Predefined evaluation. Test results MUST NOT select weights/thresholds."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from recognize import Detector,ROOT,annotate
from metrics import score,summary

def warp(image,box,matrix):
    if matrix.shape==(2,3):matrix=np.vstack([matrix,[0,0,1]])
    x1,y1,x2,y2=box
    corners=np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]])
    points=cv2.perspectiveTransform(corners,matrix.astype(float))[0]
    new_box=np.r_[points.min(axis=0),points.max(axis=0)]
    assert (new_box[:2]>=0).all() and (new_box[2:]<=[640,480]).all(),new_box
    return cv2.warpPerspective(image,matrix,(640,480),borderValue=(114,114,114)),new_box.tolist()

def variants(image,box):
    for angle in [-20,20]:
        yield f'rotate_{angle}',*warp(image,box,cv2.getRotationMatrix2D((320,240),angle,1))
    for scale in [.65,.8,1.2]:
        yield f'scale_{scale}',*warp(image,box,cv2.getRotationMatrix2D((320,240),0,scale))
    src=np.float32([[0,0],[639,0],[639,479],[0,479]])
    for name,dst in [('perspective_left',[[40,20],[580,55],[625,440],[10,465]]),
                     ('perspective_right',[[60,50],[625,10],[600,465],[10,430]])]:
        yield name,*warp(image,box,cv2.getPerspectiveTransform(src,np.float32(dst)))
    yield 'dim',np.clip(image.astype(float)*.65+8,0,255).astype(np.uint8),box
    yield 'bright',np.clip(image.astype(float)*1.2+8,0,255).astype(np.uint8),box
    yield 'blur',cv2.GaussianBlur(image,(5,5),1.),box
    _,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,50])
    yield 'jpeg50',cv2.imdecode(encoded,cv2.IMREAD_COLOR),box
    x1,y1,x2,y2=np.rint(box).astype(int)
    isolated=np.full_like(image,127)
    xa,ya,xb,yb=max(0,x1-14),max(0,y1-14),min(640,x2+14),min(480,y2+14)
    isolated[ya:yb,xa:xb]=image[ya:yb,xa:xb]
    yield 'context_removed',isolated,box

def main():
    audit=json.loads((ROOT/'annotation-audit.json').read_text())
    cases=[]
    for r in audit['records']:
        path=Path(r['image'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256'],path
        im=cv2.imread(str(path));box=r['bbox_xyxy']
        cases.append(dict(case=r['id'],split=r['split'],source=r['source'],kind='original',image=im,box=box))
        old_regression=r['source']=='20261008' and int(r['frame']) in {33,38,40,49,50}
        if r['split']=='test' or old_regression:
            for kind,image,newbox in variants(im,box):
                cases.append(dict(case=r['id']+'_'+kind,split='test' if r['split']=='test' else 'old_known_regression',source=r['source'],kind=kind,image=image,box=newbox))
        x1,y1,x2,y2=np.rint(box).astype(int)
        absent=im.copy();absent[max(0,y1-5):min(480,y2+5),max(0,x1-5):min(640,x2+5)]=40
        cases.append(dict(case=r['id']+'_synthetic_absent',split=r['split'],source=r['source'],kind='negative',image=absent,box=None))
    for name,im in [('black',np.zeros((480,640,3),np.uint8)),('white',np.full((480,640,3),255,np.uint8)),
                    ('noise',np.random.default_rng(109).integers(0,256,(480,640,3),dtype=np.uint8))]:
        cases.append(dict(case=name,split='synthetic',source='synthetic',kind='negative',image=im,box=None))
    old=Path('/home/vnv/cup-perception-v7-20261008')
    models=[('v7_current',old/'weights/best.pt',.05),
            ('v7_last',old/'runs/direct-cup/weights/last.pt',.5),
            ('v8_candidate',Path('/home/vnv/cup-perception-v8-20261009/weights/best.pt'),.225),
            ('v9_candidate',ROOT/'weights/best.pt',None)]
    reports={};results={}
    previews=ROOT/'comparison-previews';previews.mkdir(exist_ok=True)
    for name,weights,threshold in models:
        detector=Detector(weights=weights,confidence=threshold)
        rows=[]
        for c in cases:
            result=detector.predict(c['image'])
            row={k:v for k,v in c.items() if k!='image'}
            row.update(score(result['detections'],c['box']),detections=result['detections'])
            rows.append(row)
            if c['source']=='20261009' and c['kind']=='original':
                preview=annotate(c['image'],result)
                x1,y1,x2,y2=np.rint(c['box']).astype(int)
                cv2.rectangle(preview,(x1,y1),(x2,y2),(255,100,0),1)
                cv2.imwrite(str(previews/(name+'_'+c['case']+'.jpg')),preview)
        results[name]=rows
        positive=[r for r in rows if r['kind']=='original']
        negative=[r for r in rows if r['kind']=='negative']
        reports[name]=dict(weights_sha256=detector.implementation['weights_sha256'],confidence_threshold=detector.confidence,
            all_originals=summary(positive),
            old_50_regression=summary([r for r in positive if r['source']=='20261008']),
            new_19_replay=summary([r for r in positive if r['source']=='20261009']),
            train=summary([r for r in positive if r['split']=='train']),
            checkpoint_validation=summary([r for r in positive if r['split']=='val']),
            reserved_new_test=summary([r for r in positive if r['split']=='test']),
            transformed_reserved_new_test=summary([r for r in rows if r['split']=='test' and r['kind'] not in ['original','negative']]),
            transformed_old_known_regression=summary([r for r in rows if r['split']=='old_known_regression']),
            by_transform={kind:summary([r for r in rows if r['kind']==kind]) for kind in sorted({r['kind'] for r in rows}-{'original','negative'})},
            synthetic_negatives=dict(frames=len(negative),false_positive_frames=sum(bool(r['fp']) for r in negative),failures=[r['case'] for r in negative if r['fp']]))
        print(name,json.dumps({k:v for k,v in reports[name].items() if k!='by_transform'}),flush=True)
        # Uniform full-frame visualization, including misses and multiple candidates.
        new_rows=[r for r in positive if r['source']=='20261009']
        sheet=np.full((1056,1600,3),235,np.uint8)
        for i,r in enumerate(new_rows):
            preview=cv2.imread(str(previews/(name+'_'+r['case']+'.jpg')))
            row,col=divmod(i,5)
            sheet[row*264:row*264+240,col*320:(col+1)*320]=cv2.resize(preview,(320,240))
            label=f"{r['case']} {r['split']} {'OK' if r['unique_correct'] else 'FAIL'}"
            cv2.putText(sheet,label,(col*320+5,row*264+257),0,.4,(0,0,0),1)
        cv2.imwrite(str(ROOT/(name+'-new-data-grid.jpg')),sheet)
    report=dict(models=reports,
        qualification='New 0016-0019 are excluded from v9 training, initialization, checkpoint selection and threshold calibration. Same-session similar-view temporal holdout, not independent field accuracy. v9 trains on former v8 old test frames, so all old-view results are development regression only. Projective transformations cannot create unseen physical occlusions/sides. Synthetic negatives are not real empty-scene validation.',
        deployment='Candidate only. Existing v7, v8 and task2 files and remote machines remain unchanged.')
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2))
    (ROOT/'comparison-replay.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
