"""Replay against USER labels, with separate held-out and synthetic results."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from recognize import Detector,ROOT,annotate

def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    intersect=float(np.prod(size))
    return intersect/(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-intersect+1e-9)

def summary(rows):
    tp=sum(r['true_positives'] for r in rows)
    fp=sum(r['false_positives'] for r in rows)
    fn=sum(r['false_negatives'] for r in rows)
    return dict(frames=len(rows),true_positives=tp,false_positives=fp,false_negatives=fn,
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                unique_correct_candidates=sum(r['unique_correct_candidate'] for r in rows),
                failures=[r['frame'] for r in rows if not r['unique_correct_candidate']],
                mean_best_iou=float(np.mean([r['best_iou'] for r in rows])))

def score(result,box):
    overlaps=[iou(d['bbox_xyxy_px'],box) for d in result['detections']]
    best=max(overlaps,default=0.)
    tp=int(best>=.5)
    return dict(best_iou=best,true_positives=tp,false_positives=len(overlaps)-tp,
                false_negatives=1-tp,unique_correct_candidate=bool(tp and result['candidate_valid_2d']))

def main(data,weights):
    detector=Detector(weights=weights)
    rows=[];tests=[];negative=[]
    audit=json.loads((ROOT/'annotation-audit.json').read_text())
    previews=ROOT/'previews';previews.mkdir(exist_ok=True)
    for r in audit['records']:
        image=cv2.imread(str(data/'color'/(r['frame']+'.jpg')))
        result=detector.predict(image,'CP2L863000RG')
        rows.append(dict(frame=r['frame'],split=r['split'],**score(result,r['bbox_xyxy']),result=result))
        preview=annotate(image,result)
        # Manual GT blue; predictions green. Always show complete original frame.
        x1,y1,x2,y2=np.rint(r['bbox_xyxy']).astype(int)
        cv2.rectangle(preview,(x1,y1),(x2,y2),(255,100,0),1)
        cv2.imwrite(str(previews/(r['frame']+'.jpg')),preview)
        absent=image.copy();absent[max(0,y1-5):min(480,y2+5),max(0,x1-5):min(640,x2+5)]=40
        ar=detector.predict(absent)
        negative.append(dict(frame=r['frame'],split=r['split'],kind='synthetic_cup_removed',
                             passed=len(ar['detections'])==0,detections=ar['detections']))
        if r['split']=='val':
            # Context removed; the actual cup pixels are unchanged. This checks
            # direct recognition without seeing any identifiable whole machine.
            isolated=np.full_like(image,127)
            xa,ya,xb,yb=max(0,x1-12),max(0,y1-12),min(640,x2+12),min(480,y2+12)
            isolated[ya:yb,xa:xb]=image[ya:yb,xa:xb]
            _,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,65])
            variants={'machine_context_removed':isolated,
                'dim':np.clip(image.astype(float)*.8+8,0,255).astype(np.uint8),
                'bright':np.clip(image.astype(float)*1.15+5,0,255).astype(np.uint8),
                'blur':cv2.GaussianBlur(image,(3,3),.6),
                'jpeg65':cv2.imdecode(encoded,cv2.IMREAD_COLOR),
                'shift':cv2.warpAffine(image,np.float64([[1,0,12],[0,1,-8]]),(640,480))}
            for kind,im in variants.items():
                box=np.array(r['bbox_xyxy'])+([12,-8,12,-8] if kind=='shift' else [0,0,0,0])
                predicted=detector.predict(im)
                tests.append(dict(frame=r['frame'],kind=kind,**score(predicted,box),detections=predicted['detections']))
                if kind=='machine_context_removed':cv2.imwrite(str(previews/(r['frame']+'_no_machine.jpg')),annotate(im,predicted))
    for name,image in [('black',np.zeros((480,640,3),np.uint8)),
                       ('white',np.full((480,640,3),255,np.uint8)),
                       ('noise',np.random.default_rng(12).integers(0,256,(480,640,3),dtype=np.uint8))]:
        p=detector.predict(image)
        negative.append(dict(kind=name,passed=not p['detections'],detections=p['detections']))
    report=dict(implementation=detector.implementation,confidence_threshold=detector.confidence,iou_threshold=.5,
                original_replay=summary(rows),train_replay=summary([r for r in rows if r['split']=='train']),
                development_validation=summary([r for r in rows if r['split']=='val']),
                synthetic_validation_variants={k:summary([r for r in tests if r['kind']==k]) for k in variants},
                negative_checks=len(negative),negative_passed=sum(r['passed'] for r in negative),
                negative_failures=[r for r in negative if not r['passed']],
                median_inference_ms=float(np.median([r['result']['inference_ms'] for r in rows])),
                qualification='Same-session development validation, NOT independent field accuracy. Last 10 frames excluded from training, but used for checkpoint selection. Synthetic negatives are not real absent-cup validation. Box centers are not measured rims or grasp poses.')
    (ROOT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    (ROOT/'replay.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (ROOT/'stress-report.json').write_text(json.dumps(tests,indent=2,allow_nan=False))
    (ROOT/'negative-checks.json').write_text(json.dumps(negative,indent=2,allow_nan=False))
    for page in range(2):
        sheet=np.full((1320,1600,3),235,np.uint8)
        for j,r in enumerate(rows[page*25:(page+1)*25]):
            im=cv2.imread(str(previews/(r['frame']+'.jpg')));row,col=divmod(j,5)
            sheet[row*264:row*264+240,col*320:(col+1)*320]=cv2.resize(im,(320,240))
            label=f"{r['frame']} {r['split']} IoU={r['best_iou']:.2f}"
            cv2.putText(sheet,label,(col*320+5,row*264+257),0,.45,(0,0,0),1)
        cv2.imwrite(str(ROOT/f'full-frame-grid-{page+1}.jpg'),sheet)
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path)
    a=p.parse_args();main(a.data,a.weights)
