"""Separate train replay, grouped development validation, and synthetic checks."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from recognize import Detector,ROOT,annotate,NAMES,box_iou
from metrics import score,summarize

def main(source):
    detector=Detector();audit=json.loads((ROOT/'annotation-audit.json').read_text())
    rows=[];variants=[];negative=[];unknown=[]
    previews=ROOT/'previews';previews.mkdir(exist_ok=True)
    for r in audit['records']:
        image=cv2.imread(str(source/r['filename']));prediction=detector.predict(image)
        cup_boxes=[b['bbox_xyxy_px'] for b in r['boxes'] if b['label']=='cup']
        target_boxes=[b['bbox_xyxy_px'] for b in r['boxes'] if b['label']=='target']
        overlaps=any(box_iou(c,t)>0 for c in cup_boxes for t in target_boxes)
        rows.append(dict(filename=r['filename'],split=r['split'],source_group=r['source_group'],
            labeled_cup_target_overlap=overlaps,
            scores=score(prediction['detections'],r['boxes']),result=prediction))
        cv2.imwrite(str(previews/r['filename']),annotate(image,prediction))
        absent=image.copy();isolated=np.full_like(image,127)
        for box in r['boxes']:
            x1,y1,x2,y2=np.rint(box['bbox_xyxy_px']).astype(int)
            absent[max(0,y1-7):min(480,y2+7),max(0,x1-7):min(640,x2+7)]=40
            xa,ya,xb,yb=max(0,x1-14),max(0,y1-14),min(640,x2+14),min(480,y2+14)
            isolated[ya:yb,xa:xb]=image[ya:yb,xa:xb]
        nr=detector.predict(absent)
        negative.append(dict(filename=r['filename'],kind='synthetic_both_objects_removed',passed=not nr['detections'],detections=nr['detections']))
        if r['split']=='val':
            _,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,65])
            images={'context_removed':isolated,
                'dim':np.clip(image.astype(float)*.8+8,0,255).astype(np.uint8),
                'bright':np.clip(image.astype(float)*1.15+5,0,255).astype(np.uint8),
                'blur':cv2.GaussianBlur(image,(3,3),.6),'jpeg65':cv2.imdecode(encoded,cv2.IMREAD_COLOR),
                'shift':cv2.warpAffine(image,np.float64([[1,0,10],[0,1,-6]]),(640,480))}
            for kind,im in images.items():
                gt=[{**b,'bbox_xyxy_px':(np.array(b['bbox_xyxy_px'])+([10,-6,10,-6] if kind=='shift' else [0,0,0,0])).tolist()} for b in r['boxes']]
                result=detector.predict(im)
                variants.append(dict(filename=r['filename'],kind=kind,scores=score(result['detections'],gt),detections=result['detections']))
                if kind=='context_removed':cv2.imwrite(str(previews/(Path(r['filename']).stem+'_no_context.jpg')),annotate(im,result))
    for kind,im in [('black',np.zeros((480,640,3),np.uint8)),('white',np.full((480,640,3),255,np.uint8)),
                    ('noise',np.random.default_rng(7).integers(0,256,(480,640,3),dtype=np.uint8))]:
        result=detector.predict(im);negative.append(dict(kind=kind,passed=not result['detections'],detections=result['detections']))
    for filename in audit['excluded_missing_labels']:
        result=detector.predict(cv2.imread(str(source/filename)))
        unknown.append(dict(filename=filename,result=result,qualification='No user labels; not scored and not assumed negative'))
    report=dict(implementation=detector.implementation,confidence_thresholds=detector.thresholds,iou_match_threshold=.5,
        original_replay=summarize(rows),train_replay=summarize([r for r in rows if r['split']=='train']),
        development_validation=summarize([r for r in rows if r['split']=='val']),
        validation_by_source={s:summarize([r for r in rows if r['split']=='val' and r['source_group']==s]) for s in ['old_step02','new_20261008']},
        validation_by_overlap={str(state):summarize([r for r in rows if r['split']=='val' and r['labeled_cup_target_overlap']==state]) for state in [False,True]},
        synthetic_validation_variants={k:summarize([r for r in variants if r['kind']==k]) for k in images},
        negative_checks=len(negative),negative_passed=sum(r['passed'] for r in negative),
        negative_failures=[r for r in negative if not r['passed']],
        median_inference_ms=float(np.median([r['result']['inference_ms'] for r in rows])),
        qualification='Development evaluation, not independent field accuracy. Thresholds fit training only; validation selects checkpoint. Missing-label images excluded from metrics. Synthetic negatives are not true empty-scene validation. 2D boxes and overlap are not grasp or placement proof.')
    for name,value in [('verification.json',report),('replay.json',rows),('stress-report.json',variants),('negative-checks.json',negative),('unlabeled-diagnostics.json',unknown)]:
        (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    for page in range(3):
        sheet=np.full((4*272,5*320,3),235,np.uint8)
        for j,r in enumerate(rows[page*20:(page+1)*20]):
            image=cv2.imread(str(previews/r['filename']));row,col=divmod(j,5)
            sheet[row*272:row*272+240,col*320:(col+1)*320]=cv2.resize(image,(320,240))
            title=r['filename'].split('_')[0]+' '+r['split']+' '+('OK' if all(s['tp']==1 and s['fp']==0 and s['fn']==0 for s in r['scores'].values()) else 'CHECK')
            cv2.putText(sheet,title,(col*320+4,row*272+259),0,.48,(0,0,0),1)
        cv2.imwrite(str(ROOT/f'full-frame-grid-{page+1}.jpg'),sheet)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);a=p.parse_args();main(a.images)
