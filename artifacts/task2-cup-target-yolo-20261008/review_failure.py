"""Document a failed development case without changing annotations or scores."""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
from recognize import ROOT,box_iou,annotate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);a=p.parse_args()
    name='018_old_s02_episode_000083_f00338.jpg'
    audit=json.loads((ROOT/'annotation-audit.json').read_text())
    gt=next(r for r in audit['records'] if r['filename']==name)
    replay=json.loads((ROOT/'replay.json').read_text())
    result=next(r['result'] for r in replay if r['filename']==name)
    image=cv2.imread(str(a.images/name));left=image.copy()
    for b in gt['boxes']:
        x1,y1,x2,y2=np.rint(b['bbox_xyxy_px']).astype(int)
        color=(0,150,255) if b['label']=='cup' else (255,220,0)
        cv2.rectangle(left,(x1,y1),(x2,y2),color,2)
        cv2.putText(left,b['label'],(x1,max(12,y1-4)),0,.4,color,1)
    right=annotate(image,result)
    sheet=np.full((515,1280,3),235,np.uint8)
    sheet[35:,:640]=left;sheet[35:,640:]=right
    cv2.putText(sheet,'USER ANNOTATION - full original frame',(10,25),0,.6,(0,0,0),1)
    cv2.putText(sheet,'MODEL PREDICTION - same full original frame',(650,25),0,.6,(0,0,0),1)
    cv2.imwrite(str(ROOT/'failure-018-comparison.jpg'),sheet)
    target_gt=next(b['bbox_xyxy_px'] for b in gt['boxes'] if b['label']=='target')
    target_pred=result['target_bbox_xyxy_px']
    report=dict(filename=name,ground_truth_target=target_gt,predicted_target=target_pred,
        target_iou=box_iou(target_gt,target_pred),required_iou=.5,
        conclusion='Target box is smaller and lower than the user annotation; counted as failed localization. Cup is correct. No annotation changed and no sample moved into training.')
    (ROOT/'failure-analysis.json').write_text(json.dumps(report,indent=2))
    rows=list(csv.DictReader((ROOT/'training-results.csv').open()))
    best=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    summary=dict(completed_epochs=len(rows),best_logged_epoch=int(best['epoch']),
        best_logged_validation_map50_95=float(best['metrics/mAP50-95(B)']),
        total_training_seconds=float(rows[-1]['time']),device='cpu')
    (ROOT/'training-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(failure=report,training=summary),indent=2))
