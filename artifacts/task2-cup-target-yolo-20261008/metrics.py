"""Class-aware one-to-one IoU matching for annotated detection checks."""
import numpy as np
from recognize import NAMES,box_iou

def score(detections,ground_truth):
    result={}
    for label in NAMES.values():
        ds=sorted([d for d in detections if d['label']==label],key=lambda d:d['confidence'],reverse=True)
        gt=[g for g in ground_truth if g['label']==label];matched=set();tp=0;overlaps=[]
        for d in ds:
            choices=[(box_iou(d['bbox_xyxy_px'],g['bbox_xyxy_px']),j) for j,g in enumerate(gt) if j not in matched]
            best,j=max(choices,default=(0,None))
            if best>=.5:matched.add(j);tp+=1;overlaps.append(best)
        result[label]=dict(tp=tp,fp=len(ds)-tp,fn=len(gt)-tp,matched_ious=overlaps)
    return result

def summarize(rows):
    result={}
    for label in NAMES.values():
        tp=sum(r['scores'][label]['tp'] for r in rows);fp=sum(r['scores'][label]['fp'] for r in rows);fn=sum(r['scores'][label]['fn'] for r in rows)
        overlaps=[v for r in rows for v in r['scores'][label]['matched_ious']]
        result[label]=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
            mean_matched_iou=float(np.mean(overlaps)) if overlaps else None,
            failures=[r['filename'] for r in rows if r['scores'][label]['fp'] or r['scores'][label]['fn']])
    return dict(frames=len(rows),per_class=result,
        both_classes_correct=sum(all(r['scores'][label]['tp']>0 and r['scores'][label]['fp']==0 and r['scores'][label]['fn']==0 for label in NAMES.values()) for r in rows))
