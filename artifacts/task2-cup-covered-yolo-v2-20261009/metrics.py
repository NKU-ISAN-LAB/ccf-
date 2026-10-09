"""One-to-one multi-object matching; one prediction cannot count for two cups."""
import numpy as np
def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    return inter/(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-inter+1e-9)
def score(detections,boxes):
    ds=sorted(detections,key=lambda d:d['confidence'],reverse=True);matched=set();pairs=[]
    for d in ds:
        overlap,j=max([(iou(d['bbox_xyxy_px'],b),j) for j,b in enumerate(boxes) if j not in matched],default=(0,None))
        if overlap>=.5:
            matched.add(j);pairs.append(dict(gt_index=j,iou=overlap,confidence=d['confidence']))
    tp=len(matched);fp=len(ds)-tp;fn=len(boxes)-tp
    return dict(tp=tp,fp=fp,fn=fn,all_correct=fp==0 and fn==0,matches=pairs)
def summary(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    pairs=[p for r in rows for p in r['matches']]
    return dict(frames=len(rows),tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                all_correct_frames=sum(r['all_correct'] for r in rows),
                mean_matched_iou=float(np.mean([p['iou'] for p in pairs])) if pairs else None,
                failures=[r['case'] for r in rows if not r['all_correct']])
