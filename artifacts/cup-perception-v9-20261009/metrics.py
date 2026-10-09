import numpy as np

def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    intersection=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    return intersection/(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-intersection+1e-9)

def score(detections,box):
    overlaps=[iou(d['bbox_xyxy_px'],box) for d in detections] if box is not None else []
    best=max(overlaps,default=0.)
    tp=int(best>=.5)
    return dict(tp=tp,fp=len(detections)-tp,fn=int(box is not None)-tp,
                best_iou=best,unique_correct=bool(tp and len(detections)==1),
                confidence=max([d['confidence'] for d,o in zip(detections,overlaps) if o>=.5],default=0.))

def summary(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    return dict(frames=len(rows),tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                unique_correct=sum(r['unique_correct'] for r in rows),
                mean_best_iou=float(np.mean([r['best_iou'] for r in rows])) if rows else None,
                median_matched_confidence=float(np.median([r['confidence'] for r in rows if r['tp']])) if tp else None,
                failures=[r['case'] for r in rows if r['fn'] or r['fp']])
