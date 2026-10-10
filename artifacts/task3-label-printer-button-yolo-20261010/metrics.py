"""Class-aware, one-to-one box matching (IoU >= .5); both buttons counted."""
import numpy as np
from recognize import NAMES, box_iou


def score(detections, ground_truth):
    result = {}
    for label in NAMES.values():
        ds = sorted([d for d in detections if d['label'] == label], key=lambda d: d['confidence'], reverse=True)
        gt = [g for g in ground_truth if g['label'] == label]
        matched, overlaps, errors = set(), [], []
        for d in ds:
            choices = [(box_iou(d['bbox_xyxy_px'], g['bbox_xyxy_px']), j) for j, g in enumerate(gt) if j not in matched]
            best, j = max(choices, default=(0, None))
            if best >= .5:
                matched.add(j)
                overlaps.append(best)
                a, b = np.asarray(d['bbox_xyxy_px']), np.asarray(gt[j]['bbox_xyxy_px'])
                errors.append(float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)))
        result[label] = dict(tp=len(matched), fp=len(ds)-len(matched), fn=len(gt)-len(matched),
                             matched_ious=overlaps, center_errors_px=errors)
    return result


def summarize(rows):
    result = {}
    for label in NAMES.values():
        tp, fp, fn = [sum(r['scores'][label][k] for r in rows) for k in ['tp', 'fp', 'fn']]
        ious = [v for r in rows for v in r['scores'][label]['matched_ious']]
        errors = [v for r in rows for v in r['scores'][label]['center_errors_px']]
        result[label] = dict(tp=tp, fp=fp, fn=fn, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
            f1=2*tp/max(1, 2*tp+fp+fn), mean_matched_iou=float(np.mean(ious)) if ious else None,
            mean_matched_center_error_px=float(np.mean(errors)) if errors else None,
            failures=[r['filename'] for r in rows if r['scores'][label]['fp'] or r['scores'][label]['fn']])
    return dict(frames=len(rows), per_class=result, all_objects_correct=sum(
        all(r['scores'][name]['fp'] == r['scores'][name]['fn'] == 0 for name in NAMES.values()) for r in rows))
