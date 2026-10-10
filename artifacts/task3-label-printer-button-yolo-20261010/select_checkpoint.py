"""Thresholds selected on training originals, checkpoint on development originals.

The nine held-out test images are never opened here.
"""
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from recognize import Detector, filter_candidates, NAMES
from metrics import score, summarize

ROOT = Path(__file__).resolve().parent


def evaluate(cached, thresholds):
    return [dict(filename=r['filename'], scores=score(filter_candidates(r['candidates'], thresholds), r['boxes'])) for r in cached]


def main():
    records = json.loads((ROOT/'annotation-audit.json').read_text())['records']
    choices = []
    for name in ['best', 'last']:
        path = ROOT/f'runs/printer-button/weights/{name}.pt'
        detector = Detector(weights=path, confidence={'label_printer': .1, 'button': .05})
        cached = {s: [] for s in ['train', 'val']}
        for r in records:
            if r['split'] not in cached:
                continue
            ds = detector.candidates(cv2.imread(r['source']))
            cached[r['split']].append(dict(filename=r['filename'], boxes=r['boxes'], candidates=ds))
        # Sequential per-class training-only selection: printer first, then buttons.
        thresholds = dict(label_printer=.1, button=.05)
        scans = {}
        for label in NAMES.values():
            scans[label] = []
            start = .1 if label == 'label_printer' else .05
            for threshold in np.arange(start, .851, .025):
                value = round(float(threshold), 3)
                summary = summarize(evaluate(cached['train'], dict(thresholds, **{label: value})))
                scans[label].append(dict(threshold=value, metrics=summary['per_class'][label]))
            best_f1 = max(x['metrics']['f1'] for x in scans[label])
            ties = [x for x in scans[label] if abs(x['metrics']['f1']-best_f1) < 1e-10]
            thresholds[label] = ties[len(ties)//2]['threshold']
        train = summarize(evaluate(cached['train'], thresholds))
        val = summarize(evaluate(cached['val'], thresholds))
        choice = dict(checkpoint=name, source=str(path), confidence_thresholds=thresholds, train=train, val=val, scans=scans)
        choices.append(choice)
        (ROOT/f'calibration-{name}-candidates.json').write_text(json.dumps(cached, indent=2))
        print(json.dumps({k: choice[k] for k in ['checkpoint', 'confidence_thresholds', 'train', 'val']}, indent=2), flush=True)
    def rank(c):
        pc = c['val']['per_class']
        return (np.mean([p['f1'] for p in pc.values()]), c['val']['all_objects_correct'],
                np.mean([p['mean_matched_iou'] or 0 for p in pc.values()]))
    winner = max(choices, key=rank)
    (ROOT/'weights').mkdir(exist_ok=True)
    shutil.copy2(winner['source'], ROOT/'weights/best.pt')
    config = dict(algorithm='task3-printer-button-yolo11n-20261010', classes=NAMES,
                  confidence_thresholds=winner['confidence_thresholds'],
                  weights_sha256=hashlib.sha256((ROOT/'weights/best.pt').read_bytes()).hexdigest(),
                  compatible_weights_sha256=[], imgsz=640, rect=False, agnostic_nms=False,
                  predicted_printer_roi_proposal_threshold=.1, roi_max_count=10,
                  crop_policy='square max(160, 1.8 * max predicted printer side), clipped to image',
                  button_function=None, robot_motion_ready=False)
    (ROOT/'runtime-config.json').write_text(json.dumps(config, indent=2))
    (ROOT/'checkpoint-selection.json').write_text(json.dumps(dict(
        selected=winner['checkpoint'], policy='Per-class max training F1, middle threshold among ties; checkpoint chosen by development macro F1, all-correct frames, mean IoU. No test images used.',
        choices=choices), indent=2))


if __name__ == '__main__':
    main()
