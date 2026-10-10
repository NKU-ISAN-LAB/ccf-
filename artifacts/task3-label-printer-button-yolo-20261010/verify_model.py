"""End-to-end evaluation: full original frames, predicted ROIs only."""
import json
from pathlib import Path
import cv2
import numpy as np
from recognize import Detector, annotate, crop_bounds, make_detection, filter_candidates
from metrics import score, summarize

ROOT = Path(__file__).resolve().parent


def variants(image, boxes):
    for name, gain in [('dark_075', .75), ('bright_125', 1.25)]:
        yield name, np.clip(image.astype(float)*gain, 0, 255).astype(np.uint8), boxes
    yield 'blur3', cv2.GaussianBlur(image, (3, 3), .7), boxes
    h, w = image.shape[:2]
    for name, angle, scale in [('rotate_minus8', -8, 1), ('rotate_plus8', 8, 1), ('scale_080', 0, .8)]:
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        warped = cv2.warpAffine(image, matrix, (w, h), borderValue=(40, 40, 40))
        transformed = []
        for box in boxes:
            x1, y1, x2, y2 = box['bbox_xyxy_px']
            corners = np.array([[x1,y1,1], [x2,y1,1], [x2,y2,1], [x1,y2,1]]) @ matrix.T
            xyxy = np.r_[corners.min(0), corners.max(0)]
            xyxy = np.clip(xyxy, [0]*4, [w,h,w,h])
            if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                transformed.append(dict(box, bbox_xyxy_px=xyxy.tolist()))
        yield name, warped, transformed


def main():
    audit = json.loads((ROOT/'annotation-audit.json').read_text())
    detector = Detector()
    originals, stresses, negatives, direct = [], [], [], []
    preview_dir = ROOT/'previews'
    preview_dir.mkdir(exist_ok=True)
    for r in audit['records']:
        image = cv2.imread(r['source'])
        result = detector.predict(image)
        row = dict(filename=r['filename'], split=r['split'], result=result, scores=score(result['detections'], r['boxes']))
        originals.append(row)
        raw = [make_detection(d, image.shape[1], image.shape[0]) for d in detector._raw(image)]
        direct_detections = filter_candidates([d for d in raw if d is not None], detector.thresholds)
        direct.append(dict(filename=r['filename'], split=r['split'], scores=score(direct_detections, r['boxes'])))
        cv2.imwrite(str(preview_dir/r['filename']), annotate(image, result))
        if r['split'] != 'test':
            continue
        for name, modified, boxes in variants(image, r['boxes']):
            prediction = detector.predict(modified)
            stresses.append(dict(filename=r['filename']+':'+name, result=prediction, scores=score(prediction['detections'], boxes)))
        absent = image.copy()
        for b in r['boxes']:
            x1, y1, x2, y2 = np.rint(b['bbox_xyxy_px']).astype(int)
            absent[max(0,y1-5):min(480,y2+5), max(0,x1-5):min(640,x2+5)] = 40
        prediction = detector.predict(absent)
        negatives.append(dict(filename=r['filename']+':synthetic_absent', result=prediction, scores=score(prediction['detections'], [])))
    summary = dict(weights_sha256=detector.implementation['weights_sha256'],
        confidence_thresholds=detector.thresholds, matching='class-aware one-to-one IoU >= 0.5',
        all_originals=summarize(originals),
        splits={s: summarize([r for r in originals if r['split']==s]) for s in ['train','val','test']},
        full_frame_only_diagnostic={s: summarize([r for r in direct if r['split']==s]) for s in ['train','val','test']},
        full_frame_only_note='Same weights and thresholds as crop-refined runtime; diagnostic ablation, not a separately calibrated competitor.',
        test_synthetic_variants=summarize(stresses), test_synthetic_absent=summarize(negatives),
        limitations=['Same physical device and mostly similar views; test results are not field accuracy.',
            'Synthetic variants/erased-object negatives do not establish novel viewpoints or real negative-scene robustness.',
            'Mean pixel error is only against manual box centers on matched boxes, not robot coordinate accuracy.',
            'Button class does not distinguish functions; no automatic button selection or pressing.'])
    (ROOT/'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT/'replay.json').write_text(json.dumps(dict(originals=originals, stress=stresses, synthetic_absent=negatives), indent=2))
    for page in range(3):
        sheet = np.full((1088,1600,3),235,np.uint8)
        for j,r in enumerate(originals[page*20:(page+1)*20]):
            im = cv2.imread(str(preview_dir/r['filename']))
            row,col = divmod(j,5)
            sheet[row*272:row*272+240,col*320:(col+1)*320] = cv2.resize(im,(320,240))
            ok = all(s['fp']==s['fn']==0 for s in r['scores'].values())
            text = r['filename'][:3]+' '+r['split']+(' OK' if ok else ' CHECK')
            cv2.putText(sheet,text,(col*320+4,row*272+258),0,.5,(0,100,0) if ok else (0,0,220),1)
        cv2.imwrite(str(ROOT/f'prediction-sheet-{page+1}.jpg'),sheet)
    # Clearly labelled enlarged review panels; inference still used full images.
    for index in [0,38,44,54]:
        r=audit['records'][index]
        im=cv2.imread(str(preview_dir/r['filename']))
        l,t,rr,bb=crop_bounds(r['boxes'][0]['bbox_xyxy_px'],640,480)
        detail=cv2.resize(im[t:bb,l:rr],(600,600))
        cv2.putText(detail,'Review zoom only; full-frame inference',(10,590),0,.6,(0,0,220),1)
        cv2.imwrite(str(ROOT/f'prediction-detail-{r["filename"][:3]}.jpg'),detail)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
