"""Development replay and synthetic-negative tests, not independent accuracy."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from recognize import ROOT,Detector,annotate

def main(data):
    detector=Detector();serial=detector.manifest['camera_serial']
    references={r['frame'] for r in detector.manifest['references']}
    cases=[];rows=[];integrity=[]
    (ROOT/'previews').mkdir(exist_ok=True)
    for path in sorted((data/'color').glob('*.jpg')):
        meta=json.loads((data/'metadata'/f'{path.stem}.json').read_text())
        image=cv2.imread(str(path));depth=cv2.imread(str(data/'depth'/f'{path.stem}.png'),-1)
        integrity.append(bool(image.shape==(480,640,3) and depth.shape==(480,640)
                         and depth.dtype==np.uint16 and meta['camera_serial']==serial
                         and meta['calibration']['serial']==serial
                         and meta['color_timestamp_ms']==meta['depth_timestamp_ms']))
        result=detector.predict(image,meta['camera_serial'])
        rows.append({'frame':path.stem,'used_as_reference':path.stem in references,**result})
        cv2.imwrite(str(ROOT/'previews'/path.name),annotate(image,result))
        if not result['candidate_valid_2d']:continue
        # These boxes were visually reviewed in review-crops-0/1.jpg. They are
        # detection-guided synthetic occlusions, NOT real empty-dispenser photos.
        x1,y1,x2,y2=np.rint(result['cup_bbox_xyxy_px']).astype(int)
        removed=image.copy();removed[max(0,y1-3):y2+4,max(0,x1-3):x2+4]=35
        absent=detector.predict(removed,serial)
        cases.append({'frame':path.stem,'test':'synthetic_cup_removed_table_cups_retained',
                      'passed':not absent['candidate_valid_2d'],'result':absent})
    original=cv2.imread(str(data/'color/0001.jpg'))
    for name,image,camera in [('wrong_camera',original,'CP2L863000KV'),
                              ('no_camera_identity',original,None),
                              ('wrong_resolution',original[::2,::2],serial),
                              ('black',np.zeros_like(original),serial),
                              ('white',np.full_like(original,255),serial),
                              ('noise',np.random.default_rng(9).integers(0,256,original.shape,dtype=np.uint8),serial),
                              ('upside_down',cv2.rotate(original,cv2.ROTATE_180),serial)]:
        result=detector.predict(image,camera)
        cases.append({'test':name,'passed':not result['candidate_valid_2d'] and result['visible_lower_rim_uv'] is None})
    for path in (ROOT/'test_assets').glob('negative-camera-*.jpg'):
        result=detector.predict(cv2.imread(str(path)),serial)
        cases.append({'test':'other_view_'+path.name,'passed':not result['candidate_valid_2d']})
    nonref=[r for r in rows if not r['used_as_reference']]
    report={'dataset':data.name,'frames':len(rows),'integrity_pass':sum(integrity),
            'references':sorted(references),'detected':sum(r['candidate_valid_2d'] for r in rows),
            'fixed_view_first_30_detected':sum(r['candidate_valid_2d'] for r in rows[:30]),
            'nonreference_frames':len(nonref),'nonreference_detected':sum(r['candidate_valid_2d'] for r in nonref),
            'rejected_frames':[r['frame'] for r in rows if not r['candidate_valid_2d']],
            'negative_checks':len(cases),'negative_passed':sum(c['passed'] for c in cases),
            'negative_failures':[c for c in cases if not c['passed']],
            'median_inference_ms':float(np.median([r.get('inference_ms',0) for r in rows])),
            'qualification':'Same-session development replay, not independent accuracy. References selected during development. No real absent-cup images. No 3D grasp output.'}
    (ROOT/'replay.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'negative-checks.json').write_text(json.dumps(cases,indent=2))
    (ROOT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if all(integrity) and all(c['passed'] for c in cases) else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().data))
