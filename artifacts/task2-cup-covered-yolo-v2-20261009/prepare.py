"""Keep cup_covered only; add reviewed open-cup/holder NEGATIVE examples."""
import hashlib,json,shutil
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
BASE=Path('/home/vnv/task2-cup-covered-yolo-20261009')
if __name__=='__main__':
    out=ROOT/'dataset'
    if out.exists():raise FileExistsError(out)
    shutil.copytree(BASE/'dataset',out)
    for cache in out.glob('labels/*.cache'):cache.unlink() # Only copied generated caches, not source data.
    # Threshold calibration reads train; holdout negative probes are stored separately.
    (out/'images/negative_test').mkdir();(out/'labels/negative_test').mkdir()
    old=json.loads(Path('/home/vnv/task2-cup-target-yolo-20261008/annotation-audit.json').read_text())
    records=[]
    for prefix,split in [('001_','train'),('005_','train'),('003_','val'),('007_','negative_test')]:
        r=next(r for r in old['records'] if r['filename'].startswith(prefix))
        path=Path(old['source_image_directory'])/r['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['source_image_sha256']
        image=cv2.imread(str(path));cases=[('full',image)]
        for b in r['boxes']:
            x1,y1,x2,y2=np.rint(b['bbox_xyxy_px']).astype(int)
            crop=image[max(0,y1-16):min(480,y2+16),max(0,x1-16):min(640,x2+16)]
            for scale in [1.,2.]:
                patch=cv2.resize(crop,None,fx=scale,fy=scale);h,w=patch.shape[:2]
                canvas=np.full_like(image,127);left=(640-w)//2;top=(480-h)//2
                canvas[top:top+h,left:left+w]=patch;cases.append((b['label']+f'_scale{scale}',canvas))
        for kind,im in cases:
            stem='negative_'+prefix+kind
            assert cv2.imwrite(str(out/'images'/split/(stem+'.jpg')),im)
            (out/'labels'/split/(stem+'.txt')).write_text('')
            records.append(dict(case=stem,split=split,image=str(out/'images'/split/(stem+'.jpg')),
                                source=str(path),source_sha256=r['source_image_sha256'],episode=r['episode'],
                                kind=kind,boxes_xyxy=[],qualification='Visually checked open cup and empty holder, no cup_covered positive labels'))
    (ROOT/'negative-source-audit.json').write_text(json.dumps(dict(records=records,
        split_policy='Negative source episodes 000000/000020 train, 000010 development val, 000031 reserved negative test. Prior observed failure is now explicitly training/regression, not independent test.'),indent=2))
    audit=json.loads((BASE/'annotation-audit.json').read_text())
    audit.update(training_images=52,development_validation_images=8,
        refinement='Init from initial covered-cup candidate; add 10 real-source negative images/crops to train and 5 to val. Existing 5 positive test frames reused for regression; new 5 negative probes withheld.')
    (ROOT/'annotation-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cup_covered\n')
    print('Prepared 52 train, 8 validation, 5 positive regression-test and 5 withheld negative probes')
