"""Validate user YOLO labels; build a temporally blocked development split."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

def main(data, labels):
    out = ROOT / 'dataset'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    images = sorted((data / 'color').glob('*.jpg'))
    assert len(images) == 50, len(images)
    records = []
    with zipfile.ZipFile(labels) as archive:
        assert set(archive.namelist()) == {p.stem + '.txt' for p in images}
        for i, path in enumerate(images):
            raw = archive.read(path.stem + '.txt').decode('utf-8-sig').strip()
            lines = [line.split() for line in raw.splitlines() if line.strip()]
            assert len(lines) == 1 and len(lines[0]) == 5, path.name
            cls, x, y, w, h = map(float, lines[0])
            assert cls == 0 and np.isfinite([x,y,w,h]).all()
            assert w > 0 and h > 0 and 0 <= x-w/2 < x+w/2 <= 1 and 0 <= y-h/2 < y+h/2 <= 1
            image = cv2.imread(str(path))
            assert image is not None and image.shape == (480,640,3)
            box = np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]
            records.append(dict(frame=path.stem, split='train' if i < 40 else 'val',
                                bbox_xyxy=box.tolist(), yolo=[0,x,y,w,h],
                                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                label_sha256=hashlib.sha256(archive.read(path.stem+'.txt')).hexdigest()))
    # Complete validation before creating any dataset artifacts.
    for split in ['train','val']:
        (out/'images'/split).mkdir(parents=True)
        (out/'labels'/split).mkdir(parents=True)
    rng = np.random.default_rng(42)
    for r,path in zip(records,images):
        split,stem = r['split'],r['frame']
        shutil.copy2(path,out/'images'/split/path.name)
        label = ' '.join(map(str,r['yolo']))+'\n'
        (out/'labels'/split/(stem+'.txt')).write_text(label)
        if split == 'train':
            image = cv2.imread(str(path)); x1,y1,x2,y2 = np.rint(r['bbox_xyxy']).astype(int)
            # Explicitly remove machine context from a second positive view.
            # Keep the labeled cup and a small local margin, not a machine ROI.
            isolated = np.full_like(image,int(rng.integers(35,220)))
            xa,ya,xb,yb=max(0,x1-12),max(0,y1-12),min(640,x2+12),min(480,y2+12)
            isolated[ya:yb,xa:xb]=image[ya:yb,xa:xb]
            cv2.imwrite(str(out/'images'/split/(stem+'_cup_only.jpg')),isolated)
            (out/'labels'/split/(stem+'_cup_only.txt')).write_text(label)
            if int(stem)%2 == 0:
                removed=image.copy()
                removed[max(0,y1-5):min(480,y2+5),max(0,x1-5):min(640,x2+5)]=40
                cv2.imwrite(str(out/'images'/split/(stem+'_synthetic_absent.jpg')),removed)
                (out/'labels'/split/(stem+'_synthetic_absent.txt')).write_text('')
    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: exposed_cup\n')
    report=dict(label_zip_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
                source_dataset=data.name,classes={'0':'exposed_cup'},records=records,
                split_policy='0001-0040 train, 0041-0050 contiguous development validation. First 30 near-duplicates stay in training. Same session, possible similar views; NOT independent field accuracy.',
                training_images=100,validation_images=10,
                augmentation='40 original train + 40 cup-only context-masked positives + 20 synthetic absent-cup negatives; all derived from training frames only. No real absent-cup images.')
    (ROOT/'annotation-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for page in range(2):
        sheet=np.full((1320,1600,3),235,np.uint8)
        for j,r in enumerate(records[page*25:(page+1)*25]):
            im=cv2.imread(str(data/'color'/(r['frame']+'.jpg')))
            x1,y1,x2,y2=np.rint(r['bbox_xyxy']).astype(int)
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),1)
            row,col=divmod(j,5)
            sheet[row*264:row*264+240,col*320:(col+1)*320]=cv2.resize(im,(320,240))
            cv2.putText(sheet,r['frame']+' '+r['split'],(col*320+5,row*264+257),0,.5,(0,0,0),1)
        cv2.imwrite(str(ROOT/f'manual-labels-{page+1}.jpg'),sheet)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    a=p.parse_args();main(a.data,a.labels)
