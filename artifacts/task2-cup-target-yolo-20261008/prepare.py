"""Audit user two-class annotations and build grouped development data."""
import argparse,hashlib,json,shutil,zipfile
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
NAMES={0:'cup',1:'target'}
VAL_EPISODES={'episode_000010','episode_000083','episode_000136','episode_000188'}
# Oblique/new-camera view group, held out together. Still one recording session.
VAL_NEW={'0033.jpg','0038.jpg','0040.jpg','0049.jpg','0050.jpg'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(source,archive):
    target=ROOT/'dataset'
    if target.exists():raise FileExistsError(target)
    manifest=json.loads((source/'manifest.json').read_text());records=[];missing=[];removed=[]
    with zipfile.ZipFile(archive) as z:
        expected={Path(r['filename']).stem+'.txt' for r in manifest['records']}
        assert set(z.namelist())<=expected,sorted(set(z.namelist())-expected)
        for r in manifest['records']:
            path=source/'images'/r['filename'];assert sha(path)==r['image_sha256']
            member=path.stem+'.txt'
            if member not in z.namelist():missing.append(r['filename']);continue
            raw=z.read(member).decode('utf-8-sig');boxes=[];lines=[]
            for number,line in enumerate(raw.splitlines(),1):
                fields=line.split()
                if not fields:continue
                assert len(fields)==5,(member,number,line)
                cls,x,y,w,h=map(float,fields)
                assert cls in NAMES and np.isfinite([cls,x,y,w,h]).all(),(member,number)
                if w<=0 or h<=0:
                    removed.append(dict(file=member,line=number,original=line,reason='non_positive_width_or_height'))
                    continue
                assert 0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1,(member,number)
                box=(np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]).tolist()
                boxes.append(dict(class_id=int(cls),label=NAMES[int(cls)],bbox_xyxy_px=box))
                lines.append(line)
            assert boxes,(member,'No valid boxes; do not infer negative annotation')
            split='val' if r.get('episode') in VAL_EPISODES or r.get('original_name') in VAL_NEW else 'train'
            records.append(dict(filename=path.name,source_group=r['source_group'],
                episode=r.get('episode'),original_name=r.get('original_name'),
                visual_stage_note=r['visual_stage_note'],split=split,boxes=boxes,
                source_image_sha256=sha(path),source_label_sha256=hashlib.sha256(z.read(member)).hexdigest(),
                clean_label='\n'.join(lines)+'\n'))
    assert len(records)==56 and len(missing)==4
    for split in ['train','val']:
        (target/'images'/split).mkdir(parents=True);(target/'labels'/split).mkdir(parents=True)
    rng=np.random.default_rng(7);train_index=0;derived=[]
    for r in records:
        path=source/'images'/r['filename'];split=r['split'];stem=path.stem
        image=cv2.imread(str(path));assert image.shape==(480,640,3)
        shutil.copy2(path,target/'images'/split/path.name)
        (target/'labels'/split/(stem+'.txt')).write_text(r['clean_label'])
        if split=='train':
            isolated=np.full_like(image,int(rng.integers(35,215)))
            for box in r['boxes']:
                x1,y1,x2,y2=np.rint(box['bbox_xyxy_px']).astype(int)
                x1,y1,x2,y2=max(0,x1-14),max(0,y1-14),min(640,x2+14),min(480,y2+14)
                isolated[y1:y2,x1:x2]=image[y1:y2,x1:x2]
            cv2.imwrite(str(target/'images/train'/(stem+'_objects_only.jpg')),isolated)
            (target/'labels/train'/(stem+'_objects_only.txt')).write_text(r['clean_label'])
            derived.append(dict(source=path.name,name=stem+'_objects_only.jpg',kind='context_removed_positive'))
            if train_index%4==0:
                negative=image.copy()
                for box in r['boxes']:
                    x1,y1,x2,y2=np.rint(box['bbox_xyxy_px']).astype(int)
                    negative[max(0,y1-7):min(480,y2+7),max(0,x1-7):min(640,x2+7)]=40
                cv2.imwrite(str(target/'images/train'/(stem+'_synthetic_absent.jpg')),negative)
                (target/'labels/train'/(stem+'_synthetic_absent.txt')).write_text('')
                derived.append(dict(source=path.name,name=stem+'_synthetic_absent.jpg',kind='synthetic_both_objects_removed'))
            train_index+=1
    (target/'data.yaml').write_text(f'path: {target}\ntrain: images/train\nval: images/val\nnames:\n  0: cup\n  1: target\n')
    report=dict(classes=NAMES,label_archive_sha256=sha(archive),source_manifest_sha256=sha(source/'manifest.json'),
        source_image_directory=str(source/'images'),labeled_images=len(records),
        excluded_missing_labels=missing,removed_invalid_boxes=removed,
        split_counts=dict(Counter(r['split'] for r in records)),
        class_counts=dict(Counter(b['label'] for r in records for b in r['boxes'])),
        training_images_with_augmentation=len(list((target/'images/train').glob('*.jpg'))),
        validation_episodes=sorted(VAL_EPISODES),validation_new_originals=sorted(VAL_NEW),
        split_qualification='Old videos split by entire episode. New oblique-view group held together, but still same capture session with possible similar views. Development validation for checkpoint selection, NOT independent field accuracy.',
        records=records,derived_training_images=derived)
    (ROOT/'annotation-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for page in range(3):
        sheet=np.full((4*272,5*320,3),235,np.uint8)
        for j,r in enumerate(records[page*20:(page+1)*20]):
            im=cv2.imread(str(source/'images'/r['filename']))
            for b in r['boxes']:
                x1,y1,x2,y2=np.rint(b['bbox_xyxy_px']).astype(int)
                color=(0,150,255) if b['class_id']==0 else (255,220,0)
                cv2.rectangle(im,(x1,y1),(x2,y2),color,1)
                cv2.putText(im,b['label'],(x1,max(10,y1-3)),0,.35,color,1)
            row,col=divmod(j,5)
            sheet[row*272:row*272+240,col*320:(col+1)*320]=cv2.resize(im,(320,240))
            cv2.putText(sheet,r['filename'].split('_')[0]+' '+r['split'],(col*320+5,row*272+260),0,.5,(0,0,0),1)
        cv2.imwrite(str(ROOT/f'manual-labels-{page+1}.jpg'),sheet)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','derived_training_images']},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    a=p.parse_args();main(a.source,a.labels)
