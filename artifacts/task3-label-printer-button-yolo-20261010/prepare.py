"""Audit user labels and build independent printer+button training data."""
import hashlib,json,shutil,zipfile
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/vnv/label-printer-60-20261010')
LABELS=Path('/home/vnv/yizhi/yzb2026-competition-ws/labels_my-project-name_2026-10-10-10-32-46.zip')
NAMES={0:'label_printer',1:'button'}

def crop_bounds(box,width,height):
    x1,y1,x2,y2=box;side=max(160,int(np.ceil(max(x2-x1,y2-y1)*1.8)))
    side=min(side,width,height)
    left=int(np.clip((x1+x2)/2-side/2,0,width-side));top=int(np.clip((y1+y2)/2-side/2,0,height-side))
    return left,top,left+side,top+side

def write_example(out,split,stem,image,boxes):
    assert cv2.imwrite(str(out/'images'/split/(stem+'.jpg')),image,[cv2.IMWRITE_JPEG_QUALITY,98])
    height,width=image.shape[:2];lines=[]
    for b in boxes:
        x1,y1,x2,y2=b['bbox_xyxy_px'];assert 0<=x1<x2<=width and 0<=y1<y2<=height
        lines.append(f"{b['class_id']} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    (out/'labels'/split/(stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''))

def main():
    out=ROOT/'dataset'
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((SOURCE/'manifest.json').read_text());records=[]
    with zipfile.ZipFile(LABELS) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())=={Path(r['filename']).stem+'.txt' for r in manifest['records']}
        for r in manifest['records']:
            filename=r['filename'];stem=Path(filename).stem;path=SOURCE/'images'/filename
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256']
            raw=z.read(stem+'.txt');boxes=[]
            for line in raw.decode('utf-8-sig').splitlines():
                if not line.strip():continue
                row=list(map(float,line.split()));assert len(row)==5
                cls,x,y,w,h=row;assert cls in NAMES and np.isfinite(row).all() and w>0 and h>0
                assert 0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1
                box=(np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[640,480,640,480]).tolist()
                boxes.append(dict(class_id=int(cls),label=NAMES[int(cls)],bbox_xyxy_px=box))
            assert sum(b['class_id']==0 for b in boxes)==1
            assert sum(b['class_id']==1 for b in boxes) in [0,2]
            group=r['source_group'];episode=r.get('episode');original=r.get('original_name')
            if group=='rgb_20261009' or episode=='episode_000199' or (group=='rgb_20261008' and original=='0040.jpg'):split='test'
            elif episode in ['episode_000105','episode_000175'] or (group=='rgb_20261008' and original in ['0033.jpg','0038.jpg']):split='val'
            else:split='train'
            records.append(dict(filename=filename,source=str(path),source_group=group,episode=episode,original_name=original,split=split,
                                boxes=boxes,image_sha256=r['image_sha256'],label_sha256=hashlib.sha256(raw).hexdigest()))
    for split in ['train','val','test']:
        (out/'images'/split).mkdir(parents=True);(out/'labels'/split).mkdir(parents=True)
    for i,r in enumerate(records):
        im=cv2.imread(r['source']);assert im.shape==(480,640,3)
        split=r['split'];stem=Path(r['filename']).stem
        write_example(out,split,stem,im,r['boxes']);shutil.copy2(r['source'],out/'images'/split/r['filename'])
        # GT crops only in TRAIN/DEVELOPMENT validation, never used by runtime or test inference.
        if split in ['train','val']:
            printer=next(b['bbox_xyxy_px'] for b in r['boxes'] if b['class_id']==0)
            left,top,right,bottom=crop_bounds(printer,640,480)
            bs=[dict(b,bbox_xyxy_px=(np.array(b['bbox_xyxy_px'])-[left,top,left,top]).tolist()) for b in r['boxes']]
            write_example(out,split,stem+'_printer_crop',im[top:bottom,left:right],bs)
        if split=='train' and i%5==0:
            absent=im.copy()
            for b in r['boxes']:
                x1,y1,x2,y2=np.rint(b['bbox_xyxy_px']).astype(int)
                absent[max(0,y1-5):min(480,y2+5),max(0,x1-5):min(640,x2+5)]=40
            write_example(out,split,stem+'_synthetic_absent',absent,[])
    audit=dict(classes=NAMES,label_zip=str(LABELS),label_zip_sha256=hashlib.sha256(LABELS.read_bytes()).hexdigest(),records=records,
        split_counts={s:dict(images=sum(r['split']==s for r in records),classes={name:sum(b['class_id']==cls for r in records if r['split']==s for b in r['boxes']) for cls,name in NAMES.items()}) for s in ['train','val','test']},
        training_images=len(list((out/'images/train').glob('*.jpg'))),validation_images_with_gt_crops=len(list((out/'images/val').glob('*.jpg'))),
        policy='Same episode across steps kept together. Test: entire Oct09 capture (6), old episode199 (2), truncated Oct08 frame0040 (1). Val: old episodes105/175 plus Oct08 frames0033/0038. Test excluded from training, threshold selection and checkpoint selection. GT crops only used for train/development; test inference must use predicted printer boxes.',
        qualification='Only one physical setup/device. Old episodes share nearly identical printer views. This is not independent field accuracy or arbitrary-model generalization.',
        button_semantics='Two same-class button instances, no function identity; pixel detections are not robot pressing coordinates.')
    (ROOT/'annotation-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: label_printer\n  1: button\n')
    for page in range(3):
        sheet=np.full((1088,1600,3),235,np.uint8)
        for j,r in enumerate(records[page*20:(page+1)*20]):
            im=cv2.imread(r['source'])
            for b in r['boxes']:
                x1,y1,x2,y2=np.rint(b['bbox_xyxy_px']).astype(int)
                cv2.rectangle(im,(x1,y1),(x2,y2),(255,220,0) if b['class_id']==0 else (0,255,0),1)
            row,col=divmod(j,5);sheet[row*272:row*272+240,col*320:(col+1)*320]=cv2.resize(im,(320,240))
            cv2.putText(sheet,r['filename'][:3]+' '+r['split'],(col*320+4,row*272+257),0,.5,(0,0,0),1)
        cv2.imwrite(str(ROOT/f'manual-label-review-{page+1}.jpg'),sheet)
    # Enlargements are review aids only, not training images or inferred labels.
    for index in [0,38,44,54]:
        r=records[index];im=cv2.imread(r['source']);box=r['boxes'][0]['bbox_xyxy_px'];l,t,rr,bb=crop_bounds(box,640,480)
        for b in r['boxes']:
            x1,y1,x2,y2=np.rint(b['bbox_xyxy_px']).astype(int);cv2.rectangle(im,(x1,y1),(x2,y2),(255,220,0) if b['class_id']==0 else (0,255,0),1)
        cv2.imwrite(str(ROOT/f'label-detail-{index+1:03d}.jpg'),cv2.resize(im[t:bb,l:rr],(600,600)))
    print(json.dumps({k:v for k,v in audit.items() if k!='records'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
