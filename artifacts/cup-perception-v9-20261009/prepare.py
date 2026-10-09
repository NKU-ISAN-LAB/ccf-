"""Merge dated label sources with namespaced names and a reserved new test block."""
import hashlib,json,shutil
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
V8=Path('/home/vnv/cup-perception-v8-20261009')
def write_example(out,split,stem,image,box):
    assert image.shape==(480,640,3)
    assert cv2.imwrite(str(out/'images'/split/(stem+'.jpg')),image)
    text=''
    if box is not None:
        x1,y1,x2,y2=box
        assert 0<=x1<x2<=640 and 0<=y1<y2<=480
        text=f'0 {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}\n'
    (out/'labels'/split/(stem+'.txt')).write_text(text)

def main():
    out=ROOT/'dataset'
    if out.exists():raise FileExistsError(out)
    old=json.loads((V8/'annotation-audit.json').read_text())
    new=json.loads((ROOT/'supplement-audit.json').read_text())
    records=[]
    for r in old['records']:
        r=dict(r);r['source']='20261008';r['id']='old_'+r['frame']
        r['image']=str(Path(old['source_images'])/(r['frame']+'.jpg'))
        r['previous_split']=r['split']
        # Former v8 test failures are now explicit training/regression examples.
        if r['split']=='test':r['split']='train'
        records.append(r)
    for r in new['records']:
        r=dict(r);r['source']='20261009';r['id']='new_'+r['frame']
        n=int(r['frame']);r['split']='train' if n<=12 else 'val' if n<=15 else 'test'
        records.append(r)
    for split in ['train','val','test']:
        (out/'images'/split).mkdir(parents=True)
        (out/'labels'/split).mkdir(parents=True)
    rng=np.random.default_rng(209)
    for r in records:
        path=Path(r['image']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256']
        split=r['split'];stem=r['id']
        if split=='unused_near_duplicate':continue
        image=cv2.imread(str(path));box=np.array(r['bbox_xyxy'])
        write_example(out,split,stem,image,box)
        shutil.copy2(path,out/'images'/split/(stem+'.jpg'))
        if split!='train':continue
        x1,y1,x2,y2=np.rint(box).astype(int)
        isolated=np.full_like(image,int(rng.integers(45,205)))
        xa,ya,xb,yb=max(0,x1-14),max(0,y1-14),min(640,x2+14),min(480,y2+14)
        isolated[ya:yb,xa:xb]=image[ya:yb,xa:xb]
        write_example(out,split,stem+'_context_removed',isolated,box)
        cw,ch=384,288
        left=int(np.clip((x1+x2)/2-cw*.45,0,640-cw));top=int(np.clip((y1+y2)/2-ch*.55,0,480-ch))
        crop=cv2.resize(image[top:top+ch,left:left+cw],(640,480))
        crop_box=(box-[left,top,left,top])*[640/cw,480/ch,640/cw,480/ch]
        write_example(out,split,stem+'_local_scale',crop,crop_box)
        if int(r['frame'])%2==0:
            absent=image.copy();absent[max(0,y1-5):min(480,y2+5),max(0,x1-5):min(640,x2+5)]=40
            write_example(out,split,stem+'_synthetic_absent',absent,None)
    report=dict(records=records,classes={'0':'exposed_cup'},
        label_zip_hashes={'20261008':old['label_zip_sha256'],'20261009':new['label_zip_sha256']},
        split_policy='Old: 22 train, 4 development validation, 24 unused near-duplicates. New: 0001-0012 train, 0013-0015 validation, 0016-0019 reserved chronological test. Old v8 held-out views are now training/regression examples, not independent test. New test never used in v8 initialization or this training/threshold selection; same-session similar viewpoints still limit independence.',
        counts={s:sum(r['split']==s for r in records) for s in ['train','val','test','unused_near_duplicate']},
        training_images=len(list((out/'images/train').glob('*.jpg'))),
        augmentation='Each train image: original + context removed + local scale; even-index train frames add synthetic removed-cup negative. No real absent-cup labels.',
        init_weights=str(V8/'weights/best.pt'),init_sha256=hashlib.sha256((V8/'weights/best.pt').read_bytes()).hexdigest())
    (ROOT/'annotation-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: exposed_cup\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
