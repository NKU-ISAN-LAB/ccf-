"""Inspect supplemental user labels without modifying their source files."""
import hashlib,json,zipfile
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/vnv/yizhi/yzb2026-competition-ws/CCF数据集-20261009-123651')
LABELS=SOURCE.parent/'labels_my-project-name_2026-10-09-12-52-34.zip'
if __name__=='__main__':
    records=[]
    with zipfile.ZipFile(LABELS) as z:
        names=sorted(z.namelist())
        assert len(names)==len(set(names))
        for name in names:
            assert Path(name).name==name and name.endswith('.txt')
            text=z.read(name).decode('utf-8-sig').strip()
            lines=[l.split() for l in text.splitlines() if l.strip()]
            assert len(lines)==1 and len(lines[0])==5,name
            cls,x,y,w,h=map(float,lines[0])
            assert cls==0 and np.isfinite([x,y,w,h]).all() and w>0 and h>0
            assert 0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1
            path=SOURCE/'color'/(Path(name).stem+'.jpg')
            im=cv2.imread(str(path));assert im is not None,name
            height,width=im.shape[:2]
            box=(np.array([x-w/2,y-h/2,x+w/2,y+h/2])*[width,height,width,height]).tolist()
            records.append(dict(frame=path.stem,image=str(path),image_size_wh=[width,height],bbox_xyxy=box,
                                yolo=[0,x,y,w,h],image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                label_sha256=hashlib.sha256(z.read(name)).hexdigest()))
    sheet=np.full((4*264,5*320,3),235,np.uint8)
    for i,r in enumerate(records):
        im=cv2.imread(r['image']);x1,y1,x2,y2=np.rint(r['bbox_xyxy']).astype(int)
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
        row,col=divmod(i,5)
        sheet[row*264:row*264+240,col*320:(col+1)*320]=cv2.resize(im,(320,240))
        cv2.putText(sheet,r['frame'],(col*320+5,row*264+257),0,.5,(0,0,0),1)
    cv2.imwrite(str(ROOT/'supplement-label-review.jpg'),sheet)
    report=dict(source=str(SOURCE),label_zip=str(LABELS),label_zip_sha256=hashlib.sha256(LABELS.read_bytes()).hexdigest(),
                count=len(records),class_ids=[0],records=records)
    (ROOT/'supplement-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
