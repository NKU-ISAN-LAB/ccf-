"""Uniform full-frame review: never crop successful frames differently."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from recognize import ROOT,annotate

parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
data=parser.parse_args().data
rows=json.loads((ROOT/'replay.json').read_text())
for page in range((len(rows)+24)//25):
    sheet=np.full((1320,1600,3),235,np.uint8)
    for j,r in enumerate(rows[page*25:page*25+25]):
        image=cv2.imread(str(data/'color'/f"{r['frame']}.jpg"))
        assert image.shape==(480,640,3)
        preview=annotate(image,r);row,col=divmod(j,5)
        sheet[row*264:row*264+240,col*320:(col+1)*320]=cv2.resize(preview,(320,240))
        label=r['frame']+(' DETECTED' if r['candidate_valid_2d'] else ' NOT DETECTED')
        cv2.putText(sheet,label,(col*320+5,row*264+257),0,.5,(0,0,0),1)
    cv2.imwrite(str(ROOT/f'full-frame-grid-{page+1}.jpg'),sheet)
