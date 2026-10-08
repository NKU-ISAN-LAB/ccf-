"""Independent two-class task-2 model, initialized from generic COCO weights."""
import os,argparse
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('MKL_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pretrained',required=True);p.add_argument('--epochs',type=int,default=60)
    a=p.parse_args();torch.set_num_threads(6)
    model=YOLO(a.pretrained);assert len(model.names)==80,'Require generic COCO weights'
    model.train(data=str(ROOT/'dataset/data.yaml'),epochs=a.epochs,imgsz=640,batch=8,
        device='cpu',workers=0,project=str(ROOT/'runs'),name='cup-target',exist_ok=False,
        seed=7,deterministic=True,patience=20,optimizer='AdamW',lr0=.001,lrf=.05,
        weight_decay=.0005,freeze=10,amp=False,cache=False,plots=True,
        mosaic=0.,mixup=0.,copy_paste=0.,fliplr=0.,flipud=0.,degrees=8.,
        translate=.12,scale=.25,perspective=0.,hsv_h=.015,hsv_s=.3,hsv_v=.3,close_mosaic=0)
