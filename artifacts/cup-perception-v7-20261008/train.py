"""Fine-tune a generic pretrained detector, never the old nut/cup-tube model."""
import argparse,os
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('MKL_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pretrained',required=True)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--name',default='direct-cup')
    a=p.parse_args()
    torch.set_num_threads(6)
    model=YOLO(a.pretrained)
    assert len(model.names)==80,'Use generic COCO pretrained weights, not old task weights.'
    model.train(data=str(ROOT/'dataset/data.yaml'),epochs=a.epochs,imgsz=640,
                batch=8,device='cpu',workers=0,project=str(ROOT/'runs'),name=a.name,
                exist_ok=False,seed=42,deterministic=True,patience=18,
                optimizer='AdamW',lr0=.001,lrf=.05,weight_decay=.0005,
                freeze=10,amp=False,cache=False,plots=True,
                mosaic=0.,mixup=0.,copy_paste=0.,fliplr=0.,flipud=0.,
                degrees=8.,translate=.15,scale=.3,perspective=0.,
                hsv_h=.015,hsv_s=.35,hsv_v=.3,close_mosaic=0)
