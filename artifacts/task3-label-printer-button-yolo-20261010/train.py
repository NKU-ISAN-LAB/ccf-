import os
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('MKL_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.set_num_threads(6)
    model=YOLO('/home/vnv/yizhi/yzb2026-competition-ws/yolo11n.pt');assert len(model.names)==80
    model.train(data=str(ROOT/'dataset/data.yaml'),epochs=60,imgsz=640,batch=8,device='cpu',workers=0,
        project=str(ROOT/'runs'),name='printer-button',exist_ok=False,seed=510,deterministic=True,patience=18,
        optimizer='AdamW',lr0=.0005,lrf=.1,weight_decay=.0005,freeze=4,amp=False,cache=False,plots=True,
        mosaic=0.,mixup=0.,copy_paste=0.,fliplr=0.,flipud=0.,degrees=10.,translate=.1,scale=.25,
        shear=3.,perspective=.0001,hsv_h=.01,hsv_s=.3,hsv_v=.3,close_mosaic=0)
