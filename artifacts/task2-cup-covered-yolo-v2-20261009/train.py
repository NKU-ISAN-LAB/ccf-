import os
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('MKL_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.set_num_threads(6)
    model=YOLO('/home/vnv/task2-cup-covered-yolo-20261009/weights/best.pt')
    assert model.names=={0:'cup_covered'}
    model.train(data=str(ROOT/'dataset/data.yaml'),epochs=20,imgsz=640,batch=8,device='cpu',workers=0,
        project=str(ROOT/'runs'),name='covered-cup',exist_ok=False,seed=409,deterministic=True,
        patience=8,optimizer='AdamW',lr0=.00015,lrf=.15,weight_decay=.0005,
        freeze=0,amp=False,cache=False,plots=True,mosaic=0.,mixup=0.,copy_paste=0.,fliplr=0.,flipud=0.,
        degrees=12.,translate=.1,scale=.3,shear=4.,perspective=.00015,hsv_h=.015,hsv_s=.3,hsv_v=.25,close_mosaic=0)
