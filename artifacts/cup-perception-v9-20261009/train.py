"""Warm-start from v8, retain old-view replay, learn manually labeled new views."""
import os
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('MKL_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    audit=json.loads((ROOT/'annotation-audit.json').read_text())
    weights=Path(audit['init_weights'])
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==audit['init_sha256']
    torch.set_num_threads(6)
    model=YOLO(str(weights));assert model.names=={0:'exposed_cup'}
    model.train(data=str(ROOT/'dataset/data.yaml'),epochs=35,imgsz=640,batch=8,device='cpu',workers=0,
        project=str(ROOT/'runs'),name='supplement-finetune',exist_ok=False,seed=209,deterministic=True,
        patience=15,optimizer='AdamW',lr0=.0001,lrf=.15,weight_decay=.0005,
        freeze=0,amp=False,cache=False,plots=True,
        mosaic=0.,mixup=0.,copy_paste=0.,fliplr=0.,flipud=0.,
        degrees=15.,translate=.12,scale=.35,shear=5.,perspective=.00025,
        hsv_h=.015,hsv_s=.35,hsv_v=.3,close_mosaic=0)
