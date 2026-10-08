"""Export the trained task-2 checkpoint without changing the source weight."""
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    model=YOLO(str(ROOT/'weights/best.pt'))
    assert model.names=={0:'cup',1:'target'}
    model.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,device='cpu')
