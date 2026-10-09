"""Package only the validation-selected checkpoint; do not alter older models."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.set_num_threads(4)
    run=ROOT/'runs/supplement-finetune'
    weights=ROOT/'weights';weights.mkdir(exist_ok=True)
    if (weights/'best.pt').exists():raise FileExistsError('Refusing to overwrite packaged weights')
    shutil.copy2(run/'weights/best.pt',weights/'best.pt')
    shutil.copy2(run/'results.csv',ROOT/'training-results.csv')
    shutil.copy2(run/'args.yaml',ROOT/'training-args.yaml')
    rows=list(csv.DictReader((run/'results.csv').open()))
    best=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    model=YOLO(str(weights/'best.pt'))
    assert model.names=={0:'exposed_cup'}
    export=model.export(format='onnx',imgsz=640,batch=1,device='cpu',opset=17,
                        simplify=False,dynamic=False,half=False)
    assert Path(export)==weights/'best.onnx'
    report=dict(epochs_completed=len(rows),best_validation_epoch=int(best['epoch']),
                selected_by='7-frame development validation mAP50-95 (4 old + 3 new), never reserved new test',
                best_validation_metrics=best,
                hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.iterdir() if p.is_file()})
    (ROOT/'training-summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
