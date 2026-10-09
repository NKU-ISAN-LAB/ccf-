import csv,hashlib,json,shutil
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.set_num_threads(4);run=ROOT/'runs/covered-cup';weights=ROOT/'weights';weights.mkdir(exist_ok=True)
    if (weights/'best.pt').exists():raise FileExistsError('Do not overwrite packaged model')
    selection=json.loads((ROOT/'checkpoint-selection.json').read_text())
    shutil.copy2(Path(selection['selected_weights']),weights/'best.pt')
    shutil.copy2(run/'results.csv',ROOT/'training-results.csv');shutil.copy2(run/'args.yaml',ROOT/'training-args.yaml')
    rows=list(csv.DictReader((run/'results.csv').open()));best=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    model=YOLO(str(weights/'best.pt'));assert model.names=={0:'cup_covered'}
    model.export(format='onnx',imgsz=640,batch=1,device='cpu',opset=17,simplify=False,dynamic=False,half=False)
    report=dict(epochs_completed=len(rows),training_best_map_epoch=int(best['epoch']),
        packaged_checkpoint=selection['selected'],packaged_epoch=int(best['epoch']) if selection['selected']=='best' else len(rows),
        selected_by=selection['policy'],best_validation_map_metrics=best,
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.iterdir() if p.is_file()})
    (ROOT/'training-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
