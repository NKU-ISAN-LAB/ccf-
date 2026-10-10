import csv
import hashlib
import json
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    torch.set_num_threads(4)
    weights = ROOT/'weights'
    model = YOLO(str(weights/'best.pt'))
    assert model.names == {0:'label_printer',1:'button'}
    model.export(format='onnx',imgsz=640,batch=1,device='cpu',opset=17,simplify=False,dynamic=False,half=False)
    config = json.loads((ROOT/'runtime-config.json').read_text())
    config['compatible_weights_sha256'] = [hashlib.sha256((weights/'best.onnx').read_bytes()).hexdigest()]
    (ROOT/'runtime-config.json').write_text(json.dumps(config,indent=2))
    run = ROOT/'runs/printer-button'
    for source, target in [('results.csv','training-results.csv'),('args.yaml','training-args.yaml')]:
        shutil.copy2(run/source,ROOT/target)
    rows = list(csv.DictReader((run/'results.csv').open()))
    best = max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    selection = json.loads((ROOT/'checkpoint-selection.json').read_text())
    summary = dict(epochs_completed=len(rows), trainer_best_map_epoch=int(best['epoch']),
        packaged_checkpoint=selection['selected'], packaged_epoch=int(best['epoch']) if selection['selected']=='best' else int(rows[-1]['epoch']),
        trainer_best_metrics=best, trainer_metrics_note='Validation mixes six full frames and six GT-derived crops; end-to-end verification.json uses only full frames and predicted crops.',
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.iterdir() if p.is_file()})
    (ROOT/'training-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
