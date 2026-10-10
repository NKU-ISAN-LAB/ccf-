"""Package portable inference + evidence, excluding raw images and training runs."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    verification=json.loads((ROOT/'verification.json').read_text())
    parity=json.loads((ROOT/'onnx-parity.json').read_text())
    assert parity['frames']==parity['passed']==60
    preservation={
        '/home/vnv/cup-perception-v9-20261009/weights/best.pt':'ad2b1e27e2ba8e146c80697e1be53b9302e70e5fbb4b331c294f549c9899eafc',
        '/home/vnv/task2-cup-target-yolo-20261008/weights/best.pt':'f51e8527d8caec8f2608c2d4eb1c60c49bc5a6693b60e3ebddf1f6453af9fec5',
        '/home/vnv/task2-cup-covered-yolo-v2-20261009/weights/best.pt':'1a042dbd1ccd17368db53b763cb29c7088a27ca7c11ab8906f5b8a698366c51b',
    }
    for path,digest in preservation.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    audit=json.loads((ROOT/'annotation-audit.json').read_text())
    for r in audit['records']:
        assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['image_sha256']
    assert hashlib.sha256(Path(audit['label_zip']).read_bytes()).hexdigest()==audit['label_zip_sha256']
    (ROOT/'preservation-check.json').write_text(json.dumps(dict(previous_weights_unchanged=preservation,
        original_images_unchanged=60,label_zip_unchanged=True),indent=2))
    files=[]
    for pattern in ['*.py','*.json','README.md','requirements.txt','training-results.csv','training-args.yaml','prediction-sheet-*.jpg','prediction-detail-*.jpg','negative-failures.jpg','unit-tests.log','weights/*']:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file() and p.name!='package-verification.json')
    files=sorted(set(files))
    target=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in files:
            assert z.read(str(Path(ROOT.name)/p.relative_to(ROOT)))==p.read_bytes()
    report=dict(zip=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
                files=len(files),raw_training_images_included=False,pretrained_coco_weights_included=False,
                contents={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (ROOT/'package-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='contents'},indent=2))
