"""Paired v5/v6 synthetic stress replay; no new reference frames or training."""
import argparse,importlib.util,json,sys
from pathlib import Path
import cv2
import numpy as np
from recognize import Detector,ROOT,StabilityGate

parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--baseline',type=Path,default=ROOT.parent/'cup-perception-v5-20261008/recognize.py')
args=parser.parse_args()
DATA=args.data
spec=importlib.util.spec_from_file_location('v5_baseline',args.baseline)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
before=old.Detector();after=Detector();serial='CP2L863000RG'
records=[]
for name in ['0002','0031','0032','0038','0045','0047','0049','0050']:
    image=cv2.imread(str(DATA/'color'/f'{name}.jpg'))
    _,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,65])
    changes={'original':image,'dim':np.clip(image.astype(float)*.8+8,0,255).astype(np.uint8),
             'bright':np.clip(image.astype(float)*1.15+5,0,255).astype(np.uint8),
             'blur3':cv2.GaussianBlur(image,(3,3),.6),
             'jpeg65':cv2.imdecode(encoded,cv2.IMREAD_COLOR),
             'shift':cv2.warpAffine(image,np.float64([[1,0,5],[0,1,-4]]),(640,480))}
    base=after.predict(image,serial)
    for kind,im in changes.items():
        a=before.predict(im,serial);b=after.predict(im,serial)
        drift=None
        if b['candidate_valid_2d'] and base['candidate_valid_2d']:
            expected=np.array(base['visible_lower_rim_uv'])+([5,-4] if kind=='shift' else [0,0])
            drift=float(np.linalg.norm(np.array(b['visible_lower_rim_uv'])-expected))
        records.append({'frame':name,'change':kind,'v5':a['candidate_valid_2d'],'v6':b['candidate_valid_2d'],
                        'point':b['visible_lower_rim_uv'],'drift_from_original_px':drift})
    # Keep the table and holder; occlude only the observed lower edge, separately
    # from the full-cup removals tested by verify.py.
    if base['candidate_valid_2d']:
        x1,y1,x2,y2=np.rint(base['cup_bbox_xyxy_px']).astype(int)
        point=base['visible_lower_rim_uv'];occluded=image.copy()
        occluded[max(y1,int(point[1])-7):y2+2,max(0,x1-2):x2+2]=35
        b=after.predict(occluded,serial)
        records.append({'frame':name,'change':'lower_rim_occluded','expected':False,'v6':b['candidate_valid_2d']})
positive=[r for r in records if 'v5' in r];negative=[r for r in records if 'expected' in r]
report={'positive_variants':len(positive),'v5_detected':sum(r['v5'] for r in positive),
        'v6_detected':sum(r['v6'] for r in positive),'v6_failures':[r for r in positive if not r['v6']],
        'rim_occlusions':len(negative),'rim_occlusions_rejected':sum(not r['v6'] for r in negative),
        'max_positive_drift_px':max(r['drift_from_original_px'] or 0 for r in positive),
        'qualification':'Synthetic development perturbations of existing data, not field validation.',
        'records':records}
(ROOT/'stress-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
