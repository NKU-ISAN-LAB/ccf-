import hashlib,json,unittest
from pathlib import Path
import cv2,numpy as np
from recognize import ROOT
from metrics import score
from evaluate import variants
class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=json.loads((ROOT/'annotation-audit.json').read_text())
    def test_counts_and_split(self):
        rs=self.audit['records'];self.assertEqual(len(rs),20);self.assertEqual(sum(len(r['boxes_xyxy']) for r in rs),59)
        self.assertEqual({r['frame'] for r in rs if r['split']=='test'},{f'{n:04d}' for n in range(16,21)})
        for split in ['train','val','test']:
            stems={p.stem.split('_')[0] for p in (ROOT/'dataset/images'/split).glob('*.jpg') if not p.stem.startswith('negative_')}
            self.assertEqual(stems,{r['frame'] for r in rs if r['split']==split})
    def test_originals_unchanged(self):
        for r in self.audit['records']:
            self.assertEqual(hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),r['image_sha256'])
            self.assertEqual(hashlib.sha256((ROOT/'dataset/images'/r['split']/(r['frame']+'.jpg')).read_bytes()).hexdigest(),r['image_sha256'])
    def test_label_pairs_and_positive_counts(self):
        for split in ['train','val','test']:
            images=list((ROOT/'dataset/images'/split).glob('*.jpg'));labels=list((ROOT/'dataset/labels'/split).glob('*.txt'))
            self.assertEqual({p.stem for p in images},{p.stem for p in labels})
            for p in labels:
                lines=p.read_text().splitlines()
                if not lines:
                    self.assertTrue('synthetic_absent' in p.stem or p.stem.startswith('negative_'));continue
                original=next(r for r in self.audit['records'] if r['frame']==p.stem.split('_')[0])
                self.assertEqual(len(lines),len(original['boxes_xyxy']))
                for line in lines:
                    cls,x,y,w,h=map(float,line.split());self.assertEqual(cls,0)
                    self.assertTrue(0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1)
    def test_transforms_preserve_all_objects_and_source(self):
        for r in self.audit['records']:
            if r['split']!='test':continue
            image=cv2.imread(r['image']);original=image.copy();vs=list(variants(image,r['boxes_xyxy']))
            self.assertEqual(len(vs),10);self.assertTrue(np.array_equal(image,original))
            for name,im,boxes in vs:
                self.assertEqual(len(boxes),3);self.assertEqual(im.shape,(480,640,3))
                for x1,y1,x2,y2 in boxes:self.assertTrue(0<=x1<x2<=640 and 0<=y1<y2<=480,name)
    def test_one_to_one_matching(self):
        d={'bbox_xyxy_px':[10,10,30,30],'confidence':.9}
        r=score([d],[[10,10,30,30],[10,10,30,30]])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,0,1))
        r=score([d,d],[[10,10,30,30]])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,0));self.assertFalse(r['all_correct'])
    def test_negative_episode_split_and_empty_labels(self):
        rs=json.loads((ROOT/'negative-source-audit.json').read_text())['records']
        groups={s:{r['episode'] for r in rs if r['split']==s} for s in ['train','val','negative_test']}
        self.assertFalse(groups['train']&groups['val']);self.assertFalse(groups['train']&groups['negative_test']);self.assertFalse(groups['val']&groups['negative_test'])
        self.assertEqual(len(rs),20)
        for r in rs:
            label=ROOT/'dataset/labels'/r['split']/(r['case']+'.txt')
            self.assertEqual(label.read_text(),'')
            self.assertEqual(hashlib.sha256(Path(r['source']).read_bytes()).hexdigest(),r['source_sha256'])
if __name__=='__main__':unittest.main()
