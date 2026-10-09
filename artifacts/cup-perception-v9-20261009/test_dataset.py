import hashlib,json,unittest
from pathlib import Path
import cv2,numpy as np
from recognize import ROOT
from compare import variants,warp
from metrics import score

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=json.loads((ROOT/'annotation-audit.json').read_text())

    def test_namespaced_ids_and_new_test_isolation(self):
        rs=self.audit['records'];self.assertEqual(len(rs),69)
        self.assertEqual(len({r['id'] for r in rs}),69)
        self.assertEqual({r['id'] for r in rs if r['split']=='test'},{f'new_{n:04d}' for n in range(16,20)})
        self.assertEqual(sum(r['source']=='20261009' and r['split']=='train' for r in rs),12)
        self.assertEqual(sum(r['source']=='20261009' and r['split']=='val' for r in rs),3)
        for split in ['train','val','test']:
            expected={r['id'] for r in rs if r['split']==split}
            images={ '_'.join(p.stem.split('_')[:2]) for p in (ROOT/'dataset/images'/split).glob('*.jpg')}
            self.assertEqual(images,expected)

    def test_originals_and_initialization_unchanged(self):
        for r in self.audit['records']:
            self.assertEqual(hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),r['image_sha256'])
            if r['split']=='unused_near_duplicate':continue
            self.assertEqual(hashlib.sha256((ROOT/'dataset/images'/r['split']/(r['id']+'.jpg')).read_bytes()).hexdigest(),r['image_sha256'])
        self.assertEqual(hashlib.sha256(Path(self.audit['init_weights']).read_bytes()).hexdigest(),self.audit['init_sha256'])

    def test_image_label_pairing_and_bounds(self):
        for split in ['train','val','test']:
            images=list((ROOT/'dataset/images'/split).glob('*.jpg'))
            labels=list((ROOT/'dataset/labels'/split).glob('*.txt'))
            self.assertEqual({p.stem for p in images},{p.stem for p in labels})
            for p in labels:
                t=p.read_text().strip()
                if not t:
                    self.assertEqual(split,'train');self.assertIn('_synthetic_absent',p.stem);continue
                cls,x,y,w,h=map(float,t.split())
                self.assertEqual(cls,0);self.assertGreater(w,0);self.assertGreater(h,0)
                self.assertTrue(0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1)

    def test_transform_coordinates_and_no_input_mutation(self):
        for r in self.audit['records']:
            if r['split']!='test' and r['id'] not in {'old_0033','old_0038','old_0040','old_0049','old_0050'}:continue
            image=cv2.imread(r['image']);before=image.copy();items=list(variants(image,r['bbox_xyxy']))
            self.assertEqual(len(items),12);self.assertTrue(np.array_equal(image,before))
            for kind,im,box in items:
                self.assertEqual(im.shape,(480,640,3));self.assertEqual(im.dtype,np.uint8)
                x1,y1,x2,y2=box;self.assertTrue(0<=x1<x2<=640 and 0<=y1<y2<=480,kind)
        im=np.zeros((480,640,3),np.uint8);box=[10,20,30,40]
        out,b=warp(im,box,np.eye(3));self.assertTrue(np.array_equal(out,im));self.assertEqual(b,box)

    def test_duplicate_not_counted_as_unique_success(self):
        d={'bbox_xyxy_px':[10,10,30,30],'confidence':.9}
        r=score([d,d],[10,10,30,30]);self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,0))
        self.assertFalse(r['unique_correct'])
        r=score([d],[50,50,70,70]);self.assertEqual((r['tp'],r['fp'],r['fn']),(0,1,1))

if __name__=='__main__':unittest.main()
