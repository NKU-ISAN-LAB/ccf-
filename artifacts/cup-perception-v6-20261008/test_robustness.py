"""No added reference frames; actual failure-image and false-positive regressions."""
import json,os,unittest
from pathlib import Path
import cv2,numpy as np
from recognize import Detector,ROOT

DATA=Path(os.environ.get('CUP_TEST_DATA','missing-dataset'))/'color'

class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DATA/'0032.jpg').is_file():
            raise RuntimeError('Set CUP_TEST_DATA to the dataset directory containing color/.')
        cls.detector=Detector()

    def test_reference_bank_unchanged(self):
        old=ROOT.parent/'cup-perception-v5-20261008/reference/manifest.json'
        self.assertEqual((ROOT/'reference/manifest.json').read_bytes(),old.read_bytes())

    def test_five_original_failures_localize_visible_rim(self):
        expected={'0032':(128,266),'0038':(219,207),'0045':(182,276),
                  '0047':(145,234),'0049':(250,185)}
        for name,point in expected.items():
            with self.subTest(frame=name):
                r=self.detector.predict(cv2.imread(str(DATA/f'{name}.jpg')),'CP2L863000RG')
                self.assertTrue(r['candidate_valid_2d'],r)
                self.assertLess(np.linalg.norm(np.array(r['visible_lower_rim_uv'])-point),6)
                self.assertIsNone(r['grasp_pose']);self.assertFalse(r['robot_motion_ready'])

    def test_straight_blue_and_white_strip_not_curved_cup(self):
        image=cv2.imread(str(DATA/'0032.jpg'));image[229:276,99:162]=35
        cv2.rectangle(image,(105,248),(152,263),(100,40,30),-1)
        cv2.line(image,(105,265),(152,265),(230,230,230),3)
        self.assertFalse(self.detector.predict(image,'CP2L863000RG')['candidate_valid_2d'])

    def test_station_hidden_no_fixed_coordinate_fallback(self):
        image=cv2.imread(str(DATA/'0032.jpg'));image[:280,160:370]=100
        r=self.detector.predict(image,'CP2L863000RG')
        self.assertFalse(r['candidate_valid_2d']);self.assertIsNone(r['visible_lower_rim_uv'])

    def test_invalid_input_and_identity_fail_closed(self):
        for image,serial in [(None,'CP2L863000RG'),(np.zeros((480,640),np.uint8),'CP2L863000RG'),
                             (np.zeros((480,640,3),np.uint8),'CP2L863000KV')]:
            r=self.detector.predict(image,serial)
            self.assertFalse(r['candidate_valid_2d']);self.assertIsNone(r['visible_lower_rim_uv'])

    def test_every_reference_has_specific_rejection_diagnostics(self):
        r=self.detector.predict(np.zeros((480,640,3),np.uint8),'CP2L863000RG')
        self.assertEqual(len(r['registration_diagnostics']),7)
        for attempts in r['registration_diagnostics'].values():
            self.assertEqual(len(attempts),2)
            for a in attempts:
                self.assertFalse(a['passed']);self.assertEqual(a['failed_gate'],'insufficient_live_features')
                self.assertIn('thresholds',a)
        json.dumps(r,allow_nan=False)

if __name__=='__main__':unittest.main()
