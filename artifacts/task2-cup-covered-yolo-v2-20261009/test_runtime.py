import unittest
from unittest.mock import Mock,patch
import numpy as np,torch
from recognize import Detector
class RuntimeTests(unittest.TestCase):
    def detector(self,boxes=()):
        d=Detector.__new__(Detector);d.confidence=.2;d.device='cpu';d.implementation={}
        pred=Mock();pred.boxes.data=torch.tensor(boxes,dtype=torch.float32).reshape(-1,6)
        d.model=Mock();d.model.predict.return_value=[pred];return d
    def test_multiple_covered_cups_are_all_returned(self):
        d=self.detector([[10,20,40,50,.9,0],[100,120,150,160,.8,0],[200,220,240,260,.7,0]])
        with patch('cv2.SIFT_create',side_effect=AssertionError('No station matching')):
            r=d.predict(np.zeros((480,640,3),np.uint8))
        self.assertEqual(len(r['detections']),3);self.assertEqual(len(r['covered_cup_centers_uv']),3)
        self.assertEqual(set(r['objects']),{'cup_covered'})
        self.assertFalse(r['objects']['cup_covered']['unique']);self.assertIsNone(r['single_covered_cup_center_uv'])
        self.assertIsNone(r['selected_grasp_candidate']);self.assertIsNone(r['grasp_pose']);self.assertFalse(r['robot_motion_ready'])
    def test_single_target_semantics(self):
        r=self.detector([[10,20,40,50,.9,0]]).predict(np.zeros((480,640,3),np.uint8))
        self.assertEqual(r['single_covered_cup_center_uv'],[25,35]);self.assertIsNone(r['seal_quality_verified'])
        self.assertEqual(r['detections'][0]['label'],'cup_covered');self.assertNotIn('target',r['objects'])
    def test_no_detection_no_stale_coordinates(self):
        r=self.detector().predict(np.zeros((480,640,3),np.uint8))
        self.assertEqual(r['covered_cup_centers_uv'],[]);self.assertIsNone(r['single_covered_cup_center_uv'])
    def test_invalid_input(self):
        d=self.detector()
        for im in [None,np.zeros((480,640)),np.zeros((480,640,3),float),np.zeros((5,5,3),np.uint8)]:
            self.assertFalse(d.predict(im)['ok'])
        d.model.predict.assert_not_called()
    def test_unknown_classes_invalid_boxes_and_low_scores_ignored(self):
        d=self.detector([[1,1,2,2,.9,1],[1,1,2,2,.1,0],[float('nan'),1,2,2,.9,0],[2,1,1,2,.9,0]])
        self.assertEqual(d.predict(np.zeros((480,640,3),np.uint8))['detections'],[])
    def test_camera_identity_not_gating(self):
        for serial in [None,'other-camera']:
            r=self.detector([[1,1,20,20,.9,0]]).predict(np.zeros((480,640,3),np.uint8),serial)
            self.assertEqual(len(r['detections']),1)
    def test_old_model_classes_rejected_even_with_explicit_threshold(self):
        for names in [{0:'cup',1:'target'},{0:'exposed_cup'}]:
            with patch('pathlib.Path.read_bytes',return_value=b'test'),patch('recognize.YOLO',return_value=Mock(names=names)):
                with self.assertRaisesRegex(ValueError,'Wrong class mapping'):
                    Detector(weights='/unused-test-weights.pt',confidence=.45)
    def test_invalid_threshold_rejected(self):
        for threshold in [0,1,-.1,float('nan'),float('inf')]:
            with patch('pathlib.Path.read_bytes',return_value=b'test'):
                with self.assertRaises(ValueError):Detector(weights='/unused-test-weights.pt',confidence=threshold)
if __name__=='__main__':unittest.main()
