import json,unittest
from unittest.mock import Mock,patch
import numpy as np,torch
from recognize import Detector,NAMES,box_iou
from metrics import score

class RuntimeTests(unittest.TestCase):
    def detector(self,boxes=()):
        d=Detector.__new__(Detector);d.thresholds={'cup':.2,'target':.3}
        d.device='cpu';d.implementation={};d.model=Mock()
        output=Mock();output.boxes.data=torch.tensor(boxes,dtype=torch.float32).reshape(-1,6)
        d.model.predict.return_value=[output];return d
    def image(self):return np.zeros((480,640,3),np.uint8)
    def test_overlapping_classes_preserved(self):
        d=self.detector([[100,100,150,150,.9,0],[100,100,150,150,.8,1]])
        r=d.predict(self.image())
        self.assertTrue(r['both_unique_2d']);self.assertAlmostEqual(r['cup_target_bbox_iou'],1.)
        self.assertFalse(d.model.predict.call_args.kwargs['agnostic_nms'])
        self.assertIsNone(r['placement_confirmed']);self.assertIsNone(r['grasp_pose'])
        self.assertFalse(r['robot_motion_ready']);json.dumps(r,allow_nan=False)
    def test_cup_does_not_require_target(self):
        d=self.detector([[100,100,150,150,.9,0]])
        with patch('cv2.SIFT_create',side_effect=AssertionError('No machine matching')):r=d.predict(self.image())
        self.assertEqual(r['cup_center_uv'],[125,125]);self.assertIsNone(r['target_center_uv'])
        self.assertFalse(r['requires_station_registration'])
    def test_target_does_not_require_cup(self):
        r=self.detector([[100,100,150,150,.9,1]]).predict(self.image())
        self.assertEqual(r['target_center_uv'],[125,125]);self.assertIsNone(r['cup_center_uv'])
    def test_multiple_cups_not_arbitrarily_selected(self):
        r=self.detector([[10,10,30,30,.9,0],[80,80,110,110,.8,0],[100,100,150,150,.9,1]]).predict(self.image())
        self.assertIsNone(r['cup_center_uv']);self.assertIsNotNone(r['target_center_uv'])
        self.assertEqual(r['objects']['cup']['count'],2);self.assertFalse(r['both_unique_2d'])
    def test_independent_thresholds(self):
        r=self.detector([[10,10,30,30,.25,0],[80,80,110,110,.25,1]]).predict(self.image())
        self.assertTrue(r['objects']['cup']['detected']);self.assertFalse(r['objects']['target']['detected'])
    def test_invalid_input_does_not_reach_model(self):
        d=self.detector()
        for image in [None,np.zeros((480,640)),np.zeros((480,640,3),float),np.zeros((1,1,3),np.uint8)]:
            self.assertFalse(d.predict(image)['ok'])
        d.model.predict.assert_not_called()
    def test_nonfinite_unknown_or_degenerate_boxes_rejected(self):
        r=self.detector([[float('nan'),1,5,6,.9,0],[1,1,5,6,float('inf'),0],
                         [1,1,5,6,.9,2],[2,2,2,5,.9,1]]).predict(self.image())
        self.assertEqual(r['detections'],[])
    def test_absence_clears_previous_coordinates(self):
        d=self.detector([[1,1,5,5,.9,0]]);self.assertIsNotNone(d.predict(self.image())['cup_center_uv'])
        d.model.predict.return_value[0].boxes.data=torch.empty((0,6))
        self.assertIsNone(d.predict(self.image())['cup_center_uv'])
    def test_camera_serial_is_metadata(self):
        r=self.detector([[1,1,5,5,.9,0]]).predict(self.image(),'another-camera')
        self.assertEqual(r['camera_serial'],'another-camera');self.assertTrue(r['objects']['cup']['detected'])
    def test_matching_is_class_aware_and_one_to_one(self):
        gt=[dict(label=label,bbox_xyxy_px=[10,10,20,20]) for label in NAMES.values()]
        ds=[dict(**g,confidence=.9) for g in gt]
        result=score(ds+[ds[0]],gt)
        self.assertEqual(result['cup']['tp'],1);self.assertEqual(result['cup']['fp'],1)
        self.assertEqual(result['target']['tp'],1);self.assertEqual(result['target']['fp'],0)

if __name__=='__main__':unittest.main()
