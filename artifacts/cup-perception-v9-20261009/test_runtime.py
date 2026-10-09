import json,unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from recognize import Detector,StabilityGate

class RuntimeTests(unittest.TestCase):
    def detector(self,boxes=()):
        detector=Detector.__new__(Detector)
        detector.confidence=.5;detector.device='cpu';detector.implementation={}
        output=Mock()
        output.boxes.data=torch.tensor(boxes,dtype=torch.float32).reshape(-1,6)
        detector.model=Mock();detector.model.predict.return_value=[output]
        return detector

    def test_direct_detection_does_not_call_station_registration(self):
        detector=self.detector([[100,120,150,150,.9,0]])
        with patch('cv2.SIFT_create',side_effect=AssertionError('Must not match a machine')):
            result=detector.predict(np.zeros((480,640,3),np.uint8))
        self.assertTrue(result['candidate_valid_2d'])
        self.assertFalse(result['requires_station_registration'])
        self.assertEqual(result['cup_center_uv'],[125,135])
        self.assertIsNone(result['visible_lower_rim_uv'])
        self.assertIsNone(result['grasp_pose']);self.assertFalse(result['robot_motion_ready'])
        json.dumps(result,allow_nan=False)

    def test_no_detection_has_no_coordinates(self):
        result=self.detector().predict(np.zeros((480,640,3),np.uint8))
        self.assertFalse(result['candidate_valid_2d']);self.assertIsNone(result['cup_center_uv'])
        self.assertEqual(result['reasons'],['no_exposed_cup_detected'])

    def test_multiple_candidates_are_not_arbitrarily_selected(self):
        result=self.detector([[10,10,40,40,.9,0],[100,100,130,130,.8,0]]).predict(np.zeros((480,640,3),np.uint8))
        self.assertEqual(len(result['detections']),2)
        self.assertFalse(result['candidate_valid_2d']);self.assertIsNone(result['cup_bbox_xyxy_px'])

    def test_invalid_image_does_not_reach_model(self):
        d=self.detector()
        for im in [None,np.zeros((480,640)),np.zeros((480,640,3),float),np.zeros((5,5,3),np.uint8)]:
            self.assertFalse(d.predict(im)['candidate_valid_2d'])
        d.model.predict.assert_not_called()

    def test_nonfinite_predictions_are_not_published(self):
        d=self.detector([[float('nan'),1,2,3,.9,0],[1,1,2,3,float('inf'),0]])
        self.assertFalse(d.predict(np.zeros((480,640,3),np.uint8))['candidate_valid_2d'])

    def test_input_identity_is_metadata_not_machine_gate(self):
        d=self.detector([[100,120,150,150,.9,0]])
        for serial in [None,'CP2L863000RG','another-camera']:
            r=d.predict(np.zeros((480,640,3),np.uint8),serial)
            self.assertTrue(r['candidate_valid_2d']);self.assertEqual(r['camera_serial'],serial)

    def candidate(self,x=20):
        return {'candidate_valid_2d':True,'cup_center_uv':[x,30]}

    def test_three_frames_needed(self):
        g=StabilityGate()
        for frame in range(1,4):
            self.assertEqual(g.update(self.candidate(),frame,float(frame))['stable_2d'],frame==3)

    def test_invalid_detection_resets_no_old_point_reuse(self):
        g=StabilityGate()
        for frame in range(1,4):g.update(self.candidate(),frame,float(frame))
        r=g.update({'candidate_valid_2d':False,'cup_center_uv':None},4,4.)
        self.assertFalse(r['stable_2d']);self.assertEqual(len(g.points),0)
        self.assertFalse(g.update(self.candidate(),5,5.)['stable_2d'])

    def test_duplicate_gap_and_large_jitter(self):
        for frame,time in [(2,3.),(3,10.),(3,float('nan'))]:
            g=StabilityGate();g.update(self.candidate(),1,1.);g.update(self.candidate(),2,2.)
            self.assertFalse(g.update(self.candidate(),frame,time)['stable_2d'])
            self.assertEqual(len(g.points),0)
        g=StabilityGate();g.update(self.candidate(),1,1.);g.update(self.candidate(),2,2.)
        self.assertFalse(g.update(self.candidate(40),3,3.)['stable_2d'])

if __name__=='__main__':unittest.main()
