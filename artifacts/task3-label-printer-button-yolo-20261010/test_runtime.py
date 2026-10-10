import unittest
from unittest.mock import patch
import numpy as np
from recognize import Detector, NAMES, crop_bounds, make_detection, class_nms, filter_candidates
from metrics import score, summarize


def det(box=(20, 20, 30, 30), cls=1, confidence=.8, offset=(0, 0), parent=None):
    return make_detection([*box, confidence, cls], 640, 480, offset, parent=parent)


class RuntimeTests(unittest.TestCase):
    def test_coordinate_mapping(self):
        d = det(offset=(100, 200))
        self.assertEqual(d['bbox_xyxy_px'], [120, 220, 130, 230])
        self.assertEqual(d['center_uv'], [125, 225])

    def test_bad_detection(self):
        self.assertIsNone(det(confidence=float('nan')))
        self.assertIsNone(det(cls=2))
        self.assertIsNone(det(box=(30, 20, 20, 30)))

    def test_crop_at_edges(self):
        for box in [[0, 0, 30, 40], [600, 450, 640, 480], [0, 0, 640, 480]]:
            l, t, r, b = crop_bounds(box, 640, 480)
            self.assertTrue(0 <= l < r <= 640 and 0 <= t < b <= 480)
            self.assertEqual(r-l, b-t)

    def test_class_aware_nms(self):
        self.assertEqual(len(class_nms([det(cls=0), det(cls=1)])), 2)
        self.assertEqual(len(class_nms([det(), det(confidence=.7)])), 1)
        self.assertEqual(len(class_nms([det(), det(box=(31, 20, 41, 30))])), 2)

    def test_threshold_parent_gate(self):
        ds = [det(parent=.2), det(box=(40, 40, 50, 50), parent=.8)]
        self.assertEqual(len(filter_candidates(ds, dict(label_printer=.3, button=.4))), 1)

    def test_each_button_matches_only_once(self):
        gt = [det(), det(box=(31, 20, 41, 30))]
        s = score([det(), det()], gt)['button']
        self.assertEqual((s['tp'], s['fp'], s['fn']), (1, 1, 1))

    def test_all_correct_requires_all_buttons(self):
        gt = [det(cls=0), det(), det(box=(31, 20, 41, 30))]
        summary = summarize([dict(filename='x', scores=score([det(cls=0), det()], gt))])
        self.assertEqual(summary['all_objects_correct'], 0)

    def test_invisible_buttons_can_be_absent(self):
        summary = summarize([dict(filename='x', scores=score([det(cls=0)], [det(cls=0)]))])
        self.assertEqual(summary['all_objects_correct'], 1)

    def test_invalid_image_no_inference(self):
        d = Detector.__new__(Detector)
        d.thresholds = dict(label_printer=.3, button=.3)
        d.implementation = {}
        for image in [None, np.zeros((60, 60)), np.zeros((60, 60, 3), dtype=float)]:
            result = d.predict(image)
            self.assertFalse(result['ok'])
            self.assertFalse(result['robot_motion_ready'])

    def test_predicted_crop_not_gt(self):
        d = Detector.__new__(Detector)
        calls = []
        def raw(im):
            calls.append(im.shape)
            if len(calls) == 1:
                return np.array([[200, 200, 300, 300, .9, 0]])
            return np.array([[20, 30, 28, 38, .9, 1], [80, 90, 88, 98, .8, 1]])
        d._raw = raw
        candidates = d.candidates(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(calls, [(480, 640, 3), (180, 180, 3)])
        buttons = [x for x in candidates if x['class_id'] == 1]
        self.assertEqual(len(buttons), 1)  # outside-printer candidate rejected
        self.assertEqual(buttons[0]['bbox_xyxy_px'], [240, 250, 248, 258])

    def test_multiple_buttons_no_arbitrary_press(self):
        d = Detector.__new__(Detector)
        d.thresholds = dict(label_printer=.3, button=.3)
        d.implementation = {}
        d.candidates = lambda image: [det(cls=0), det(), det(box=(31, 20, 41, 30))]
        result = d.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(len(result['objects']['button']), 2)
        self.assertIsNone(result['press_pose'])
        self.assertIsNone(result['button_function'])
        self.assertFalse(result['robot_motion_ready'])

    def test_wrong_class_mapping_rejected(self):
        with patch('recognize.Path.read_bytes', return_value=b'fake'), patch('recognize.YOLO') as model:
            model.return_value.names = {0:'cup',1:'target'}
            with self.assertRaisesRegex(ValueError, 'Wrong class mapping'):
                Detector(weights='fake.pt', confidence=.3)

    def test_invalid_threshold_rejected(self):
        with patch('recognize.Path.read_bytes', return_value=b'fake'):
            for thresholds in [.01, float('nan'), {'label_printer':.4}, {'label_printer':.5,'button':1.0}]:
                with self.assertRaises(ValueError):
                    Detector(weights='fake.pt', confidence=thresholds)

    def test_wrong_hash_rejected(self):
        with patch('recognize.Path.read_bytes', return_value=b'fake'), patch('recognize.Path.read_text', return_value='{"weights_sha256":"incorrect"}'):
            with self.assertRaisesRegex(ValueError, 'Weights and calibrated config'):
                Detector(weights='fake.pt')


if __name__ == '__main__':
    unittest.main()
