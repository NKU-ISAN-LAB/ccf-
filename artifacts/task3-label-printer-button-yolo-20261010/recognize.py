"""Independent printer/button detector. Predicted printer ROI refines tiny buttons.

Coordinates are original color-image pixels, never robot pressing coordinates.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
NAMES = {0: 'label_printer', 1: 'button'}


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    return intersection / (float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection + 1e-9)


def crop_bounds(box, width, height):
    x1, y1, x2, y2 = box
    side = min(max(160, int(np.ceil(max(x2-x1, y2-y1)*1.8))), width, height)
    left = int(np.clip((x1+x2)/2-side/2, 0, width-side))
    top = int(np.clip((y1+y2)/2-side/2, 0, height-side))
    return left, top, left+side, top+side


def make_detection(raw, width, height, offset=(0, 0), method='full_frame', parent=None):
    if len(raw) != 6 or not np.isfinite(raw).all():
        return None
    x1, y1, x2, y2, confidence, cls = map(float, raw)
    if cls not in NAMES or not 0 <= confidence <= 1:
        return None
    ox, oy = offset
    box = np.clip([x1+ox, y1+oy, x2+ox, y2+oy], [0]*4, [width, height, width, height]).tolist()
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return dict(class_id=int(cls), label=NAMES[int(cls)], confidence=confidence,
                bbox_xyxy_px=box, center_uv=[(x1+x2)/2, (y1+y2)/2],
                method=method, parent_printer_confidence=parent)


def contains(box, center):
    return box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]


def class_nms(detections, iou=.3):
    kept = []
    for d in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        if not any(d['class_id'] == k['class_id'] and box_iou(d['bbox_xyxy_px'], k['bbox_xyxy_px']) > iou for k in kept):
            kept.append(d)
    return kept


def filter_candidates(candidates, thresholds):
    return class_nms([d for d in candidates if d['confidence'] >= thresholds[d['label']]
                     and (d['parent_printer_confidence'] is None
                          or d['parent_printer_confidence'] >= thresholds['label_printer'])])


class Detector:
    def __init__(self, root=ROOT, weights=None, confidence=None, device='cpu'):
        root = Path(root)
        self.weights = Path(weights) if weights else root/'weights/best.pt'
        digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        if confidence is None:
            config = json.loads((root/'runtime-config.json').read_text())
            if digest not in [config['weights_sha256'], *config.get('compatible_weights_sha256', [])]:
                raise ValueError('Weights and calibrated config do not match')
            confidence = config['confidence_thresholds']
        elif isinstance(confidence, (int, float)):
            confidence = {name: float(confidence) for name in NAMES.values()}
        if (set(confidence) != set(NAMES.values()) or
                not all(np.isfinite(v) and .05 <= v < 1 for v in confidence.values()) or
                confidence['label_printer'] < .1):
            raise ValueError('printer threshold must be >= .1; button >= .05; both finite and < 1')
        self.thresholds = dict(confidence)
        self.device = device
        torch.set_num_threads(4)
        self.model = YOLO(str(self.weights), task='detect')
        if self.model.names != NAMES:
            raise ValueError(f'Wrong class mapping: {self.model.names}')
        self.implementation = dict(weights_sha256=digest, recognize_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def _raw(self, image):
        pred = self.model.predict(image, imgsz=640, rect=False, conf=.05, iou=.45,
                                  agnostic_nms=False, max_det=50, device=self.device, verbose=False)[0]
        return pred.boxes.data.cpu().numpy()

    def candidates(self, image):
        """No ground-truth or fixed station coordinates used here."""
        h, w = image.shape[:2]
        full = [make_detection(raw, w, h) for raw in self._raw(image)]
        full = [d for d in full if d is not None]
        printers = [d for d in full if d['class_id'] == 0]
        rois = sorted([d for d in printers if d['confidence'] >= .1], key=lambda d: d['confidence'], reverse=True)[:10]
        buttons = [d for d in full if d['class_id'] == 1 and not any(contains(crop_bounds(p['bbox_xyxy_px'], w, h), d['center_uv']) for p in rois)]
        for p in rois:
            l, t, r, b = crop_bounds(p['bbox_xyxy_px'], w, h)
            box = np.asarray(p['bbox_xyxy_px'])
            pad = np.tile((box[2:]-box[:2])*.05, 2)*[-1, -1, 1, 1]
            for raw in self._raw(image[t:b, l:r]):
                if raw[-1] != 1:
                    continue
                d = make_detection(raw, w, h, (l, t), 'predicted_printer_crop', p['confidence'])
                if d is not None and contains(box+pad, d['center_uv']):
                    buttons.append(d)
        return printers+buttons

    def predict(self, image):
        start = time.perf_counter()
        result = dict(ok=True, algorithm='task3-printer-button-yolo11n-20261010',
                      detections=[], confidence_thresholds=self.thresholds,
                      implementation=self.implementation, coordinate_frame='original_color_image_pixels',
                      coordinate_semantics='2D bounding-box centers; not physical button centers or 3D press poses',
                      button_function=None, press_pose=None, robot_motion_ready=False, reasons=[])
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                or image.shape[2] != 3 or min(image.shape[:2]) < 32):
            result.update(ok=False, reasons=['invalid_bgr_uint8_image'])
            return result
        result['image_size_wh'] = [image.shape[1], image.shape[0]]
        result['detections'] = filter_candidates(self.candidates(image), self.thresholds)
        result['objects'] = {name: [d for d in result['detections'] if d['label'] == name] for name in NAMES.values()}
        for name, ds in result['objects'].items():
            if not ds:
                result['reasons'].append(name+'_not_detected')
        result['inference_ms'] = round((time.perf_counter()-start)*1000, 2)
        return result


def annotate(image, result):
    out = image.copy()
    for d in result['detections']:
        x1, y1, x2, y2 = np.rint(d['bbox_xyxy_px']).astype(int)
        color = (255, 220, 0) if d['class_id'] == 0 else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        # Keep text clear of adjacent tiny button boxes.
        text = f"{d['label']} {d['confidence']:.2f}"
        if d['class_id'] == 0:
            cv2.putText(out, text, (max(0, x1), max(14, y1-5)), 0, .4, color, 1)
    cv2.putText(out, '2D ONLY / BUTTON FUNCTION NOT IDENTIFIED', (8, out.shape[0]-10), 0, .42, (0, 0, 220), 1)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--printer-conf', type=float)
    parser.add_argument('--button-conf', type=float)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    if (args.printer_conf is None) != (args.button_conf is None):
        parser.error('Set both per-class thresholds, or neither to use the calibrated defaults')
    for output in [args.output, args.json_output]:
        if output and output.resolve() == args.image.resolve():
            parser.error('Refusing to overwrite input image')
    im = cv2.imread(str(args.image))
    if im is None:
        parser.error('Cannot read input image')
    conf = None if args.printer_conf is None else dict(label_printer=args.printer_conf, button=args.button_conf)
    result = Detector(weights=args.weights, confidence=conf).predict(im)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(encoded)
    if args.output and not cv2.imwrite(str(args.output), annotate(im, result)):
        raise SystemExit('Cannot write preview')
    if args.json_output:
        args.json_output.write_text(encoded)
