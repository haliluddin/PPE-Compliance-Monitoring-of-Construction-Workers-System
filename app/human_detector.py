# human_detector.py
import os
from ultralytics import YOLO
import numpy as np

YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "cpu")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "/app/yolov8n.pt")

class LocalYOLO:
    def __init__(self, model_path=None):
        path = model_path or YOLO_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"YOLO model not found at {path}")
        self.model = YOLO(path)

    def predict_person_boxes(self, image):
        try:
            results = self.model.predict(source=image, device=YOLO_DEVICE, imgsz=640, conf=0.25, verbose=False)
        except Exception:
            return []
        out = []
        if not results:
            return out
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            return out
        try:
            xyxy = boxes.xyxy.cpu().numpy(); confs = boxes.conf.cpu().numpy(); cls = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            xyxy = np.array(boxes.xyxy); confs = np.array(boxes.conf); cls = np.array(boxes.cls).astype(int)
        for (x1,y1,x2,y2),c,cl in zip(xyxy, confs, cls):
            if int(cl) != 0:
                continue
            out.append((float(x1),float(y1),float(x2),float(y2), float(c)))
        return out

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        try:
            _detector = LocalYOLO()
        except Exception:
            _detector = None
    return _detector
