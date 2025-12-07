import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
import base64 as _b64
import base64
import json
import time
import cv2
import numpy as np
import logging
from sqlalchemy import or_
from datetime import datetime, timezone, timedelta
from threading import Lock
import threading
from urllib.parse import urlparse
from app.decision_logic import process_frame, init_pose
from app.database import SessionLocal
from app.models import Violation, Job, Worker, Notification
try:
    from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
except Exception:
    InferenceServerClient = None
try:
    import redis
except Exception:
    redis = None
try:
    from .celery_app import celery
except Exception:
    celery = None
from app.tracker import get_global_tracker
log = logging.getLogger(__name__)
TRITON_URL = os.environ.get("TRITON_URL", "")
TRITON_MODEL = os.environ.get("TRITON_MODEL_NAME", "ppe_yolo")
HUMAN_MODEL = os.environ.get("HUMAN_MODEL_NAME", "person_yolo")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
WS_CHANNEL = os.environ.get("WS_CHANNEL", "ppe_results")
USE_LOCAL_YOLO = os.environ.get("USE_LOCAL_YOLO", "false").lower() in ("1", "true", "yes")
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "5"))
LOCAL_PERSON_MODEL = os.environ.get("LOCAL_PERSON_MODEL", "/workspace/ppe-monitor/yolov8n.pt")
LOCAL_PPE_MODEL = os.environ.get("LOCAL_PPE_MODEL", "/workspace/ppe-monitor/best.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.35"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))
OCR_GPU = os.environ.get("OCR_GPU", "false").lower() in ("1", "true", "yes")
_TRITON_CLIENT = None
_REDIS_CLIENT = None
local_yolo = None
_local_yolo_lock = Lock()
_OCR_READER = None
_OCR_LOCK = Lock()
PH_TZ = timezone(timedelta(hours=8))
_POSE_INSTANCE = None
_POSE_LOCK = Lock()
def sanitize_triton_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()
    if "://" in url:
        p = urlparse(url)
        return p.netloc or p.path
    return url
def get_triton_client():
    global _TRITON_CLIENT
    if _TRITON_CLIENT is not None:
        return _TRITON_CLIENT
    if not TRITON_URL:
        _TRITON_CLIENT = None
        return None
    if InferenceServerClient is None:
        _TRITON_CLIENT = None
        return None
    try:
        triton_url_sanitized = sanitize_triton_url(TRITON_URL)
        _TRITON_CLIENT = InferenceServerClient(url=triton_url_sanitized)
    except Exception:
        _TRITON_CLIENT = None
    return _TRITON_CLIENT
def get_redis():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if redis is None:
        _REDIS_CLIENT = None
        return None
    try:
        _REDIS_CLIENT = redis.Redis.from_url(REDIS_URL)
    except Exception:
        _REDIS_CLIENT = None
    return _REDIS_CLIENT
def get_ocr_reader():
    global _OCR_READER
    with _OCR_LOCK:
        if _OCR_READER is not None:
            return _OCR_READER
        try:
            import easyocr
            import torch
            gpu_flag = False
            try:
                gpu_flag = torch.cuda.is_available()
            except Exception:
                gpu_flag = False
            try:
                _OCR_READER = easyocr.Reader(['en'], gpu=gpu_flag)
            except Exception:
                _OCR_READER = easyocr.Reader(['en'], gpu=False)
        except Exception:
            _OCR_READER = None
        return _OCR_READER
def _init_local_yolo():
    global local_yolo
    with _local_yolo_lock:
        if local_yolo is not None:
            return local_yolo
        try:
            import torch
            from ultralytics import YOLO
            person_model = None
            ppe_model = None
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                person_model = YOLO(LOCAL_PERSON_MODEL)
                try:
                    if device != "cpu":
                        person_model.to(device)
                except Exception:
                    pass
            except Exception:
                try:
                    person_model = YOLO("yolov8n.pt")
                    try:
                        if device != "cpu":
                            person_model.to(device)
                    except Exception:
                        pass
                except Exception:
                    person_model = None
            try:
                ppe_model = YOLO(LOCAL_PPE_MODEL)
                try:
                    if device != "cpu":
                        ppe_model.to(device)
                except Exception:
                    pass
            except Exception:
                ppe_model = None
            class _LocalYOLOWrapper:
                def __init__(self, person_model, ppe_model, conf=YOLO_CONF, iou=YOLO_IOU):
                    self.person_model = person_model
                    self.ppe_model = ppe_model
                    self.conf = float(conf)
                    self.iou = float(iou)
                def predict_person_boxes(self, frame):
                    if self.person_model is None:
                        return []
                    try:
                        results = self.person_model.predict(source=frame, conf=self.conf, iou=self.iou, classes=[0], verbose=False)
                        if not results:
                            return []
                        r = results[0]
                        boxes_attr = getattr(r, "boxes", None)
                        if boxes_attr is None or len(boxes_attr) == 0:
                            return []
                        try:
                            xyxy = boxes_attr.xyxy.cpu().numpy()
                            confs = boxes_attr.conf.cpu().numpy()
                        except Exception:
                            xyxy = np.array(boxes_attr.xyxy)
                            confs = np.array(boxes_attr.conf)
                        out = []
                        for (x1, y1, x2, y2), c in zip(xyxy, confs):
                            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
                        return out
                    except Exception:
                        return []
                def predict_ppe_outputs(self, frame):
                    if self.ppe_model is None:
                        return {}
                    try:
                        results = self.ppe_model.predict(source=frame, conf=self.conf, iou=self.iou, classes=[0,1,2,3], verbose=False)
                        if not results:
                            return {}
                        r = results[0]
                        boxes_attr = getattr(r, "boxes", None)
                        if boxes_attr is None or len(boxes_attr) == 0:
                            return {}
                        try:
                            xyxy = boxes_attr.xyxy.cpu().numpy()
                            confs = boxes_attr.conf.cpu().numpy()
                            cls = boxes_attr.cls.cpu().numpy().astype(int)
                        except Exception:
                            xyxy = np.array(boxes_attr.xyxy)
                            confs = np.array(boxes_attr.conf)
                            try:
                                cls = np.array(boxes_attr.cls).astype(int)
                            except Exception:
                                cls = np.zeros((len(confs),), dtype=int)
                        arr = []
                        for (x1,y1,x2,y2), c, cl in zip(xyxy, confs, cls):
                            arr.append([float(x1), float(y1), float(x2), float(y2), float(c), int(cl)])
                        if not arr:
                            return {}
                        return {"ppe_boxes": np.array(arr)}
                    except Exception:
                        return {}
            local_yolo = _LocalYOLOWrapper(person_model, ppe_model)
        except Exception:
            local_yolo = None
    return local_yolo
def _parse_person_boxes_from_triton_outputs(outputs, H, W):
    if outputs is None:
        return []
    arrays = []
    for v in outputs.values():
        if v is None:
            continue
        a = np.array(v)
        if a.ndim == 2 and a.shape[0] > 0:
            arrays.append(a)
    if not arrays:
        return []
    arr = arrays[0]
    boxes = []
    if arr.ndim == 2 and arr.shape[1] >= 6:
        if arr.shape[1] == 6:
            for row in arr:
                x1,y1,x2,y2,conf,cls = row[:6]
                cls = int(cls)
                if max(x1, x2, y1, y2) <= 1.01:
                    x1c = float(x1) * W
                    y1c = float(y1) * H
                    x2c = float(x2) * W
                    y2c = float(y2) * H
                else:
                    x1c = float(x1)
                    y1c = float(y1)
                    x2c = float(x2)
                    y2c = float(y2)
                if cls == 0:
                    boxes.append((max(0.0, x1c), max(0.0, y1c), min(W-1.0, x2c), min(H-1.0, y2c), float(conf)))
        else:
            for row in arr:
                x,y,w,h,conf = float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4])
                probs = row[5:]
                if probs.size>0:
                    cls = int(np.argmax(probs))
                    cls_conf = float(probs[cls])
                else:
                    cls = int(row[-1]) if row.size>5 else -1
                    cls_conf = 1.0
                final_conf = float(conf*cls_conf)
                if max(x,y,w,h) <= 1.01:
                    cx = x * W
                    cy = y * H
                    ww = w * W
                    hh = h * H
                else:
                    cx = x
                    cy = y
                    ww = w
                    hh = h
                x1 = cx - ww/2
                y1 = cy - hh/2
                x2 = cx + ww/2
                y2 = cy + hh/2
                x1c = max(0.0, x1)
                y1c = max(0.0, y1)
                x2c = min(W-1.0, x2)
                y2c = min(H-1.0, y2)
                if cls == 0:
                    boxes.append((x1c, y1c, x2c, y2c, final_conf))
    return boxes
def get_pose_instance():
    global _POSE_INSTANCE
    with _POSE_LOCK:
        if _POSE_INSTANCE is None:
            try:
                _POSE_INSTANCE = init_pose()
            except Exception:
                _POSE_INSTANCE = None
        return _POSE_INSTANCE
def _process_image(image_bytes, meta=None):
    sess = None
    try:
        if isinstance(image_bytes, str):
            try:
                image_bytes = _b64.b64decode(image_bytes)
            except Exception:
                pass
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        triton = get_triton_client()
        pose = get_pose_instance()
        person_boxes = None
        triton_out_for_ppe = None
        meta = meta or {}
        raw_ts = meta.get("ts")
        frame_ts_dt = None
        if raw_ts is not None:
            try:
                frame_ts_float = float(raw_ts)
                frame_ts_dt = datetime.fromtimestamp(frame_ts_float, tz=timezone.utc)
            except Exception:
                frame_ts_dt = datetime.now(timezone.utc)
        else:
            frame_ts_dt = None
        if triton is not None:
            try:
                human_meta = triton.get_model_metadata(HUMAN_MODEL)
                human_inputs = human_meta.get('inputs', [])
                human_outputs = human_meta.get('outputs', [])
                human_input_name = human_inputs[0]['name'] if human_inputs else None
                human_output_names = [o['name'] for o in human_outputs] if human_outputs else []
                if human_input_name and human_output_names:
                    img_human = cv2.resize(frame, (320, 320))
                    img_human = cv2.cvtColor(img_human, cv2.COLOR_BGR2RGB).astype("float32")/255.0
                    img_human = np.transpose(img_human, (2, 0, 1))[None,...]
                    inp = InferInput(human_input_name, img_human.shape, "FP32")
                    inp.set_data_from_numpy(img_human)
                    outputs = [InferRequestedOutput(n) for n in human_output_names]
                    res = triton.infer(HUMAN_MODEL, inputs=[inp], outputs=outputs)
                    triton_out = {n: res.as_numpy(n) for n in human_output_names}
                    parsed_person_boxes = _parse_person_boxes_from_triton_outputs(triton_out, frame.shape[0], frame.shape[1])
                    if parsed_person_boxes:
                        person_boxes = parsed_person_boxes
            except Exception:
                person_boxes = None
            try:
                ppe_meta = triton.get_model_metadata(TRITON_MODEL)
                ppe_meta_inputs = ppe_meta.get('inputs', [])
                ppe_meta_outputs = ppe_meta.get('outputs', [])
            except Exception:
                ppe_meta_inputs = []
                ppe_meta_outputs = []
            if ppe_meta_outputs:
                try:
                    ppe_input_name = ppe_meta_inputs[0]['name'] if ppe_meta_inputs else None
                    img_ppe = cv2.resize(frame, (320, 320))
                    img_ppe = cv2.cvtColor(img_ppe, cv2.COLOR_BGR2RGB).astype("float32")/255.0
                    img_ppe = np.transpose(img_ppe, (2,0,1))[None,...]
                    if ppe_input_name:
                        inp2 = InferInput(ppe_input_name, img_ppe.shape, "FP32")
                        inp2.set_data_from_numpy(img_ppe)
                        outputs2 = [InferRequestedOutput(n) for n in ppe_meta_outputs]
                        res2 = triton.infer(TRITON_MODEL, inputs=[inp2], outputs=outputs2)
                        triton_out_for_ppe = {n: res2.as_numpy(n) for n in ppe_meta_outputs}
                except Exception:
                    triton_out_for_ppe = None
        if (person_boxes is None or len(person_boxes) == 0) and USE_LOCAL_YOLO:
            ly = _init_local_yolo()
            if ly is not None:
                try:
                    person_boxes = ly.predict_person_boxes(frame)
                except Exception:
                    person_boxes = None
                try:
                    triton_out_for_ppe = ly.predict_ppe_outputs(frame)
                except Exception:
                    triton_out_for_ppe = None
        if person_boxes is None:
            person_boxes = []
        try:
            ocr_reader = get_ocr_reader()
        except Exception:
            ocr_reader = None
        result = process_frame(frame, triton_client=triton, triton_model_name=TRITON_MODEL, ocr_reader=ocr_reader, regset=set(), pose_instance=pose, person_boxes=person_boxes, triton_outputs=triton_out_for_ppe)
        annotated = None
        if isinstance(result, dict) and "annotated_bgr" in result:
            annotated = result.pop("annotated_bgr")
        job_id = meta.get("job_id")
        camera_id = meta.get("camera_id")
        frame_idx = meta.get("frame_idx")
        sess = SessionLocal()
        job = None
        if job_id is not None:
            job = sess.query(Job).filter(Job.id == job_id).first()
            if job and getattr(job, "status", None) == "queued":
                job.status = "running"
                job.started_at = datetime.now(PH_TZ)
                sess.commit()
        people = result.get("people", [])
        r = get_redis()
        publish_people = []
        for p in people:
            id_label = p.get("id")
            display_name = None
            worker_code = None
            if id_label is not None and id_label != "UNID":
                if isinstance(id_label, dict):
                    worker_code = id_label.get("worker_code") or id_label.get("worker") or id_label.get("code") or None
                    display_name = id_label.get("fullName") or id_label.get("full_name") or id_label.get("name") or None
                else:
                    s = str(id_label)
                    if s.startswith("UNREG:"):
                        worker_code = s.split(":", 1)[1]
                    else:
                        worker_code = s
                    try:
                        if worker_code:
                            worker_obj = sess.query(Worker).filter(Worker.worker_code == worker_code).first()
                            if worker_obj:
                                display_name = getattr(worker_obj, "fullName", None) or getattr(worker_obj, "name", None)
                    except Exception:
                        worker_obj = None
            id_for_publish = display_name if display_name else (worker_code if worker_code else (str(id_label) if id_label is not None else None))
            publish_people.append({"bbox": p.get("bbox"), "id": id_for_publish, "violations": p.get("violations", [])})
        annotated_b64 = None
        annotated_bytes = None
        if annotated is not None:
            try:
                should_publish_image = any((p.get("violations") or []) for p in people)
                if should_publish_image:
                    h, w = annotated.shape[:2]
                    target_w = 640
                    if w > target_w:
                        scale = target_w / float(w)
                        annotated_small = cv2.resize(annotated, (int(w*scale), int(h*scale)))
                    else:
                        annotated_small = annotated
                    _, jpgall = cv2.imencode('.jpg', annotated_small, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    annotated_b64 = base64.b64encode(jpgall.tobytes()).decode("ascii")
                    annotated_bytes = jpgall.tobytes()
                else:
                    annotated_b64 = None
                    _, jpgall = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    annotated_bytes = jpgall.tobytes()
            except Exception:
                annotated_b64 = None
                annotated_bytes = None
        payload = {"meta": meta, "people": publish_people, "boxes_by_class": result.get("boxes_by_class", {}), "annotated_jpeg_b64": annotated_b64, "timestamp": time.time()}
        if r:
            try:
                r.publish(WS_CHANNEL, json.dumps(payload, default=str))
            except Exception:
                r = None
        if not r:
            try:
                import app_triton_http
                import asyncio as _asyncio
                loop = getattr(app_triton_http, "APP_LOOP", None)
                if loop:
                    futures = []
                    for ws in list(app_triton_http.ws_clients):
                        futures.append(_asyncio.run_coroutine_threadsafe(ws.send_json(payload), loop))
                    for f in futures:
                        try:
                            f.result(timeout=1.0)
                        except Exception:
                            pass
            except Exception:
                pass
        tracker = get_global_tracker()
        try:
            tracker.update_tracks(people, frame_idx or 0, frame_ts_dt if frame_ts_dt is not None else datetime.now(timezone.utc), annotated_bytes, sess, job_id=job_id, camera_id=camera_id, job_user_id=getattr(job, "user_id", None) if job is not None else None)
        except Exception:
            sess.rollback()
        return {"status": "ok", "meta": meta, "result_summary": {"people": len(people)}}
    except Exception:
        log.exception("processing image failed")
        raise
    finally:
        try:
            if sess:
                sess.close()
        except Exception:
            pass
if celery:
    @celery.task(bind=True)
    def process_image_task(self, image_bytes, meta=None):
        return _process_image(image_bytes, meta)
else:
    def process_image_task(*args, **kwargs):
        return _process_image(*args, **kwargs)
def process_image(image_bytes, meta=None):
    return _process_image(image_bytes, meta)
