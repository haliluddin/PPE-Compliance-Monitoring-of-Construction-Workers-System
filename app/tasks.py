# app/tasks.py
from .celery_app import celery
import base64
import json
import time
import cv2
import numpy as np
import os
from urllib.parse import urlparse
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from .decision_logic import process_frame, init_pose
import redis
from app.database import SessionLocal
from app.models import Violation, Job, Worker
from datetime import datetime

TRITON_URL = os.environ.get("TRITON_URL", "triton:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL_NAME", "ppe_yolo")
HUMAN_MODEL = os.environ.get("HUMAN_MODEL_NAME", "person_yolo")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
WS_CHANNEL = os.environ.get("WS_CHANNEL", "ppe_results")
USE_LOCAL_YOLO = os.environ.get("USE_LOCAL_YOLO", "false").lower() in ("1", "true", "yes")
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "1"))

local_yolo = None

def sanitize_triton_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()
    if "://" in url:
        p = urlparse(url)
        return p.netloc or p.path
    return url

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

@celery.task(bind=True)
def process_image_task(self, image_bytes, meta=None):
    sess = None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        triton = None
        try:
            triton_url_sanitized = sanitize_triton_url(TRITON_URL)
            triton = InferenceServerClient(url=triton_url_sanitized)
        except Exception:
            triton = None
        pose = init_pose()
        person_boxes = None
        if triton is not None:
            try:
                human_meta = triton.get_model_metadata(HUMAN_MODEL)
                human_inputs = human_meta.get('inputs', [])
                human_outputs = human_meta.get('outputs', [])
                human_input_name = human_inputs[0]['name'] if human_inputs else None
                human_output_names = [o['name'] for o in human_outputs] if human_outputs else []
                if human_input_name and human_output_names:
                    img_human = cv2.resize(frame, (416, 416))
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
        if person_boxes is None and USE_LOCAL_YOLO and local_yolo is not None:
            try:
                person_boxes = local_yolo.predict_person_boxes(frame)
            except Exception:
                person_boxes = None
        result = process_frame(frame, triton_client=triton, triton_model_name=TRITON_MODEL, ocr_reader=None, regset=set(), pose_instance=pose, person_boxes=person_boxes)
        annotated = None
        if isinstance(result, dict) and "annotated_bgr" in result:
            annotated = result.pop("annotated_bgr")
        meta = meta or {}
        job_id = meta.get("job_id")
        camera_id = meta.get("camera_id")
        frame_idx = meta.get("frame_idx")
        frame_ts = meta.get("ts")
        sess = SessionLocal()
        if job_id is not None:
            job = sess.query(Job).filter(Job.id == job_id).first()
            if job and getattr(job, "status", None) == "queued":
                job.status = "running"
                job.started_at = datetime.utcnow()
                sess.commit()
        people = result.get("people", [])
        r = redis.Redis.from_url(REDIS_URL)
        publish_people = []
        for p in people:
            violations = p.get("violations", [])
            id_label = p.get("id")
            worker_code = None
            worker_id = None
            worker_obj = None
            if id_label is not None and id_label != "UNID":
                if str(id_label).startswith("UNREG:"):
                    worker_code = str(id_label).split(":", 1)[1]
                else:
                    worker_code = str(id_label)
                worker_obj = sess.query(Worker).filter(Worker.worker_code == worker_code).first()
                if worker_obj is not None:
                    worker_id = worker_obj.id
            if violations and worker_obj is not None:
                snap_bytes = None
                try:
                    if annotated is not None:
                        bbox = p.get("bbox")
                        if bbox and len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                            x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
                            crop = annotated[y1:y2, x1:x2]
                            if crop is None or crop.size == 0:
                                _, jpg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                                snap_bytes = jpg.tobytes()
                            else:
                                _, jpg = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                                snap_bytes = jpg.tobytes()
                        else:
                            _, jpg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            snap_bytes = jpg.tobytes()
                except Exception:
                    try:
                        _, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        snap_bytes = jpg.tobytes()
                    except Exception:
                        snap_bytes = None
                inference_json = {"person": p, "boxes_by_class": result.get("boxes_by_class", {})}
                should_save_violation = bool(worker_obj and getattr(worker_obj, "registered", True))
                if should_save_violation:
                    v = Violation(job_id=job_id, camera_id=camera_id, worker_id=worker_id, worker_code=worker_code, violation_types=";".join(violations), frame_index=frame_idx, frame_ts=datetime.fromtimestamp(frame_ts) if frame_ts else None, snapshot=snap_bytes, inference=inference_json, created_at=datetime.utcnow())
                    sess.add(v)
                    sess.commit()
                    sess.refresh(v)
            publish_people.append({"bbox": p.get("bbox"), "id": id_label, "violations": violations})
        annotated_b64 = None
        if annotated is not None:
            try:
                _, jpgall = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                annotated_b64 = base64.b64encode(jpgall.tobytes()).decode("ascii")
            except Exception:
                annotated_b64 = None
        payload = {"meta": meta, "people": publish_people, "boxes_by_class": result.get("boxes_by_class", {}), "annotated_jpeg_b64": annotated_b64, "timestamp": time.time()}
        try:
            r.publish(WS_CHANNEL, json.dumps(payload, default=str))
        except Exception:
            pass
        return {"status": "ok", "meta": meta, "result_summary": {"people": len(people)}}
    except Exception as e:
        raise
    finally:
        try:
            if sess:
                sess.close()
        except Exception:
            pass
