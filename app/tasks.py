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
from app.db import SessionLocal
from app.models import Violation, Job, Worker
from datetime import datetime

TRITON_URL = os.environ.get("TRITON_URL", "triton:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL_NAME", "ppe_yolo")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
WS_CHANNEL = os.environ.get("WS_CHANNEL", "ppe_results")

def sanitize_triton_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()
    if "://" in url:
        p = urlparse(url)
        return p.netloc or p.path
    return url

@celery.task(bind=True)
def process_image_task(self, image_bytes, meta=None):
    sess = None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        triton_url_sanitized = sanitize_triton_url(TRITON_URL)
        triton = InferenceServerClient(url=triton_url_sanitized)
        pose = init_pose()
        result = process_frame(frame, triton_client=triton, triton_model_name=TRITON_MODEL, ocr_reader=None, regset=set(), pose_instance=pose)
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
            if job and job.status == "queued":
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
            if id_label is not None and id_label != "UNID":
                if str(id_label).startswith("UNREG:"):
                    worker_code = str(id_label).split(":",1)[1]
                else:
                    worker_code = str(id_label)
                w = sess.query(Worker).filter(Worker.worker_code == worker_code).first()
                if w is None:
                    w = Worker(worker_code=worker_code, registered=False)
                    sess.add(w)
                    sess.commit()
                    sess.refresh(w)
                worker_id = w.id
            if violations:
                snap_bytes = None
                try:
                    if annotated is not None:
                        bbox = p.get("bbox")
                        if bbox and len(bbox) == 4:
                            x1,y1,x2,y2 = map(int, bbox)
                            x1 = max(0,x1); y1 = max(0,y1); x2 = min(frame.shape[1]-1,x2); y2 = min(frame.shape[0]-1,y2)
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
