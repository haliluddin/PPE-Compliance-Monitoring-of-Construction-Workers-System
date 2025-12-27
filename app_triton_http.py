# app_triton_http.py
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_MAX_THREADS","1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU","1")
import time
import logging
import asyncio
import aiofiles
import json
import threading
import queue
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Body, Depends, Query
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.router.auth import get_current_user
import numpy as np
import cv2
import io
import csv
import base64
try:
    import redis
except Exception:
    redis = None
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
try:
    from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
except Exception:
    InferenceServerClient = None
    InferInput = None
    InferRequestedOutput = None
from app.database import SessionLocal
from app.models import Job, Camera, Violation
from app.tasks import process_image_task, process_image
from sqlalchemy.orm import Session
log = logging.getLogger("uvicorn.error")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
INFER_REQUESTS = Counter("api_infer_requests_total", "Total inference requests")
TASKS_QUEUED = Counter("api_tasks_queued_total", "Total tasks enqueued")
TRITON_MODEL = os.environ.get("TRITON_MODEL_NAME", "ppe_yolo")
HUMAN_MODEL = os.environ.get("HUMAN_MODEL_NAME", "person_yolo")
TRITON_URL = os.environ.get("TRITON_URL", "")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
WS_CHANNEL = os.environ.get("WS_CHANNEL", "ppe_results")
tmp_upload_dir = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(tmp_upload_dir, exist_ok=True)
triton = None
triton_models_meta = {}
redis_sync = None
redis_pubsub = None
ws_clients = set()
APP_LOOP = None
STREAM_THREADS = {}
STREAM_EVENTS = {}
STREAM_QUEUES = {}
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "3"))
USE_CELERY = os.environ.get("USE_CELERY", "false").lower() in ("1","true","yes")
PH_TZ = timezone(timedelta(hours=8))
from concurrent.futures import ThreadPoolExecutor
PROCESS_WORKERS = int(os.environ.get("PROCESS_WORKERS", "3"))
STREAM_QUEUE_MAXSIZE = int(os.environ.get("STREAM_QUEUE_MAXSIZE", "32"))
PROCESS_EXECUTOR = ThreadPoolExecutor(max_workers=PROCESS_WORKERS)
PROCESS_SEM = threading.BoundedSemaphore(PROCESS_WORKERS)
def try_open_capture(source, wait_seconds=8, sleep_step=0.25):
    start = time.time()
    try:
        cap = cv2.VideoCapture(source)
    except Exception:
        cap = None
    while time.time() - start < wait_seconds:
        try:
            if cap is not None and getattr(cap, "isOpened", lambda: False)():
                return cap
        except Exception:
            pass
        try:
            if cap is not None:
                ret, _ = cap.read()
                if ret:
                    return cap
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        time.sleep(sleep_step)
        try:
            cap = cv2.VideoCapture(source)
        except Exception:
            cap = None
    try:
        if cap is not None and getattr(cap, "isOpened", lambda: False)():
            return cap
    except Exception:
        pass
    return cap
def to_iso_ph(dt):
    if dt is None:
        return None
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(PH_TZ).isoformat()
    except Exception:
        try:
            return dt.isoformat()
        except Exception:
            return None
def sanitize_triton_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()
    if "://" in url:
        p = urlparse(url)
        return p.netloc or p.path
    return url
def load_model_metadata(client, model_name):
    try:
        meta = client.get_model_metadata(model_name)
        inp = meta.get('inputs', [])
        outs = meta.get('outputs', [])
        input_name = inp[0]['name'] if inp else None
        output_names = [o['name'] for o in outs] if outs else []
        return {'input_name': input_name, 'output_names': output_names}
    except Exception:
        return {'input_name': None, 'output_names': []}
@app.on_event("startup")
def startup_event():
    global triton, triton_models_meta, redis_sync, redis_pubsub
    global APP_LOOP
    raw_url = TRITON_URL
    if not raw_url:
        triton = None
        triton_models_meta = {}
    else:
        triton_url = sanitize_triton_url(raw_url)
        max_wait = int(os.environ.get("TRITON_WAIT_SECS", "120"))
        interval = 1.5
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                if InferenceServerClient is None:
                    triton = None
                    triton_models_meta = {}
                    break
                client = InferenceServerClient(url=triton_url)
            except Exception:
                triton = None
                triton_models_meta = {}
                break
            try:
                if client.is_server_live() and client.is_server_ready():
                    triton = client
                    for model in [TRITON_MODEL, HUMAN_MODEL]:
                        try:
                            triton_models_meta[model] = load_model_metadata(triton, model)
                        except Exception:
                            triton_models_meta[model] = {'input_name': None, 'output_names': []}
                    break
                try:
                    if hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass
            except Exception:
                try:
                    if hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass
            time.sleep(interval)
            interval = min(interval * 1.5, 10.0)
    try:
        if redis is not None:
            redis_sync = redis.Redis.from_url(REDIS_URL)
            redis_pubsub = redis_sync.pubsub()
            try:
                redis_pubsub.subscribe(WS_CHANNEL)
            except Exception:
                pass
        else:
            redis_sync = None
            redis_pubsub = None
    except Exception:
        redis_sync = None
        redis_pubsub = None
    APP_LOOP = asyncio.get_event_loop()
    try:
        asyncio.create_task(redis_subscriber_task())
    except Exception:
        pass
@app.on_event("shutdown")
def shutdown_event():
    global triton, redis_pubsub
    try:
        if triton is not None and hasattr(triton, "close"):
            triton.close()
    except Exception:
        pass
    try:
        if redis_pubsub is not None:
            try:
                redis_pubsub.close()
            except Exception:
                pass
    except Exception:
        pass
    for job_id, evt in list(STREAM_EVENTS.items()):
        try:
            evt.set()
        except Exception:
            pass
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            try:
                await ws.receive_text()
            except Exception:
                break
            try:
                await ws.send_text("ack")
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
@app.websocket("/ws/notifications")
async def websocket_notifications(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            try:
                await ws.receive_text()
            except Exception:
                break
            try:
                await ws.send_text("ack")
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
async def redis_subscriber_task():
    global redis_pubsub
    if redis_pubsub is None:
        return
    loop = asyncio.get_event_loop()
    while True:
        try:
            message = redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                data = message.get("data")
                if data is None:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    if isinstance(data, bytes):
                        payload = json.loads(data.decode("utf-8"))
                    else:
                        payload = json.loads(data)
                except Exception:
                    payload = {"raw": str(data)}
                coros = []
                for ws in list(ws_clients):
                    try:
                        coros.append(ws.send_json(payload))
                    except Exception:
                        pass
                if coros:
                    await asyncio.gather(*coros, return_exceptions=True)
        except Exception:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.01)
@app.post("/jobs")
def create_job(payload: dict, current_user=Depends(get_current_user)):
    sess = SessionLocal()
    try:
        job_type = payload.get("job_type", "video")
        camera_id = payload.get("camera_id")
        meta = payload.get("meta", {}) or {}
        user_id = getattr(current_user, "id", None)
        job = Job(job_type=job_type, camera_id=camera_id, status="queued", meta=meta, user_id=user_id)
        sess.add(job)
        sess.commit()
        sess.refresh(job)
        return {"job_id": job.id, "status": job.status}
    finally:
        sess.close()
def _submit_processing(img_bytes, meta):
    def _run():
        try:
            process_image(img_bytes, meta)
        finally:
            try:
                PROCESS_SEM.release()
            except Exception:
                pass
    try:
        PROCESS_EXECUTOR.submit(_run)
    except Exception:
        try:
            PROCESS_SEM.release()
        except Exception:
            pass
def process_video_file(job_id: int, filepath: str, camera_id=None):
    cap = None
    try:
        cap = try_open_capture(filepath, wait_seconds=6)
    except Exception:
        cap = None
    try:
        q = queue.Queue(maxsize=STREAM_QUEUE_MAXSIZE)
        STREAM_QUEUES[job_id] = q
        stop_event = threading.Event()
        STREAM_EVENTS[job_id] = stop_event
        def consumer():
            while not stop_event.is_set():
                try:
                    item = q.get(timeout=0.5)
                except Exception:
                    continue
                if item is None:
                    break
                img_bytes, meta = item
                try:
                    acquired = PROCESS_SEM.acquire(timeout=2.0)
                except Exception:
                    acquired = False
                if not acquired:
                    log.debug("consumer: no processing slot available, skipping submit for job %s frame_idx %s", job_id, meta.get("frame_idx"))
                    continue
                try:
                    _submit_processing(img_bytes, meta)
                except Exception:
                    try:
                        PROCESS_SEM.release()
                    except Exception:
                        pass
        consumer_thread = threading.Thread(target=consumer, daemon=True)
        consumer_thread.start()
        frame_idx = 0
        sess = SessionLocal()
        job = None
        try:
            job = sess.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "running"
                job.started_at = datetime.now(timezone.utc)
                sess.commit()
        except Exception:
            sess.rollback()
        consecutive_no_frame = 0
        max_no_frame_before_stop = 120
        while True:
            try:
                if cap is None or not getattr(cap, "isOpened", lambda: False)():
                    consecutive_no_frame += 1
                    if consecutive_no_frame > max_no_frame_before_stop:
                        log.warning("process_video_file: no frames from capture, stopping job %s", job_id)
                        break
                    time.sleep(0.02)
                    frame_idx += 1
                    continue
                ret, frame = cap.read()
                if not ret:
                    consecutive_no_frame += 1
                    if consecutive_no_frame > max_no_frame_before_stop:
                        log.warning("process_video_file: consecutive read failures, stopping job %s", job_id)
                        break
                    time.sleep(0.02)
                    frame_idx += 1
                    continue
                consecutive_no_frame = 0
                if FRAME_SKIP <= 1 or (frame_idx % FRAME_SKIP) == 0:
                    small_w = 640
                    h, w = frame.shape[:2]
                    if w > small_w:
                        scale = small_w / float(w)
                        frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    else:
                        frame_small = frame
                    try:
                        _, jpg = cv2.imencode('.jpg', frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                        img_bytes = jpg.tobytes()
                    except Exception:
                        frame_idx += 1
                        continue
                    try:
                        job_draw_labels = True
                        if job is not None:
                            jm = job.meta or {}
                            job_draw_labels = bool(jm.get("draw_labels", True))
                    except Exception:
                        job_draw_labels = True
                    meta = {"job_id": job_id, "camera_id": camera_id, "frame_idx": frame_idx, "ts": time.time(), "draw_labels": job_draw_labels}
                    try:
                        q.put((img_bytes, meta), timeout=1.0)
                    except Exception:
                        log.warning("process_video_file: dropped frame (queue full) for job %s frame_idx %s", job_id, frame_idx)
                frame_idx += 1
            except Exception:
                log.exception("frame loop error in process_video_file")
                time.sleep(0.2)
                frame_idx += 1
                continue
        stop_event.set()
        try:
            q.put_nowait(None)
        except Exception:
            pass
        consumer_thread.join(timeout=60.0)
        wait_start = time.time()
        wait_timeout = int(os.environ.get("PROCESS_FINISH_WAIT_SECS", "30"))
        needed = PROCESS_WORKERS
        while time.time() - wait_start < wait_timeout:
            acquired_list = []
            all_acquired = True
            for _ in range(needed):
                try:
                    if PROCESS_SEM.acquire(blocking=False):
                        acquired_list.append(True)
                    else:
                        all_acquired = False
                        break
                except Exception:
                    all_acquired = False
                    break
            if all_acquired:
                for _ in acquired_list:
                    try:
                        PROCESS_SEM.release()
                    except Exception:
                        pass
                break
            for _ in acquired_list:
                try:
                    PROCESS_SEM.release()
                except Exception:
                    pass
            time.sleep(0.5)
        if job:
            try:
                job.status = "completed"
                job.finished_at = datetime.now(timezone.utc)
                sess.commit()
            except Exception:
                sess.rollback()
    except Exception:
        log.exception("process_video_file top-level error")
        try:
            sess2 = SessionLocal()
            try:
                job = sess2.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = "error"
                    job.finished_at = datetime.now(timezone.utc)
                    sess2.commit()
            except Exception:
                sess2.rollback()
            finally:
                sess2.close()
        except Exception:
            pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        STREAM_QUEUES.pop(job_id, None)
        STREAM_EVENTS.pop(job_id, None)
        try:
            sess.close()
        except Exception:
            pass
@app.post("/jobs/{job_id}/upload")
async def upload_job_video(job_id: int, file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    INFER_REQUESTS.inc()
    sess = SessionLocal()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        camera_id = job.camera_id
    finally:
        sess.close()
    filename = f"job_{job_id}_{int(time.time())}_{file.filename}"
    filepath = os.path.join(tmp_upload_dir, filename)
    try:
        async with aiofiles.open(filepath, "wb") as out_f:
            chunk_size = 1024 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                await out_f.write(chunk)
    except Exception as e:
        try:
            os.remove(filepath)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"failed to save upload: {str(e)}")
    TASKS_QUEUED.inc()
    if background_tasks is not None:
        background_tasks.add_task(process_video_file, job_id, filepath, camera_id)
    else:
        thread = threading.Thread(target=process_video_file, args=(job_id, filepath, camera_id), daemon=True)
        STREAM_THREADS[job_id] = thread
        thread.start()
    return {"status": "accepted", "job_id": job_id}
class StreamStart(BaseModel):
    stream_url: str
    camera_id: int = None
    job_id: int = None
    draw_labels: bool = True
def stream_loop(job_id: int, rtsp_url: str, camera_id=None, stop_event: threading.Event = None):
    cap = None
    try:
        cap = try_open_capture(rtsp_url, wait_seconds=8)
    except Exception:
        cap = None
    try:
        if cap is not None:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        q = queue.Queue(maxsize=STREAM_QUEUE_MAXSIZE)
        STREAM_QUEUES[job_id] = q
        if stop_event is None:
            stop_event = threading.Event()
            STREAM_EVENTS[job_id] = stop_event
        frame_idx = 0
        sess = SessionLocal()
        job = None
        def consumer():
            while not stop_event.is_set():
                try:
                    item = q.get(timeout=0.5)
                except Exception:
                    continue
                if item is None:
                    break
                img_bytes, meta = item
                try:
                    acquired = PROCESS_SEM.acquire(timeout=2.0)
                except Exception:
                    acquired = False
                if not acquired:
                    log.debug("stream consumer: no processing slot, skipping submit for job %s frame_idx %s", job_id, meta.get("frame_idx"))
                    continue
                try:
                    _submit_processing(img_bytes, meta)
                except Exception:
                    try:
                        PROCESS_SEM.release()
                    except Exception:
                        pass
        consumer_thread = threading.Thread(target=consumer, daemon=True)
        consumer_thread.start()
        try:
            if job_id:
                job = sess.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = "running"
                    job.started_at = datetime.now(timezone.utc)
                    sess.commit()
        except Exception:
            sess.rollback()
        consecutive_no_frame = 0
        max_no_frame_before_stop = 120
        while True:
            try:
                if stop_event is not None and stop_event.is_set():
                    break
                if cap is None or not getattr(cap, "isOpened", lambda: False)():
                    consecutive_no_frame += 1
                    if consecutive_no_frame > max_no_frame_before_stop:
                        log.warning("stream_loop: capture not opened for stream %s (job %s); stopping", rtsp_url, job_id)
                        break
                    time.sleep(1.0)
                    frame_idx += 1
                    continue
                ret, frame = cap.read()
                if not ret:
                    consecutive_no_frame += 1
                    if consecutive_no_frame > max_no_frame_before_stop:
                        log.warning("stream_loop: consecutive read failures for stream %s (job %s); stopping", rtsp_url, job_id)
                        break
                    time.sleep(0.05)
                    frame_idx += 1
                    continue
                consecutive_no_frame = 0
                if FRAME_SKIP <= 1 or (frame_idx % FRAME_SKIP) == 0:
                    small_w = 640
                    h, w = frame.shape[:2]
                    if w > small_w:
                        scale = small_w / float(w)
                        frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    else:
                        frame_small = frame
                    try:
                        _, jpg = cv2.imencode('.jpg', frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                        img_bytes = jpg.tobytes()
                    except Exception:
                        frame_idx += 1
                        continue
                    try:
                        job_draw_labels = True
                        if job is not None:
                            jm = job.meta or {}
                            job_draw_labels = bool(jm.get("draw_labels", True))
                    except Exception:
                        job_draw_labels = True
                    meta = {"job_id": job_id, "camera_id": camera_id, "frame_idx": frame_idx, "ts": time.time(), "draw_labels": job_draw_labels}
                    try:
                        q.put((img_bytes, meta), timeout=1.0)
                    except Exception:
                        log.warning("stream_loop: dropped frame (queue full) for stream job %s frame_idx %s", job_id, frame_idx)
                frame_idx += 1
            except Exception:
                log.exception("frame loop error in stream_loop")
                time.sleep(0.5)
                frame_idx += 1
                continue
        stop_event.set()
        try:
            q.put_nowait(None)
        except Exception:
            pass
        consumer_thread.join(timeout=60.0)
        wait_start = time.time()
        wait_timeout = int(os.environ.get("PROCESS_FINISH_WAIT_SECS", "30"))
        needed = PROCESS_WORKERS
        while time.time() - wait_start < wait_timeout:
            acquired_list = []
            all_acquired = True
            for _ in range(needed):
                try:
                    if PROCESS_SEM.acquire(blocking=False):
                        acquired_list.append(True)
                    else:
                        all_acquired = False
                        break
                except Exception:
                    all_acquired = False
                    break
            if all_acquired:
                for _ in acquired_list:
                    try:
                        PROCESS_SEM.release()
                    except Exception:
                        pass
                break
            for _ in acquired_list:
                try:
                    PROCESS_SEM.release()
                except Exception:
                    pass
            time.sleep(0.5)
        if job:
            job.status = "completed"
            job.finished_at = datetime.now(timezone.utc)
            sess.commit()
    except Exception:
        log.exception("stream_loop top-level error")
        try:
            sess2 = SessionLocal()
            try:
                job = sess2.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = "error"
                    job.finished_at = datetime.now(timezone.utc)
                    sess2.commit()
            except Exception:
                sess2.rollback()
            finally:
                sess2.close()
        except Exception:
            pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        STREAM_QUEUES.pop(job_id, None)
        STREAM_EVENTS.pop(job_id, None)
        try:
            sess.close()
        except Exception:
            pass
@app.put("/violations/{violation_id}/status")
def update_violation_status(violation_id: int, payload: dict = Body(...), current_user=Depends(get_current_user)):
    sess = SessionLocal()
    try:
        v = sess.query(Violation).filter(Violation.id == violation_id).first()
        if not v:
            raise HTTPException(status_code=404, detail="violation not found")
        new_status = payload.get("status")
        if new_status is None:
            raise HTTPException(status_code=400, detail="status required")
        try:
            prev_status = (v.status or "").lower()
            new_status_l = str(new_status).strip().lower()
            if new_status_l not in ("pending", "resolved", "false positive"):
                raise HTTPException(status_code=400, detail="Invalid status")
            if prev_status != new_status_l:
                v.manually_changed = True
                try:
                    v.changed_by = getattr(current_user, "id", None)
                except Exception:
                    v.changed_by = None
                try:
                    v.changed_at = datetime.now(timezone.utc)
                except Exception:
                    v.changed_at = None
            v.status = new_status_l
            if new_status_l == "resolved":
                try:
                    v.resolved_at = datetime.now(timezone.utc)
                except Exception:
                    pass
            sess.commit()
            sess.refresh(v)
        except Exception:
            sess.rollback()
            raise
        notif_payload = {"type": "status_update", "violation_id": v.id, "status": v.status, "created_at": to_iso_ph(datetime.now(timezone.utc))}
        try:
            if redis_sync is not None:
                try:
                    redis_sync.publish(WS_CHANNEL, json.dumps(notif_payload, default=str))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            loop = APP_LOOP
            if loop:
                for ws in list(ws_clients):
                    try:
                        asyncio.run_coroutine_threadsafe(ws.send_json(notif_payload), loop)
                    except Exception:
                        pass
        except Exception:
            pass
        return {"id": v.id, "status": v.status}
    finally:
        sess.close()
