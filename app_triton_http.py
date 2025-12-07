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
import json
import threading
import queue
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Body
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
log = logging.getLogger("uvicorn.error")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    from app.router.auth import router as auth_router
    app.include_router(auth_router)
except Exception:
    pass
try:
    from app.router.notifications_ws import router as notifications_ws_router
    app.include_router(notifications_ws_router)
except Exception:
    pass
try:
    from app.router.notifications import router as notifications_router
    app.include_router(notifications_router)
except Exception:
    pass
try:
    from app.router.reports import router as reports_router
    app.include_router(reports_router)
except Exception:
    pass
try:
    from app.router.violations import router as violations_router
    app.include_router(violations_router)
except Exception:
    pass
try:
    from app.router.workers import router as workers_router
    app.include_router(workers_router)
except Exception:
    pass
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
PROCESS_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("PROCESS_WORKERS","3")))
PROCESS_SEM = threading.BoundedSemaphore(int(os.environ.get("PROCESS_SEM","3")))
try:
    from app.router.auth import router as auth_router_dup
except Exception:
    pass
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
def create_job(payload: dict):
    sess = SessionLocal()
    try:
        job_type = payload.get("job_type", "video")
        camera_id = payload.get("camera_id")
        meta = payload.get("meta", {})
        job = Job(job_type=job_type, camera_id=camera_id, status="queued", meta=meta)
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
    cap = cv2.VideoCapture(filepath)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    q = queue.Queue(maxsize=2)
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
            acquired = PROCESS_SEM.acquire(blocking=False)
            if not acquired:
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
        while True:
            ret, frame = cap.read()
            if not ret:
                break
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
                meta = {"job_id": job_id, "camera_id": camera_id, "frame_idx": frame_idx, "ts": time.time()}
                try:
                    if q.full():
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                    q.put_nowait((img_bytes, meta))
                except Exception:
                    pass
            frame_idx += 1
        stop_event.set()
        try:
            q.put_nowait(None)
        except Exception:
            pass
        consumer_thread.join(timeout=1.0)
        if job:
            job.status = "completed"
            job.finished_at = datetime.now(timezone.utc)
            sess.commit()
    except Exception:
        try:
            if job:
                job.status = "error"
                job.finished_at = datetime.now(timezone.utc)
                sess.commit()
        except Exception:
            pass
    finally:
        try:
            cap.release()
        except Exception:
            pass
        STREAM_QUEUES.pop(job_id, None)
        STREAM_EVENTS.pop(job_id, None)
        sess.close()
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
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)
    if background_tasks is not None:
        background_tasks.add_task(process_video_file, job_id, filepath, camera_id)
        return {"status": "accepted", "job_id": job_id}
    else:
        thread = threading.Thread(target=process_video_file, args=(job_id, filepath, camera_id), daemon=True)
        STREAM_THREADS[job_id] = thread
        thread.start()
        return {"status": "accepted", "job_id": job_id}
class StreamStart(BaseModel):
    stream_url: str
    camera_id: int = None
    job_id: int = None
def stream_loop(job_id: int, rtsp_url: str, camera_id=None, stop_event: threading.Event = None):
    cap = cv2.VideoCapture(rtsp_url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    q = queue.Queue(maxsize=2)
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
            acquired = PROCESS_SEM.acquire(blocking=False)
            if not acquired:
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
        job = None
        if job_id:
            job = sess.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "running"
                job.started_at = datetime.now(timezone.utc)
                sess.commit()
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
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
                meta = {"job_id": job_id, "camera_id": camera_id, "frame_idx": frame_idx, "ts": time.time()}
                try:
                    if q.full():
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                    q.put_nowait((img_bytes, meta))
                except Exception:
                    pass
            frame_idx += 1
        stop_event.set()
        try:
            q.put_nowait(None)
        except Exception:
            pass
        consumer_thread.join(timeout=1.0)
        if job:
            job.status = "completed"
            job.finished_at = datetime.now(timezone.utc)
            sess.commit()
    finally:
        try:
            cap.release()
        except Exception:
            pass
        STREAM_QUEUES.pop(job_id, None)
        STREAM_EVENTS.pop(job_id, None)
        sess.close()
@app.post("/streams")
def start_stream(payload: StreamStart):
    rtsp_url = payload.stream_url
    camera_id = payload.camera_id
    job_id = payload.job_id
    if job_id is None:
        sess = SessionLocal()
        try:
            job = Job(job_type="stream", camera_id=camera_id, status="queued", meta={"stream_url": rtsp_url})
            sess.add(job)
            sess.commit()
            sess.refresh(job)
            job_id = job.id
        finally:
            sess.close()
    stop_event = threading.Event()
    STREAM_EVENTS[job_id] = stop_event
    thread = threading.Thread(target=stream_loop, args=(job_id, rtsp_url, camera_id, stop_event), daemon=True)
    STREAM_THREADS[job_id] = thread
    thread.start()
    return {"job_id": job_id, "status": "started"}
@app.post("/streams/{job_id}/stop")
def stop_stream(job_id: int):
    evt = STREAM_EVENTS.get(job_id)
    if not evt:
        raise HTTPException(status_code=404, detail="stream not found")
    evt.set()
    thread = STREAM_THREADS.get(job_id)
    if thread and thread.is_alive():
        thread.join(timeout=2.0)
    STREAM_EVENTS.pop(job_id, None)
    STREAM_THREADS.pop(job_id, None)
    STREAM_QUEUES.pop(job_id, None)
    sess = SessionLocal()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "stopped"
            job.finished_at = datetime.now(timezone.utc)
            sess.commit()
    finally:
        sess.close()
    return {"job_id": job_id, "status": "stopped"}
@app.post("/infer")
async def infer(file: UploadFile = File(...), camera_id: int = None, job_id: int = None):
    INFER_REQUESTS.inc()
    data = await file.read()
    if data is None or len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    meta = {"camera_id": camera_id, "job_id": job_id, "ts": time.time()}
    if USE_CELERY:
        try:
            b64 = base64.b64encode(data).decode("ascii")
            task = process_image_task.delay(b64, meta)
            TASKS_QUEUED.inc()
            return {"task_id": task.id, "status": "queued"}
        except Exception:
            try:
                res = process_image(data, meta)
                return {"task_id": None, "status": "processed", "result": res}
            except Exception:
                raise HTTPException(status_code=500, detail="processing failed")
    else:
        try:
            res = process_image(data, meta)
            return {"task_id": None, "status": "processed", "result": res}
        except Exception:
            try:
                b64 = base64.b64encode(data).decode("ascii")
                task = process_image_task.delay(b64, meta)
                TASKS_QUEUED.inc()
                return {"task_id": task.id, "status": "queued"}
            except Exception:
                raise HTTPException(status_code=500, detail="processing failed")
@app.post("/infer_sync")
async def infer_sync(file: UploadFile = File(...)):
    INFER_REQUESTS.inc()
    global triton, TRITON_MODEL
    if triton is None:
        raise HTTPException(status_code=503, detail="Triton server not ready")
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    img = cv2.resize(img, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    img = np.transpose(img, (2, 0, 1))[None, ...]
    meta = triton_models_meta.get(TRITON_MODEL, {})
    input_name = meta.get('input_name')
    output_names = meta.get('output_names', [])
    if not input_name or not output_names:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    inp = InferInput(input_name, img.shape, "FP32")
    inp.set_data_from_numpy(img)
    outputs = [InferRequestedOutput(n) for n in output_names]
    res = triton.infer(TRITON_MODEL, inputs=[inp], outputs=outputs)
    results = {n: res.as_numpy(n).tolist() for n in output_names}
    return {"model": TRITON_MODEL, "outputs": results}
@app.get("/jobs/{job_id}/status")
def job_status(job_id: int):
    sess = SessionLocal()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "job_id": job.id,
            "status": job.status,
            "meta": job.meta,
            "created_at": to_iso_ph(getattr(job, "created_at", None)),
            "started_at": to_iso_ph(getattr(job, "started_at", None)),
            "finished_at": to_iso_ph(getattr(job, "finished_at", None))
        }
    finally:
        sess.close()
@app.get("/violations")
def list_violations(job_id: int = None, limit: int = 50, offset: int = 0):
    sess = SessionLocal()
    try:
        q = sess.query(Violation)
        if job_id is not None:
            q = q.filter(Violation.job_id == job_id)
        rows = q.order_by(Violation.id.desc()).offset(offset).limit(limit).all()
        out = []
        for r in rows:
            worker_name = None
            try:
                if getattr(r, "worker", None):
                    worker_obj = r.worker
                    worker_name = getattr(worker_obj, "fullName", None) or getattr(worker_obj, "name", None)
            except Exception:
                worker_name = None
            if not worker_name:
                worker_name = getattr(r, "worker_name", None) or getattr(r, "worker", None) or getattr(r, "worker_code", None) or "Unknown Worker"
            camera_name = "Video Upload" if getattr(r, "camera_id", None) is None else "Unknown Camera"
            camera_location = "Video Upload" if getattr(r, "camera_id", None) is None else "Unknown Camera"
            try:
                cam = getattr(r, "camera", None)
                if cam:
                    camera_name = getattr(cam, "name", f"Camera {getattr(cam, 'id', '')}")
                    camera_location = getattr(cam, "location", "") or camera_name
            except Exception:
                pass
            snapshot_b64 = None
            try:
                snap = getattr(r, "snapshot", None)
                if snap:
                    if isinstance(snap, (bytes, bytearray)):
                        snapshot_b64 = base64.b64encode(snap).decode("ascii")
                    elif isinstance(snap, str):
                        snapshot_b64 = snap
            except Exception:
                snapshot_b64 = None
            out.append({
                "id": r.id,
                "job_id": r.job_id,
                "camera_id": r.camera_id,
                "worker_code": r.worker_code,
                "worker": worker_name,
                "violation_types": r.violation_types,
                "violation": r.violation_types,
                "frame_index": r.frame_index,
                "created_at": to_iso_ph(getattr(r, "created_at", None)),
                "status": getattr(r, "status", "Pending"),
                "camera": camera_name,
                "camera_location": camera_location,
                "snapshot": snapshot_b64
            })
        return out
    finally:
        sess.close()
@app.put("/violations/{violation_id}/status")
def update_violation_status(violation_id: int, payload: dict = Body(...)):
    sess = SessionLocal()
    try:
        v = sess.query(Violation).filter(Violation.id == violation_id).first()
        if not v:
            raise HTTPException(status_code=404, detail="violation not found")
        new_status = payload.get("status")
        if new_status is None:
            raise HTTPException(status_code=400, detail="status required")
        try:
            v.status = new_status
            if new_status.lower() == "resolved":
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
@app.get("/health")
def health():
    if os.environ.get("ALLOW_NO_TRITON","1") in ("1","true","True"):
        return {"triton_ready": True, "note": "Triton not present; local processing allowed"}
    ready = triton is not None
    return {"triton_ready": ready}
@app.get("/notifications")
def get_notifications(limit: int = 100, offset: int = 0):
    sess = SessionLocal()
    try:
        rows = sess.query(Violation).order_by(Violation.id.desc()).offset(offset).limit(limit).all()
        out = []
        for r in rows:
            worker_name = None
            try:
                if getattr(r, "worker", None):
                    worker_obj = r.worker
                    worker_name = getattr(worker_obj, "fullName", None) or getattr(worker_obj, "name", None)
            except Exception:
                worker_name = None
            if not worker_name:
                worker_name = getattr(r, "worker_name", None) or getattr(r, "worker", None) or getattr(r, "worker_code", None) or "Unknown Worker"
            camera_name = "Video Upload" if getattr(r, "camera_id", None) is None else "Unknown Camera"
            camera_location = "Video Upload" if getattr(r, "camera_id", None) is None else "Unknown Camera"
            try:
                cam = getattr(r, "camera", None)
                if cam:
                    camera_name = getattr(cam, "name", f"Camera {getattr(cam, 'id', '')}")
                    camera_location = getattr(cam, "location", "") or camera_name
            except Exception:
                pass
            snapshot_b64 = None
            try:
                snap = getattr(r, "snapshot", None)
                if snap:
                    if isinstance(snap, (bytes, bytearray)):
                        snapshot_b64 = base64.b64encode(snap).decode("ascii")
                    elif isinstance(snap, str):
                        snapshot_b64 = snap
            except Exception:
                snapshot_b64 = None
            out.append({
                "id": r.id,
                "violation_id": r.id,
                "worker_name": worker_name,
                "worker_code": getattr(r, "worker_code", None) or "N/A",
                "violation_type": getattr(r, "violation_types", None) or "Unknown Violation",
                "violation": getattr(r, "violation_types", None) or "Unknown Violation",
                "camera": camera_name,
                "camera_location": camera_location,
                "created_at": to_iso_ph(getattr(r, "created_at", None)),
                "is_read": getattr(r, "is_read", False),
                "status": getattr(r, "status", "Pending"),
                "type": "worker_violation",
                "snapshot": snapshot_b64
            })
        return out
    finally:
        sess.close()
@app.post("/notifications/{notif_id}/mark_read")
def mark_notification_read(notif_id: int):
    sess = SessionLocal()
    try:
        v = sess.query(Violation).filter(Violation.id == notif_id).first()
        if not v:
            raise HTTPException(status_code=404, detail="notification not found")
        if hasattr(v, "is_read"):
            v.is_read = True
            sess.commit()
        return {"ok": True}
    finally:
        sess.close()
@app.get("/cameras")
def list_cameras():
    sess = SessionLocal()
    try:
        cams = sess.query(Camera).all() if 'Camera' in globals() else []
        out = []
        for c in cams:
            out.append({"id": c.id, "name": getattr(c, "name", f"Camera {c.id}"), "location": getattr(c, "location", "")})
        return out
    finally:
        sess.close()
@app.post("/cameras")
def create_camera(payload: dict = Body(...)):
    sess = SessionLocal()
    try:
        name = payload.get("name") or ""
        location = payload.get("location") or ""
        stream_url = payload.get("stream_url") or ""
        cam = Camera()
        try:
            setattr(cam, "name", name)
        except Exception:
            pass
        try:
            setattr(cam, "location", location)
        except Exception:
            pass
        try:
            setattr(cam, "stream_url", stream_url)
        except Exception:
            pass
        sess.add(cam)
        sess.commit()
        sess.refresh(cam)
        return {"camera_id": getattr(cam, "id", None), "id": getattr(cam, "id", None), "name": getattr(cam, "name", None), "location": getattr(cam, "location", None)}
    finally:
        sess.close()
def _period_bounds(period: str):
    now_ph = datetime.now(PH_TZ)
    today_start_ph = now_ph.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "last_week":
        end_ph = today_start_ph
        start_ph = end_ph - timedelta(days=7)
        days = 7
    elif period == "last_month":
        end_ph = today_start_ph
        start_ph = end_ph - timedelta(days=30)
        days = 30
    else:
        start_ph = today_start_ph
        end_ph = start_ph + timedelta(days=1)
        days = 1

    start_utc = start_ph.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_ph.astimezone(timezone.utc).replace(tzinfo=None)
    return start_ph, start_utc, end_utc, days
@app.get("/reports")
def quick_reports(period: str = "today"):
    sess = SessionLocal()
    try:
        start_ph, start_utc, end_utc, days = _period_bounds(period)
        rows = sess.query(Violation).filter(Violation.created_at >= start_utc, Violation.created_at < end_utc).all()
        total_incidents = len(rows)
        counts = {}
        camera_counts = {}
        type_counts = {}
        resolved = 0
        for v in rows:
            worker_display = None
            try:
                if getattr(v, "worker", None):
                    worker_display = getattr(v.worker, "fullName", None) or getattr(v.worker, "name", None)
            except Exception:
                worker_display = None
            if not worker_display:
                worker_display = getattr(v, "worker_name", None) or getattr(v, "worker", None) or getattr(v, "worker_code", "UNKNOWN")
            counts[worker_display] = counts.get(worker_display, 0) + 1
            cam = getattr(v, "camera", None)
            if cam:
                cam_key = getattr(cam, "name", f"Camera {getattr(cam, 'id', '')}")
            else:
                cam_key = "Video Upload" if getattr(v, "camera_id", None) is None else "Unknown"
            camera_counts[cam_key] = camera_counts.get(cam_key, 0) + 1
            vt = getattr(v, "violation_types", None) or ""
            if vt:
                try:
                    parts = [p.strip() for p in str(vt).split(",") if p.strip()]
                    for p in parts:
                        type_counts[p] = type_counts.get(p, 0) + 1
                except Exception:
                    type_counts[str(vt)] = type_counts.get(str(vt), 0) + 1
            else:
                type_counts["Unknown"] = type_counts.get("Unknown", 0) + 1
            if getattr(v, "status", "").lower() == "resolved":
                resolved += 1
        top_offenders = [{"name": k, "value": v} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
        camera_data = [{"location": k, "violations": v, "risk": "High" if v > 5 else "Medium" if v > 2 else "Low"} for k, v in camera_counts.items()]
        most_violations = [{"name": k, "violations": v} for k, v in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)]
        worker_data = []
        for i, (k, v) in enumerate(sorted(counts.items(), key=lambda x: x[1], reverse=True)):
            resolved_for_worker = 0
            resolution_rate = 0
            worker_data.append({"rank": i+1, "name": k, "violations": v, "resolved": resolved_for_worker, "resolution_rate": resolution_rate})
        violation_resolution_rate = round((resolved / total_incidents * 100) if total_incidents > 0 else 0, 2)
        return {
            "total_incidents": total_incidents,
            "total_workers_involved": len(counts),
            "violation_resolution_rate": violation_resolution_rate,
            "high_risk_locations": sum(1 for d in camera_data if d["risk"] == "High"),
            "most_violations": most_violations,
            "top_offenders": top_offenders,
            "camera_data": camera_data,
            "worker_data": worker_data
        }
    finally:
        sess.close()
@app.get("/reports/performance")
def reports_performance(period: str = "today"):
    sess = SessionLocal()
    try:
        start_ph, start_utc, end_utc, days = _period_bounds(period)
        rows = sess.query(Violation).filter(Violation.created_at >= start_utc, Violation.created_at < end_utc).all()
        date_buckets = {}
        for i in range(days):
            d = (start_ph + timedelta(days=i)).strftime("%Y-%m-%d")
            date_buckets[d] = {"violations": 0, "compliance": 0}
        response_times = []
        for v in rows:
            dt = getattr(v, "created_at", None)
            if not dt:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            date_key = dt.astimezone(PH_TZ).strftime("%Y-%m-%d")
            if date_key not in date_buckets:
                date_buckets[date_key] = {"violations": 0, "compliance": 0}
            date_buckets[date_key]["violations"] += 1
            if getattr(v, "status", "").lower() == "resolved":
                date_buckets[date_key]["compliance"] += 1
            rt = getattr(v, "response_time", None)
            if isinstance(rt, (int, float)):
                response_times.append(float(rt))
            else:
                resolved_at = getattr(v, "resolved_at", None)
                created_at = getattr(v, "created_at", None)
                try:
                    if resolved_at is not None and created_at is not None:
                        if resolved_at.tzinfo is None:
                            resolved_at = resolved_at.replace(tzinfo=timezone.utc)
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        delta_min = (resolved_at - created_at).total_seconds() / 60.0
                        if delta_min >= 0:
                            response_times.append(delta_min)
                except Exception:
                    pass
        performance_over_time = [{"date": k, "violations": v["violations"], "compliance": v["compliance"]} for k, v in sorted(date_buckets.items())]
        average_response_time = round(sum(response_times) / len(response_times), 2) if response_times else 0
        return {"performance_over_time": performance_over_time, "average_response_time": average_response_time}
    finally:
        sess.close()
@app.get("/reports/export")
def reports_export(period: str = "today"):
    sess = SessionLocal()
    try:
        start_ph, start_utc, end_utc, days = _period_bounds(period)
        rows = sess.query(Violation).filter(Violation.created_at >= start_utc, Violation.created_at < end_utc).all()
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(["id", "worker_code", "worker_name", "violation_types", "camera", "camera_location", "created_at", "status"])
        for v in rows:
            worker_name = ""
            try:
                if getattr(v, "worker", None):
                    worker_name = getattr(v.worker, "fullName", None) or getattr(v.worker, "name", None)
            except Exception:
                worker_name = ""
            if not worker_name:
                worker_name = getattr(v, "worker_name", None) or getattr(v, "worker", None) or ""
            cam_name = "Video Upload" if getattr(v, "camera_id", None) is None else ""
            cam_loc = "Video Upload" if getattr(v, "camera_id", None) is None else ""
            try:
                cam = getattr(v, "camera", None)
                if cam:
                    cam_name = getattr(cam, "name", f"Camera {getattr(cam, 'id', '')}")
                    cam_loc = getattr(cam, "location", "") or cam_name
            except Exception:
                pass
            created_at_str = to_iso_ph(getattr(v, "created_at", None))
            cw.writerow([
                getattr(v, "id", ""),
                getattr(v, "worker_code", ""),
                worker_name,
                getattr(v, "violation_types", "") or getattr(v, "violation_type", ""),
                cam_name,
                cam_loc,
                created_at_str,
                getattr(v, "status", "Pending")
            ])
        output = si.getvalue()
        nowstr = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        headers = {"Content-Disposition": f'attachment; filename="report_{period}_{nowstr}.csv"'}
        return Response(content=output, media_type="text/csv", headers=headers)
    finally:
        sess.close()
