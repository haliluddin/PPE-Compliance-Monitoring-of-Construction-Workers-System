import os
import time
import logging
import asyncio
import json
import threading
from urllib.parse import urlparse
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import redis
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from app.database import SessionLocal
from app.models import Job, Camera, Violation
from app.tasks import process_image_task

log = logging.getLogger("uvicorn.error")
app = FastAPI()

cors_env = os.environ.get("CORS_ALLOW_ORIGINS", "")
if cors_env:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://5lm18p50eufoh4-5173.proxy.runpod.net",
        "https://5lm18p50eufoh4-9000.proxy.runpod.net"
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from app.router.auth import router as auth_router
    app.include_router(auth_router)
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
TRITON_URL = os.environ.get("TRITON_URL", "triton:8000")
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
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "3"))
USE_CELERY = os.environ.get("USE_CELERY", "true").lower() in ("1","true","yes")

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
    triton_url = sanitize_triton_url(raw_url)
    max_wait = int(os.environ.get("TRITON_WAIT_SECS", "120"))
    interval = 1.5
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
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
        redis_sync = redis.Redis.from_url(REDIS_URL)
        redis_pubsub = redis_sync.pubsub()
        redis_pubsub.subscribe(WS_CHANNEL)
    except Exception:
        redis_sync = None
        redis_pubsub = None
    APP_LOOP = asyncio.get_event_loop()
    asyncio.create_task(redis_subscriber_task())

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
                coros = [ws.send_json(payload) for ws in list(ws_clients)]
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

def process_video_file(job_id: int, filepath: str, camera_id=None):
    cap = cv2.VideoCapture(filepath)
    frame_idx = 0
    sess = SessionLocal()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "running"
            job.started_at = datetime.utcnow()
            sess.commit()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if FRAME_SKIP <= 1 or (frame_idx % FRAME_SKIP) == 0:
                _, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                img_bytes = jpg.tobytes()
                meta = {"job_id": job_id, "camera_id": camera_id, "frame_idx": frame_idx, "ts": time.time()}
                if USE_CELERY:
                    process_image_task.delay(img_bytes, meta)
                else:
                    try:
                        process_image_task.run(None, img_bytes, meta)
                    except Exception:
                        process_image_task.delay(img_bytes, meta)
            frame_idx += 1
        if job:
            job.status = "completed"
            job.finished_at = datetime.utcnow()
            sess.commit()
    finally:
        try:
            cap.release()
        except Exception:
            pass
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
        thread.start()
        return {"status": "accepted", "job_id": job_id}

class StreamStart(BaseModel):
    rtsp_url: str
    camera_id: int = None
    job_id: int = None

def stream_loop(job_id: int, rtsp_url: str, camera_id=None, stop_event: threading.Event = None):
    cap = cv2.VideoCapture(rtsp_url)
    frame_idx = 0
    sess = SessionLocal()
    try:
        job = None
        if job_id:
            job = sess.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "running"
                job.started_at = datetime.utcnow()
                sess.commit()
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            _, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            img_bytes = jpg.tobytes()
            meta = {"job_id": job_id, "camera_id": camera_id, "frame_idx": frame_idx, "ts": time.time()}
            if USE_CELERY:
                process_image_task.delay(img_bytes, meta)
            else:
                try:
                    process_image_task.run(None, img_bytes, meta)
                except Exception:
                    process_image_task.delay(img_bytes, meta)
            frame_idx += 1
        if job:
            job.status = "completed"
            job.finished_at = datetime.utcnow()
            sess.commit()
    finally:
        try:
            cap.release()
        except Exception:
            pass
        sess.close()

@app.post("/streams")
def start_stream(payload: StreamStart):
    rtsp_url = payload.rtsp_url
    camera_id = payload.camera_id
    job_id = payload.job_id
    if job_id is None:
        sess = SessionLocal()
        try:
            job = Job(job_type="stream", camera_id=camera_id, status="queued", meta={"rtsp_url": rtsp_url})
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
    sess = SessionLocal()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "stopped"
            job.finished_at = datetime.utcnow()
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
        task = process_image_task.delay(data, meta)
        TASKS_QUEUED.inc()
        return {"task_id": task.id, "status": "queued"}
    else:
        process_image_task.run(None, data, meta)
        return {"task_id": None, "status": "processed"}

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
        return {"job_id": job.id, "status": job.status, "meta": job.meta, "created_at": job.created_at, "started_at": job.started_at, "finished_at": job.finished_at}
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
            out.append({
                "id": r.id,
                "job_id": r.job_id,
                "camera_id": r.camera_id,
                "worker_code": r.worker_code,
                "violation_types": r.violation_types,
                "frame_index": r.frame_index,
                "created_at": r.created_at
            })
        return {"violations": out}
    finally:
        sess.close()

@app.get("/health")
def health():
    ready = True
    try:
        if triton is None:
            ready = False
    except Exception:
        ready = False
    return {"triton_ready": ready}
