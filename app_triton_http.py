import os
import time
import logging
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter

log = logging.getLogger("uvicorn.error")
app = FastAPI()

INFER_REQUESTS = Counter("api_infer_requests_total", "Total inference requests")

triton = None
MODEL = os.environ.get("TRITON_MODEL_NAME", "ppe_yolo")
input_name = None
output_names = None

def sanitize_triton_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()
    if "://" in url:
        p = urlparse(url)
        return p.netloc or p.path
    return url

def create_triton_client_safe(triton_url: str):
    return InferenceServerClient(url=triton_url)

@app.on_event("startup")
def startup_event():
    global triton, input_name, output_names, MODEL
    raw_url = os.environ.get("TRITON_URL", "triton:8000")
    triton_url = sanitize_triton_url(raw_url)
    max_wait = int(os.environ.get("TRITON_WAIT_SECS", "120"))
    interval = 1.5
    deadline = time.time() + max_wait

    log.info(f"Startup: trying Triton URL '{raw_url}' -> sanitized '{triton_url}' (wait {max_wait}s)")

    while time.time() < deadline:
        try:
            client = create_triton_client_safe(triton_url)
        except Exception as e:
            log.info(f"Triton client creation failed (invalid URL or other): {e}")
            triton = None
            input_name = None
            output_names = None
            log.warning("Triton client creation failed — API will start without Triton. Fix TRITON_URL or retry.")
            return

        try:
            if client.is_server_live() and client.is_server_ready():
                triton = client
                meta = triton.get_model_metadata(MODEL)
                input_name = meta['inputs'][0]['name']
                output_names = [o['name'] for o in meta['outputs']]
                log.info(f"Connected to Triton at '{triton_url}'; model={MODEL}; inputs={input_name}")
                return
            else:
                log.info("Triton created but server not ready yet; will retry.")
                try:
                    if hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass
        except Exception as e:
            log.info(f"Triton readiness check failed: {e}")
            try:
                if hasattr(client, "close"):
                    client.close()
            except Exception:
                pass

        time.sleep(interval)
        interval = min(interval * 1.5, 10.0)

    try:
        client = create_triton_client_safe(triton_url)
        if client.is_server_live() and client.is_server_ready():
            triton = client
            meta = triton.get_model_metadata(MODEL)
            input_name = meta['inputs'][0]['name']
            output_names = [o['name'] for o in meta['outputs']]
            log.info(f"Connected to Triton on final attempt; model={MODEL}")
            return
        else:
            try:
                if hasattr(client, "close"):
                    client.close()
            except Exception:
                pass
    except Exception as e:
        log.warning(f"Final Triton client creation attempt failed: {e}")

    triton = None
    input_name = None
    output_names = None
    log.warning(f"Triton not available after {max_wait}s — API started without Triton. /infer will return 503 until Triton becomes ready.")

@app.on_event("shutdown")
def shutdown_event():
    global triton
    try:
        if triton is not None and hasattr(triton, "close"):
            triton.close()
    except Exception:
        pass

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    global triton, input_name, output_names, MODEL
    INFER_REQUESTS.inc()

    if triton is None or not triton.is_server_ready():
        raise HTTPException(status_code=503, detail="Triton server not ready")

    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    img = cv2.resize(img, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    img = np.transpose(img, (2, 0, 1))[None, ...]

    inp = InferInput(input_name, img.shape, "FP32")
    inp.set_data_from_numpy(img)
    outputs = [InferRequestedOutput(n) for n in output_names]
    res = triton.infer(MODEL, inputs=[inp], outputs=outputs)
    results = {n: res.as_numpy(n).tolist() for n in output_names}
    return {"model": MODEL, "outputs": results}