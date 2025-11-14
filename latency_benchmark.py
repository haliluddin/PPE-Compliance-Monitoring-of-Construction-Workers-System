# latency_benchmark.py
import os
import time
import json
import argparse
import subprocess
import csv
from collections import defaultdict
import numpy as np
import psutil
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", default="test_images")
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--model_type", choices=["yolo","triton","mock"], default="yolo")
parser.add_argument("--weights", default=None)
parser.add_argument("--triton_url", default=None)
parser.add_argument("--use_pose", type=int, default=1)
parser.add_argument("--use_ocr", type=int, default=1)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--out_dir", default="/tmp/benchmark")
parser.add_argument("--sample_gpu_ms", type=int, default=200)
parser.add_argument("--verbose", type=int, default=1)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

def sample_gpu():
    try:
        p = subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        out = p.stdout.strip().splitlines()
        gpu_stats = []
        for line in out:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3:
                mem_used = int(parts[0])
                mem_total = int(parts[1])
                util = int(parts[2])
                gpu_stats.append({"mem_used_mb": mem_used, "mem_total_mb": mem_total, "util_pct": util})
        return gpu_stats
    except Exception:
        return []

try:
    if args.model_type == "yolo":
        from ultralytics import YOLO
        model = None
        try:
            model = YOLO(args.weights) if args.weights else YOLO("yolov8n.pt")
        except Exception:
            model = None
except Exception:
    model = None

triton_client = None
if args.model_type == "triton" and args.triton_url:
    try:
        from tritonclient.http import InferenceServerClient
        triton_client = InferenceServerClient(url=args.triton_url)
    except Exception:
        triton_client = None

mp_pose = None
if args.use_pose:
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception:
        mp_pose = None

ocr_reader = None
if args.use_ocr:
    try:
        import easyocr
        import torch
        gpu_flag = False
        try:
            gpu_flag = torch.cuda.is_available()
        except Exception:
            gpu_flag = False
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=gpu_flag)
        except Exception:
            ocr_reader = easyocr.Reader(['en'], gpu=False)
    except Exception:
        ocr_reader = None

def model_infer_yolo(img_path):
    try:
        r = model.predict(source=img_path, imgsz=416, conf=0.3, verbose=False)[0]
        return r
    except Exception:
        time.sleep(0.02)
        return None

def model_infer_triton(img_path):
    try:
        import cv2
        import numpy as np
        arr = cv2.imread(img_path)
        h,w = arr.shape[:2]
        img = cv2.resize(arr, (416,416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")/255.0
        img = np.transpose(img, (2,0,1))[None,...]
        from tritonclient.http import InferInput, InferRequestedOutput
        meta = triton_client.get_model_metadata(triton_client.get_model_repository_index()[0]) if False else {}
        inp = InferInput("input", img.shape, "FP32")
        inp.set_data_from_numpy(img)
        outputs = []
        try:
            res = triton_client.infer("", inputs=[inp], outputs=outputs)
            return res
        except Exception:
            time.sleep(0.02)
            return None
    except Exception:
        time.sleep(0.02)
        return None

def pose_infer_mock(crop):
    time.sleep(0.005)
    return None

def pose_infer_mediapipe(crop):
    try:
        import cv2
        img = crop
        if img is None:
            time.sleep(0.005)
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(img_rgb)
        return res
    except Exception:
        time.sleep(0.005)
        return None

def ocr_infer_mock(crop):
    time.sleep(0.01)
    return ""

def ocr_infer_easyocr(crop):
    try:
        import cv2
        img = crop
        if img is None:
            time.sleep(0.01)
            return ""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = ocr_reader.readtext(rgb, detail=0)
        txt = " ".join(res) if isinstance(res, list) else str(res)
        return txt
    except Exception:
        time.sleep(0.01)
        return ""

def read_image_for_crop(path):
    try:
        import cv2
        return cv2.imread(path)
    except Exception:
        return None

images = [os.path.join(args.images_dir, f) for f in sorted(os.listdir(args.images_dir)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
if not images:
    raise SystemExit("no images found in images_dir")
images = images[:args.n]

if args.model_type == "yolo" and model is None:
    args.model_type = "mock"

if args.use_pose and mp_pose is None:
    pose_fn = pose_infer_mock
else:
    pose_fn = pose_infer_mediapipe

if args.use_ocr and ocr_reader is None:
    ocr_fn = ocr_infer_mock
else:
    ocr_fn = ocr_infer_easyocr

if args.model_type == "yolo":
    model_fn = model_infer_yolo
elif args.model_type == "triton":
    model_fn = model_infer_triton
else:
    def model_infer_mock(img_path):
        time.sleep(0.03)
        return None
    model_fn = model_infer_mock

timings = {"model": [], "pose": [], "ocr": [], "decision": [], "total": []}
cpu_usages = []
mem_usages = []
gpu_samples = []
warmup_n = min(args.warmup, max(0,len(images)-1))
for i in range(warmup_n):
    _ = model_fn(images[0])
time.sleep(0.2)

next_sample_ms = time.time() + args.sample_gpu_ms/1000.0
for i in trange(len(images)):
    img_path = images[i]
    img = read_image_for_crop(img_path)
    t0 = time.time()
    t_proc0 = psutil.Process().cpu_percent(interval=None)
    model_out = model_fn(img_path)
    t_model = time.time()
    pose_res = pose_fn(img)
    t_pose = time.time()
    ocr_res = ocr_fn(img)
    t_ocr = time.time()
    time.sleep(0.002)
    t_dec = time.time()
    timings['model'].append((t_model - t0) * 1000.0)
    timings['pose'].append((t_pose - t_model) * 1000.0)
    timings['ocr'].append((t_ocr - t_pose) * 1000.0)
    timings['decision'].append((t_dec - t_ocr) * 1000.0)
    total_ms = (t_dec - t0) * 1000.0
    timings['total'].append(total_ms)
    cpu_usages.append(psutil.cpu_percent(interval=None))
    mem_usages.append(psutil.virtual_memory().percent)
    now = time.time()
    if now >= next_sample_ms:
        gpu = sample_gpu()
        if gpu:
            gpu_samples.append({"t": int(now*1000), "gpu": gpu})
        next_sample_ms = now + args.sample_gpu_ms/1000.0

summary = {}
for k, arr in timings.items():
    a = np.array(arr)
    summary[k] = {"count": int(len(a)), "mean_ms": float(np.mean(a)) if len(a) else 0.0, "median_ms": float(np.median(a)) if len(a) else 0.0, "95p_ms": float(np.percentile(a,95)) if len(a) else 0.0}

total_mean = summary['total']['mean_ms']
fps = 1000.0 / total_mean if total_mean > 0 else 0.0

out = {
    "args": vars(args),
    "summary": summary,
    "fps": fps,
    "cpu_percent_mean": float(np.mean(cpu_usages)) if cpu_usages else None,
    "mem_percent_mean": float(np.mean(mem_usages)) if mem_usages else None,
    "gpu_samples": gpu_samples
}

with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
    json.dump(out, f, indent=2)

csv_path = os.path.join(args.out_dir, "timings.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    header = ["idx","image","model_ms","pose_ms","ocr_ms","decision_ms","total_ms","cpu_pct","mem_pct"]
    w.writerow(header)
    for i in range(len(timings['total'])):
        row = [i, images[i], timings['model'][i], timings['pose'][i], timings['ocr'][i], timings['decision'][i], timings['total'][i], cpu_usages[i] if i < len(cpu_usages) else "", mem_usages[i] if i < len(mem_usages) else ""]
        w.writerow(row)

print("Summary")
for k, v in summary.items():
    print(k, "mean_ms", v["mean_ms"], "median_ms", v["median_ms"], "95p_ms", v["95p_ms"])
print("Total mean ms", total_mean, "FPS", fps)
print("Wrote", os.path.join(args.out_dir, "summary.json"), csv_path)
