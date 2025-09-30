import time
import statistics
import cv2
import torch
from ultralytics import YOLO

VIDEO_PATH = "./4271760-hd_1920_1080_30fps.mp4"   
MODEL_PATH = "./runs/segment/ppe_yolov8n_seg_run1/weights/best.pt"  
WARMUP = 20
NUM_FRAMES = 500  

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = YOLO(MODEL_PATH)
model.to(device)
model.predictor = None  

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video frames:", frame_count)

e2e_times = []
inference_times = []
frame_idx = 0

def run_inference_only(img):
    start = time.perf_counter()
    results = model(img, device=device, conf=0.25, verbose=False)  
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start), results

print("Warming up for", WARMUP, "frames...")
for _ in range(WARMUP):
    ret, img = cap.read()
    if not ret: break
    _t, _ = run_inference_only(img)

print("Benchmarking...")
while True:
    if NUM_FRAMES and frame_idx >= NUM_FRAMES:
        break
    ret, img = cap.read()
    if not ret:
        break
    frame_idx += 1

    t0 = time.perf_counter()
    inf_time, results = run_inference_only(img)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    inference_times.append(inf_time)
    e2e_times.append(t1 - t0)

cap.release()

def summarize(name, times):
    times_ms = [t * 1000.0 for t in times]
    return {
        "count": len(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "p95_ms": sorted(times_ms)[int(0.95 * len(times_ms)) - 1],
        "fps": 1000.0 / statistics.mean(times_ms) if statistics.mean(times_ms) > 0 else float("inf")
    }

print("\nInference-only summary:")
inf_summary = summarize("inference", inference_times)
for k, v in inf_summary.items():
    print(f" {k}: {v}")

print("\nEnd-to-end summary (includes any light per-frame overhead):")
e2e_summary = summarize("e2e", e2e_times)
for k, v in e2e_summary.items():
    print(f" {k}: {v}")

import csv
with open("yolo_speed_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_idx","inference_ms","e2e_ms"])
    for i, (inf, e2e) in enumerate(zip(inference_times, e2e_times)):
        writer.writerow([i, inf*1000.0, e2e*1000.0])

print("\nDone — results saved to yolo_speed_results.csv")
