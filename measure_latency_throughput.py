#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
from collections import defaultdict
from statistics import mean, median
import numpy as np
import csv

try:
    import psutil
except Exception:
    psutil = None

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from app.tasks import _init_local_yolo, get_ocr_reader
from app.decision_logic import init_pose, parse_triton_outputs, check_ppe, detect_torso

def get_gpu_info_fallback(index=0):
    if NVML_AVAILABLE:
        try:
            handle = nvmlDeviceGetHandleByIndex(index)
            mem = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            return {"gpu_mem_used": int(mem.used), "gpu_mem_total": int(mem.total), "gpu_util": int(util.gpu), "gpu_mem_util": int(util.memory)}
        except Exception:
            return {"gpu_mem_used": None, "gpu_mem_total": None, "gpu_util": None, "gpu_mem_util": None}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,nounits,noheader"])
        line = out.decode().strip().splitlines()[index]
        mem_used_str, util_str = [s.strip() for s in line.split(",")]
        return {"gpu_mem_used": int(mem_used_str), "gpu_mem_total": None, "gpu_util": int(util_str), "gpu_mem_util": None}
    except Exception:
        return {"gpu_mem_used": None, "gpu_mem_total": None, "gpu_util": None, "gpu_mem_util": None}

def probe_gpu_init():
    if NVML_AVAILABLE:
        try:
            nvmlInit()
            return True
        except Exception:
            return False
    try:
        subprocess.check_output(["nvidia-smi", "--help"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def list_video_files(videos_dir):
    exts = ('.mp4', '.mov', '.mkv', '.avi')
    vids = []
    for root, _, files in os.walk(videos_dir):
        for f in files:
            if f.lower().endswith(exts):
                vids.append(os.path.join(root, f))
    return sorted(vids)

def safe_read_ocr(ocr_reader, img):
    t0 = time.perf_counter()
    res = None
    try:
        res = ocr_reader.readtext(img, detail=1)
    except Exception:
        try:
            res = ocr_reader.readtext(img)
        except Exception:
            res = []
    t1 = time.perf_counter()
    return res, t1 - t0

def measure(args):
    gpu_ok = probe_gpu_init()
    if NVML_AVAILABLE and gpu_ok:
        nvmlInit()
    ly = _init_local_yolo()
    if ly is None:
        print("Local YOLO not available")
        return
    pose_inst = init_pose()
    ocr_reader = get_ocr_reader()
    videos = list_video_files(args.videos_dir)
    per_frame_rows = []
    totals = defaultdict(float)
    counts = defaultdict(int)
    frame_global_idx = 0
    gpu_samples = []
    for vpath in videos:
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            continue
        step = max(1, int(args.frame_step))
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (fidx % step) != 0:
                fidx += 1
                continue
            frame_global_idx += 1
            frame_meta = {"video": vpath, "frame_idx": fidx, "global_idx": frame_global_idx, "fps": fps}
            t_total_start = time.perf_counter()
            t_model_start = time.perf_counter()
            try:
                person_boxes = ly.predict_person_boxes(frame)
            except Exception:
                person_boxes = []
            try:
                ppe_out = ly.predict_ppe_outputs(frame)
            except Exception:
                ppe_out = {}
            t_model_end = time.perf_counter()
            model_time = t_model_end - t_model_start
            t_pose_accum = 0.0
            t_ocr_accum = 0.0
            t_decision_accum = 0.0
            boxes_by_class, _parsed_people = parse_triton_outputs(ppe_out, frame.shape[0], frame.shape[1])
            if gpu_ok:
                ginfo = get_gpu_info_fallback(args.gpu_index)
                gpu_samples.append(ginfo)
            for pi, pb in enumerate(person_boxes):
                if not pb or len(pb) < 4:
                    continue
                x1, y1, x2, y2 = int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])
                x1i, y1i, x2i, y2i = max(0, x1), max(0, y1), min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                crop = frame[y1i:y2i, x1i:x2i]
                t_pose_start = time.perf_counter()
                try:
                    res_pose = None
                    if pose_inst is not None:
                        res_pose = pose_inst.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                except Exception:
                    res_pose = None
                t_pose_end = time.perf_counter()
                t_pose = t_pose_end - t_pose_start
                t_pose_accum += t_pose
                t_ocr = 0.0
                mid = None
                if ocr_reader is not None:
                    t_ocr_start = time.perf_counter()
                    try:
                        ocr_res, ocr_t = safe_read_ocr(ocr_reader, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        t_ocr = ocr_t
                        best_txt = ""
                        best_conf = -1.0
                        for it in ocr_res:
                            if isinstance(it, (list, tuple)) and len(it) >= 2:
                                if len(it) == 3:
                                    txt = str(it[1]).strip(); conf = float(it[2])
                                else:
                                    txt = str(it[1]).strip() if len(it) > 1 else str(it[0]); conf = float(it[2]) if len(it) > 2 else 0.0
                            else:
                                txt = str(it); conf = 0.0
                            if conf > best_conf:
                                best_conf = conf; best_txt = txt
                        if best_txt:
                            mid = best_txt
                    except Exception:
                        t_ocr = time.perf_counter() - t_ocr_start
                t_ocr_accum += t_ocr
                t_decision_start = time.perf_counter()
                try:
                    violations = []
                    person_bbox = (x1i, y1i, x2i, y2i)
                    if res_pose and getattr(res_pose, "pose_landmarks", None):
                        flags = check_ppe(boxes_by_class, person_bbox, res_pose.pose_landmarks, x1i, y1i, x2i - x1i, y2i - y1i)
                        if not flags.get("helmet", False):
                            violations.append("NO HELMET")
                        else:
                            if flags.get("improper_helmet"):
                                violations.append("IMPROPER HELMET")
                        if not flags.get("vest", False):
                            violations.append("NO VEST")
                        if not flags.get("left_glove", False):
                            violations.append("NO LEFT GLOVE")
                        else:
                            if flags.get("improper_left_glove"):
                                violations.append("IMPROPER LEFT GLOVE")
                        if not flags.get("right_glove", False):
                            violations.append("NO RIGHT GLOVE")
                        else:
                            if flags.get("improper_right_glove"):
                                violations.append("IMPROPER RIGHT GLOVE")
                        if not flags.get("left_shoe", False):
                            violations.append("NO LEFT SHOE")
                        if not flags.get("right_shoe", False):
                            violations.append("NO RIGHT SHOE")
                    else:
                        vest_boxes = boxes_by_class.get(3, [])
                        helmet_boxes = boxes_by_class.get(1, [])
                        if not any((lambda a,b: a > 0.03)(0,0) for _ in []):
                            pass
                    if mid is None and ocr_reader is not None:
                        try:
                            mid_detected, txt, conf = detect_torso(ocr_reader, crop, set())
                            if mid_detected is not None:
                                mid = mid_detected
                        except Exception:
                            pass
                except Exception:
                    pass
                t_decision_end = time.perf_counter()
                t_decision = t_decision_end - t_decision_start
                t_decision_accum += t_decision
            t_total_end = time.perf_counter()
            total_time = t_total_end - t_total_start
            per_frame_rows.append({
                "global_idx": frame_global_idx,
                "video": vpath,
                "frame_idx": fidx,
                "model_time": model_time,
                "pose_time": t_pose_accum,
                "ocr_time": t_ocr_accum,
                "decision_time": t_decision_accum,
                "total_time": total_time
            })
            totals["model_time"] += model_time
            totals["pose_time"] += t_pose_accum
            totals["ocr_time"] += t_ocr_accum
            totals["decision_time"] += t_decision_accum
            totals["total_time"] += total_time
            counts["frames"] += 1
            fidx += 1
            if args.max_frames and counts["frames"] >= args.max_frames:
                break
        cap.release()
        if args.max_frames and counts["frames"] >= args.max_frames:
            break

    if NVML_AVAILABLE and gpu_ok:
        try:
            nvmlShutdown()
        except Exception:
            pass

    frames = counts["frames"]
    summary = {}
    if frames == 0:
        print("No frames processed")
        return
    summary["frames_processed"] = frames
    summary["mean_model_time"] = totals["model_time"] / frames
    summary["mean_pose_time"] = totals["pose_time"] / frames
    summary["mean_ocr_time"] = totals["ocr_time"] / frames
    summary["mean_decision_time"] = totals["decision_time"] / frames
    summary["mean_total_time"] = totals["total_time"] / frames
    summary["median_total_time"] = median([r["total_time"] for r in per_frame_rows])
    summary["fps_overall"] = frames / totals["total_time"] if totals["total_time"] > 0 else None
    summary["fps_model_only"] = frames / totals["model_time"] if totals["model_time"] > 0 else None
    summary["per_component_percent"] = {
        "model_pct": summary["mean_model_time"] / summary["mean_total_time"],
        "pose_pct": summary["mean_pose_time"] / summary["mean_total_time"],
        "ocr_pct": summary["mean_ocr_time"] / summary["mean_total_time"],
        "decision_pct": summary["mean_decision_time"] / summary["mean_total_time"]
    }
    gpu_mem_used_vals = [g.get("gpu_mem_used") for g in gpu_samples if g.get("gpu_mem_used") is not None]
    gpu_util_vals = [g.get("gpu_util") for g in gpu_samples if g.get("gpu_util") is not None]
    summary["gpu_mem_used_mean"] = int(mean(gpu_mem_used_vals)) if gpu_mem_used_vals else None
    summary["gpu_util_mean"] = float(mean(gpu_util_vals)) if gpu_util_vals else None
    out = {"summary": summary, "per_frame": per_frame_rows}
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
    if args.out_csv:
        keys = ["global_idx","video","frame_idx","model_time","pose_time","ocr_time","decision_time","total_time"]
        with open(args.out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in per_frame_rows:
                writer.writerow({k: r.get(k) for k in keys})
    print(json.dumps(summary, indent=2))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="./videos")
    ap.add_argument("--frame_step", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--gpu_index", type=int, default=0)
    ap.add_argument("--out_json", default="./latency_summary.json")
    ap.add_argument("--out_csv", default="./per_frame_timings.csv")
    args = ap.parse_args()
    measure(args)
