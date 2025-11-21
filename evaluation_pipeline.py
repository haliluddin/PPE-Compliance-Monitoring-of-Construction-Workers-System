#!/usr/bin/env python3
import os
import sys
import json
import cv2
import math
import argparse
import numpy as np
import pandas as pd
import time
import re
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from app.tasks import _init_local_yolo, get_ocr_reader
from app.decision_logic import process_frame, init_pose

def levenshtein(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]

def cer(pred: str, gt: str) -> float:
    if gt is None:
        return 1.0
    if len(gt) == 0:
        return 1.0 if pred else 0.0
    return levenshtein(pred or "", gt or "") / max(1, len(gt))

def normalize_pred_id(pred_id, regset):
    if not pred_id:
        return "UNID"
    if isinstance(pred_id, str) and pred_id.startswith("UNREG:"):
        digits = "".join(re.findall(r"\d+", pred_id))
        if digits and digits in regset:
            return digits
    digits = "".join(re.findall(r"\d+", str(pred_id)))
    if digits and digits in regset:
        return digits
    return str(pred_id)

def id_matches(pred_id, gt_code, regset, cer_thresh=0.25):
    if pred_id == gt_code:
        return True
    p = str(pred_id)
    g = str(gt_code)
    pdigits = "".join(re.findall(r"\d+", p))
    if pdigits == g:
        return True
    if pdigits and cer(pdigits, g) <= cer_thresh:
        return True
    if cer(p, g) <= cer_thresh:
        return True
    return False

def build_default_ground_truth():
    sites = {}
    sites['siteA'] = {
        'angles': ['near','between','far'],
        'workers': {
            "2":    {"helmet": True, "vest": True,  "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "751":  {"helmet": True, "vest": True,  "left_glove": False,"right_glove": False,"left_shoe": False,"right_shoe": False},
            "81":   {"helmet": False,"vest": False, "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "65":   {"helmet": True, "vest": False, "left_glove": True, "right_glove": False,"left_shoe": False,"right_shoe": False},
            "76":   {"helmet": False,"vest": True,  "left_glove": False,"right_glove": False,"left_shoe": True, "right_shoe": True},
        }
    }
    sites['siteB'] = dict(sites['siteA'])
    sites['siteC'] = {
        'angles': ['all'],
        'workers': {
            "23":  {"helmet": True, "vest": True,  "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "6":   {"helmet": True, "vest": False, "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "3":   {"helmet": True, "vest": True,  "left_glove": False,"right_glove": False,"left_shoe": True, "right_shoe": True},
            "698": {"helmet": True, "vest": True,  "left_glove": True, "right_glove": True, "left_shoe": False,"right_shoe": False},
        }
    }
    sDEF_workers = {
            "325": {"helmet": True, "vest": True, "left_glove": True, "right_glove": True, "left_shoe": True, "right_shoe": True},
            "23":  {"helmet": True, "vest": False,"left_glove": False,"right_glove": False,"left_shoe": True, "right_shoe": True},
            "6":   {"helmet": False,"vest": True, "left_glove": True, "right_glove": True, "left_shoe": False,"right_shoe": False},
            "201": {"helmet": False,"vest": False,"left_glove": False,"right_glove": False,"left_shoe": False,"right_shoe": False},
            "3":   {"helmet": True, "vest": False,"left_glove": False,"right_glove": False,"left_shoe": False,"right_shoe": False, "improper_helmet": True, "improper_left_glove": True},
    }
    for s in ['siteD','siteE','siteF']:
        sites[s] = {'angles':['near','between','far'], 'workers': dict(sDEF_workers)}
    for sname, sdata in sites.items():
        for wcode, ppe in sdata['workers'].items():
            vset = set()
            if not ppe.get('helmet', False):
                vset.add("NO HELMET")
            else:
                if ppe.get('improper_helmet', False):
                    vset.add("IMPROPER HELMET")
            if not ppe.get('vest', False):
                vset.add("NO VEST")
            if not ppe.get('left_glove', False):
                vset.add("NO LEFT GLOVE")
            else:
                if ppe.get('improper_left_glove', False):
                    vset.add("IMPROPER LEFT GLOVE")
            if not ppe.get('right_glove', False):
                vset.add("NO RIGHT GLOVE")
            else:
                if ppe.get('improper_right_glove', False):
                    vset.add("IMPROPER RIGHT GLOVE")
            if not ppe.get('left_shoe', False):
                vset.add("NO LEFT SHOE")
            if not ppe.get('right_shoe', False):
                vset.add("NO RIGHT SHOE")
            sdata['workers'][wcode]['violations'] = vset
    return sites

def run_inference_on_video(video_path, site_name, angle_name, ly, pose_instance, frame_step=30, max_frames=None, regset=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(frame_step))
    results = []
    frame_idx = 0
    pbar = tqdm(total=frame_count, desc=f"{site_name}/{angle_name}", unit="f")
    ocr_reader = None
    try:
        ocr_reader = get_ocr_reader()
    except Exception:
        ocr_reader = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx % step) == 0:
            try:
                person_boxes = ly.predict_person_boxes(frame)
            except Exception:
                person_boxes = []
            try:
                ppe_out = ly.predict_ppe_outputs(frame)
            except Exception:
                ppe_out = {}
            triton_outputs = ppe_out
            res = process_frame(frame.copy(), triton_client=None, triton_model_name=None, input_name=None, output_names=None, triton_outputs=triton_outputs, ocr_reader=ocr_reader, regset=(regset or set()), pose_instance=pose_instance, person_boxes=person_boxes)
            people = res.get("people", [])
            for idx_p, p in enumerate(people):
                bbox = p.get("bbox")
                ocr_raw = None
                ocr_digits = None
                ocr_conf = None
                if ocr_reader is not None and bbox and len(bbox) >= 4:
                    try:
                        x1, y1, x2, y2 = map(int, bbox)
                        x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            if crop is not None and crop.size > 0:
                                txts = []
                                try:
                                    txts = ocr_reader.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), detail=1)
                                except Exception:
                                    try:
                                        txts = ocr_reader.readtext(crop, detail=1)
                                    except Exception:
                                        txts = []
                                best_conf = -1.0
                                best_txt = ""
                                for it in txts:
                                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                                        if len(it) == 3:
                                            txt = str(it[1]).strip(); conf = float(it[2])
                                        else:
                                            txt = str(it[1]).strip() if len(it) > 1 else str(it[0]); conf = float(it[2]) if len(it) > 2 else 0.0
                                    else:
                                        txt = str(it)
                                        conf = 0.0
                                    if conf > best_conf:
                                        best_conf = conf; best_txt = txt
                                if best_txt:
                                    ocr_raw = best_txt
                                    ocr_digits = "".join(re.findall(r"\d+", best_txt))
                                    ocr_conf = float(best_conf) if best_conf is not None else None
                    except Exception:
                        ocr_raw = None
                        ocr_digits = None
                        ocr_conf = None
                p['ocr_raw'] = ocr_raw
                p['ocr_digits'] = ocr_digits
                p['ocr_conf'] = ocr_conf
            meta = {"site": site_name, "angle": angle_name, "frame_idx": frame_idx, "timestamp": time.time(), "fps": fps}
            results.append({"meta": meta, "people": people, "boxes_by_class": res.get("boxes_by_class", {})})
        frame_idx += 1
        pbar.update(1)
        if max_frames and frame_idx >= max_frames:
            break
    pbar.close()
    cap.release()
    return results

def compute_frame_level_metrics(predictions, ground_truth_sites):
    violation_types = ["NO HELMET","NO VEST","NO LEFT GLOVE","NO RIGHT GLOVE","NO LEFT SHOE","NO RIGHT SHOE","IMPROPER HELMET","IMPROPER LEFT GLOVE","IMPROPER RIGHT GLOVE"]
    counts = {vt: {"tp":0,"fp":0,"fn":0} for vt in violation_types}
    id_stats = {"tp":0,"fp":0,"fn":0,"total_present":0}
    cer_list = []
    for rec in predictions:
        site = rec["meta"]["site"]
        if site not in ground_truth_sites:
            continue
        gt_workers = ground_truth_sites[site]['workers']
        regset = set(gt_workers.keys())
        predicted_list = []
        for i, p in enumerate(rec.get("people", [])):
            raw_id = p.get("id")
            if (not raw_id or raw_id == "UNID") and p.get("ocr_digits"):
                raw_id = f"UNREG:{p.get('ocr_digits')}"
            norm_id = normalize_pred_id(raw_id, regset)
            pred_v = set(p.get("violations") or [])
            predicted_list.append({"idx": i, "pid": norm_id, "violations": pred_v, "ocr_digits": p.get("ocr_digits"), "ocr_raw": p.get("ocr_raw"), "used": False})
        for wcode, winfo in gt_workers.items():
            gt_v = winfo.get("violations", set())
            id_stats["total_present"] += 1
            matched_idx = None
            matched_pred = None
            for pred in predicted_list:
                if pred["used"]:
                    continue
                if id_matches(pred["pid"], wcode, regset):
                    matched_idx = pred["idx"]
                    pred["used"] = True
                    matched_pred = pred
                    break
            if matched_idx is None:
                for vt in violation_types:
                    if vt in gt_v:
                        counts[vt]["fn"] += 1
                id_stats["fn"] += 1
            else:
                id_stats["tp"] += 1
                pred_v = matched_pred["violations"]
                for vt in violation_types:
                    p_has = vt in pred_v
                    g_has = vt in gt_v
                    if p_has and g_has:
                        counts[vt]["tp"] += 1
                    elif p_has and not g_has:
                        counts[vt]["fp"] += 1
                    elif not p_has and g_has:
                        counts[vt]["fn"] += 1
                pid_used = matched_pred["pid"]
                if isinstance(pid_used, str) and pid_used != wcode and pid_used != "UNID":
                    cer_list.append(cer(pid_used, wcode))
                od = matched_pred.get("ocr_digits")
                if od and od != wcode:
                    cer_list.append(cer(od, wcode))
        for pred in predicted_list:
            if not pred["used"]:
                pid = pred["pid"]
                if pid and pid != "UNID" and pid != "":
                    id_stats["fp"] += 1
                    for vt in violation_types:
                        if vt in pred["violations"]:
                            counts[vt]["fp"] += 1
    metrics = {}
    for vt, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        metrics[vt] = {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1}
    id_prec = id_stats["tp"] / (id_stats["tp"] + id_stats["fp"]) if (id_stats["tp"]+id_stats["fp"])>0 else 0.0
    id_rec = id_stats["tp"] / (id_stats["total_present"]) if id_stats["total_present"]>0 else 0.0
    metrics["ID_MATCH"] = {"tp":id_stats["tp"], "fn":id_stats["fn"], "precision": id_prec, "recall": id_rec}
    metrics["CER_mean"] = (sum(cer_list)/len(cer_list)) if cer_list else 0.0
    return metrics, counts

def frames_to_events(predictions, ground_truth_sites, gap_tolerance=2):
    pred_events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gt_events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in predictions:
        site = rec["meta"]["site"]
        frame = rec["meta"]["frame_idx"]
        regset = set(ground_truth_sites.get(site, {}).get('workers', {}).keys())
        for i, p in enumerate(rec["people"]):
            pid_raw = p.get("id") or "UNID"
            if pid_raw == "UNID":
                od = p.get("ocr_digits")
                if od:
                    pid_raw = f"UNREG:{od}"
                else:
                    pid_raw = f"UNID_{i}"
            pid = normalize_pred_id(pid_raw, regset)
            for vt in p.get("violations") or []:
                pred_events[site][pid][vt].append(frame)
    def collapse(frames_list):
        if not frames_list:
            return []
        frames_list = sorted(set(frames_list))
        intervals = []
        start = frames_list[0]
        prev = start
        for f in frames_list[1:]:
            if f - prev <= gap_tolerance:
                prev = f
                continue
            else:
                intervals.append((start, prev))
                start = f
                prev = f
        intervals.append((start, prev))
        return intervals
    pred_intervals = defaultdict(lambda: defaultdict(dict))
    for site, workers in pred_events.items():
        for pid, vmap in workers.items():
            for vt, frames_list in vmap.items():
                if not frames_list:
                    pred_intervals[site][pid][vt] = []
                else:
                    if all(isinstance(x, int) for x in frames_list):
                        pred_intervals[site][pid][vt] = collapse(frames_list)
                    else:
                        converted = []
                        for it in frames_list:
                            if isinstance(it, (list, tuple)) and len(it) == 2:
                                converted.append((int(it[0]), int(it[1])))
                            elif isinstance(it, int):
                                converted.append((it, it))
                        if converted and all(isinstance(x, tuple) and len(x) == 2 for x in converted):
                            pred_intervals[site][pid][vt] = merge_intervals(converted)
                        else:
                            pred_intervals[site][pid][vt] = collapse([int(x) for x in frames_list if isinstance(x, int)])
    frames_per_site = defaultdict(list)
    for rec in predictions:
        frames_per_site[rec["meta"]["site"]].append(rec["meta"]["frame_idx"])
    for site, sdata in ground_truth_sites.items():
        all_frames = sorted(set(frames_per_site.get(site, [])))
        if not all_frames:
            continue
        gs = min(all_frames); ge = max(all_frames)
        for wcode in sdata['workers'].keys():
            for vt in sdata['workers'][wcode]['violations']:
                gt_events[site][wcode][vt].append((gs, ge))
    return pred_intervals, gt_events

def match_events(pred_intervals, gt_events, iou_threshold=0.5, overlap_min_frames=1):
    metrics = {}
    for site, workers in gt_events.items():
        site_metrics = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
        preds_for_site = pred_intervals.get(site, {})
        regset = set(workers.keys())
        for wcode, vmap in workers.items():
            for vt, gt_list in vmap.items():
                for gt_int in gt_list:
                    gs, ge = gt_int
                    matched = False
                    for pred_id, pvmap in preds_for_site.items():
                        if not id_matches(pred_id, wcode, regset):
                            continue
                        pred_list = pvmap.get(vt, [])
                        for pi in pred_list:
                            ps, pe = pi
                            inter = max(0, min(ge, pe) - max(gs, ps) + 1)
                            union = (ge - gs + 1) + (pe - ps + 1) - inter
                            tiou = inter / union if union>0 else 0.0
                            if tiou >= iou_threshold or inter >= overlap_min_frames:
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        site_metrics[vt]["tp"] += 1
                    else:
                        site_metrics[vt]["fn"] += 1
        for pred_id, vmap in preds_for_site.items():
            for vt, pred_list in vmap.items():
                for pi in pred_list:
                    ps, pe = pi
                    matched_any = False
                    for gt_w in workers.keys():
                        if not id_matches(pred_id, gt_w, regset):
                            continue
                        gt_for_worker = gt_events.get(site, {}).get(gt_w, {}).get(vt, [])
                        for gs, ge in gt_for_worker:
                            inter = max(0, min(ge, pe) - max(gs, ps) + 1)
                            union = (ge - gs + 1) + (pe - ps + 1) - inter
                            if union > 0 and (inter / union >= iou_threshold or inter >= overlap_min_frames):
                                matched_any = True
                                break
                        if matched_any:
                            break
                    if not matched_any:
                        site_metrics[vt]["fp"] += 1
        metrics[site] = site_metrics
    site_scores = {}
    for site, vtmap in metrics.items():
        site_scores[site] = {}
        for vt, c in vtmap.items():
            tp, fp, fn = c['tp'], c['fp'], c['fn']
            prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
            rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
            site_scores[site][vt] = {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1}
    return site_scores

def merge_intervals(list_of_intervals):
    if not list_of_intervals:
        return []
    normalized = []
    for item in list_of_intervals:
        if isinstance(item, int):
            normalized.append((item, item))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                normalized.append((int(item[0]), int(item[1])))
            except Exception:
                pass
        else:
            pass
    if not normalized:
        return []
    allints = sorted(normalized, key=lambda x: x[0])
    merged = []
    s, e = allints[0]
    for a, b in allints[1:]:
        if a <= e + 1:
            e = max(e, b)
        else:
            merged.append((s, e))
            s, e = a, b
    merged.append((s, e))
    return merged

def temporal_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0

def site_level_event_eval(pred_intervals, gt_events, tiou_thresh=0.5):
    site_scores = {}
    for site in set(list(pred_intervals.keys()) + list(gt_events.keys())):
        vmap_pred = defaultdict(list)
        vmap_gt = defaultdict(list)
        preds_for_site = pred_intervals.get(site, {})
        for pid, vtmap in preds_for_site.items():
            for vt, ints in vtmap.items():
                if ints:
                    if all(isinstance(x, int) for x in ints):
                        vmap_pred[vt].extend(merge_intervals(ints))
                    else:
                        vmap_pred[vt].extend(ints)
        gts_for_site = gt_events.get(site, {})
        for wcode, vtmap in gts_for_site.items():
            for vt, ints in vtmap.items():
                vmap_gt[vt].extend(ints)
        site_metrics = {}
        for vt in set(list(vmap_pred.keys()) + list(vmap_gt.keys())):
            pred_union = merge_intervals(vmap_pred.get(vt, []))
            gt_union = merge_intervals(vmap_gt.get(vt, []))
            tp = 0; fp = 0; fn = 0
            for g in gt_union:
                matched = any(max(0, min(g[1], p[1]) - max(g[0], p[0]) + 1) > 0 for p in pred_union)
                if matched:
                    tp += 1
                else:
                    fn += 1
            for p in pred_union:
                matched = any(max(0, min(g[1], p[1]) - max(g[0], p[0]) + 1) > 0 for g in gt_union)
                if not matched:
                    fp += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            site_metrics[vt] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
        site_scores[site] = site_metrics
    return site_scores

def compute_cer_all(predictions, ground_truth_sites):
    per_site = defaultdict(list)
    for rec in predictions:
        site = rec["meta"]["site"]
        gt_codes = set(ground_truth_sites.get(site, {}).get("workers", {}).keys())
        for p in rec.get("people", []):
            od = p.get("ocr_digits")
            pid = p.get("id") or ""
            candidates = []
            if od:
                candidates.append(od)
            digits_from_id = "".join(re.findall(r"\d+", str(pid)))
            if digits_from_id:
                candidates.append(digits_from_id)
            for cand in candidates:
                if not cand:
                    continue
                if not gt_codes:
                    continue
                best = min((cer(cand, g), g) for g in gt_codes)[0]
                per_site[site].append(best)
    site_mean = {s: (float(np.mean(vals)) if vals else 0.0) for s, vals in per_site.items()}
    all_vals = [v for vals in per_site.values() for v in vals]
    overall = float(np.mean(all_vals)) if all_vals else 0.0
    return overall, site_mean

def bar_plot_metrics(metrics_dict, title, out_png):
    if not metrics_dict:
        return
    labels = list(metrics_dict.keys())
    precisions = [metrics_dict[l]["precision"] for l in labels]
    recalls = [metrics_dict[l]["recall"] for l in labels]
    f1s = [metrics_dict[l]["f1"] for l in labels]
    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(max(8, len(labels)*0.6), 5))
    plt.bar(x - w, precisions, width=w, label="Precision")
    plt.bar(x, recalls, width=w, label="Recall")
    plt.bar(x + w, f1s, width=w, label="F1")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylim(0,1)
    plt.ylabel("score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def make_json_friendly(obj):
    if isinstance(obj, dict):
        return {k: make_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_friendly(x) for x in obj]
    if isinstance(obj, set):
        return [make_json_friendly(x) for x in list(obj)]
    return obj

def main(args):
    ground_truth = build_default_ground_truth()
    ly = _init_local_yolo()
    if ly is None:
        print("Local YOLO wrapper not available (models not found). Exiting.")
        return
    pose_instance = init_pose()
    videos = {}
    for root, _, files in os.walk(args.videos_dir):
        for f in files:
            if f.lower().endswith(('.mp4','.mov','.mkv','.avi')):
                videos[f] = os.path.join(root, f)
    mapping = {
        'siteA_near.mp4': ('siteA','near'),
        'siteA_between.mp4': ('siteA','between'),
        'siteA_far.mp4': ('siteA','far'),
        'siteB_near.mp4': ('siteB','near'),
        'siteB_between.mp4': ('siteB','between'),
        'siteB_far.mp4': ('siteB','far'),
        'siteC_all.mp4': ('siteC','all'),
        'siteD_near.mp4': ('siteD','near'),
        'siteD_between.mp4': ('siteD','between'),
        'siteD_far.mp4': ('siteD','far'),
        'siteE_near.mp4': ('siteE','near'),
        'siteE_between.mp4': ('siteE','between'),
        'siteE_far.mp4': ('siteE','far'),
        'siteF_near.mp4': ('siteF','near'),
        'siteF_between.mp4': ('siteF','between'),
        'siteF_far.mp4': ('siteF','far'),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    all_predictions = []
    for fname, (site, angle) in mapping.items():
        path = videos.get(fname)
        if path is None:
            print(f"WARNING: expected file {fname} not found in {args.videos_dir}, skipping.")
            continue
        print(f"Processing {fname} -> site={site} angle={angle}")
        gt_worker_codes = set(ground_truth[site]['workers'].keys())
        preds = run_inference_on_video(path, site, angle, ly, pose_instance, frame_step=args.frame_step, max_frames=args.max_frames, regset=gt_worker_codes)
        outjson = os.path.join(args.out_dir, f"predictions_{site}_{angle}.json")
        with open(outjson, "w") as fh:
            json.dump(make_json_friendly(preds), fh, indent=2)
        all_predictions.extend(preds)
        frame_metrics_vid, counts_vid = compute_frame_level_metrics(preds, ground_truth)
        with open(os.path.join(args.out_dir, f"frame_level_metrics_{site}_{angle}.json"), "w") as fh:
            json.dump(make_json_friendly(frame_metrics_vid), fh, indent=2)
        vt_metrics_vid = {k: v for k, v in frame_metrics_vid.items() if k != "ID_MATCH" and k != "CER_mean"}
        if vt_metrics_vid:
            bar_plot_metrics(vt_metrics_vid, f"Frame-level violation metrics {site}-{angle}", os.path.join(args.out_dir, f"frame_level_violations_{site}_{angle}.png"))
        df_rows = []
        for vt, m in vt_metrics_vid.items():
            df_rows.append({"violation":vt, "precision":m["precision"], "recall":m["recall"], "f1":m["f1"], "tp":m["tp"], "fp":m["fp"], "fn":m["fn"]})
        df_vid = pd.DataFrame(df_rows)
        df_vid.to_csv(os.path.join(args.out_dir, f"frame_violation_summary_{site}_{angle}.csv"), index=False)
        pred_intervals_vid, gt_events_vid = frames_to_events(preds, ground_truth, gap_tolerance=args.event_gap)
        event_scores_vid = match_events(pred_intervals_vid, gt_events_vid, iou_threshold=args.event_iou, overlap_min_frames=1)
        with open(os.path.join(args.out_dir, f"event_level_metrics_{site}_{angle}.json"), "w") as fh:
            json.dump(make_json_friendly(event_scores_vid), fh, indent=2)
        agg_event_vid = {}
        for sname, vtmap in event_scores_vid.items():
            for vt, m in vtmap.items():
                agg_event_vid.setdefault(vt, {"precision":[],"recall":[],"f1":[]})
                agg_event_vid[vt]["precision"].append(m["precision"])
                agg_event_vid[vt]["recall"].append(m["recall"])
                agg_event_vid[vt]["f1"].append(m["f1"])
        agg_event_avg_vid = {vt: {"precision": float(np.mean(vals["precision"])) if vals["precision"] else 0.0, "recall": float(np.mean(vals["recall"])) if vals["recall"] else 0.0, "f1": float(np.mean(vals["f1"])) if vals["f1"] else 0.0} for vt, vals in agg_event_vid.items()}
        if agg_event_avg_vid:
            bar_plot_metrics(agg_event_avg_vid, f"Event-level violation metrics {site}-{angle}", os.path.join(args.out_dir, f"event_level_violations_{site}_{angle}.png"))
        id_metrics_vid = {"ID_MATCH": frame_metrics_vid.get("ID_MATCH", {"precision":0,"recall":0})}
        plt.figure(figsize=(4,4))
        plt.bar(['precision','recall'], [id_metrics_vid["ID_MATCH"]["precision"], id_metrics_vid["ID_MATCH"]["recall"]])
        plt.ylim(0,1)
        plt.title(f"ID matching (frame-level) {site}-{angle}")
        plt.savefig(os.path.join(args.out_dir, f"id_matching_{site}_{angle}.png"))
        plt.close()
        overall_cer_vid = frame_metrics_vid.get("CER_mean", 0.0)
        with open(os.path.join(args.out_dir, f"cer_{site}_{angle}.txt"), "w") as fh:
            fh.write(f"mean_CER:{overall_cer_vid}\n")
    gt_json = os.path.join(args.out_dir, "ground_truth_generated.json")
    with open(gt_json, "w") as fh:
        json.dump(make_json_friendly(ground_truth), fh, indent=2)
    frame_metrics, counts = compute_frame_level_metrics(all_predictions, ground_truth)
    with open(os.path.join(args.out_dir, "frame_level_metrics.json"), "w") as fh:
        json.dump(make_json_friendly(frame_metrics), fh, indent=2)
    pred_intervals, gt_events = frames_to_events(all_predictions, ground_truth, gap_tolerance=args.event_gap)
    event_scores = match_events(pred_intervals, gt_events, iou_threshold=args.event_iou, overlap_min_frames=1)
    with open(os.path.join(args.out_dir, "event_level_metrics.json"), "w") as fh:
        json.dump(make_json_friendly(event_scores), fh, indent=2)
    site_level_scores = site_level_event_eval(pred_intervals, gt_events, tiou_thresh=args.event_iou)
    with open(os.path.join(args.out_dir, "event_level_metrics_sitelevel.json"), "w") as fh:
        json.dump(make_json_friendly(site_level_scores), fh, indent=2)
    vt_metrics = {k: v for k, v in frame_metrics.items() if k != "ID_MATCH" and k != "CER_mean"}
    bar_plot_metrics(vt_metrics, "Frame-level violation metrics", os.path.join(args.out_dir, "frame_level_violations.png"))
    agg_event = {}
    for site, vtmap in event_scores.items():
        for vt, m in vtmap.items():
            agg_event.setdefault(vt, {"precision":[],"recall":[],"f1":[]})
            agg_event[vt]["precision"].append(m["precision"])
            agg_event[vt]["recall"].append(m["recall"])
            agg_event[vt]["f1"].append(m["f1"])
    agg_event_avg = {vt: {"precision": float(np.mean(vals["precision"])) if vals["precision"] else 0.0, "recall": float(np.mean(vals["recall"])) if vals["recall"] else 0.0, "f1": float(np.mean(vals["f1"])) if vals["f1"] else 0.0} for vt, vals in agg_event.items()}
    bar_plot_metrics(agg_event_avg, "Event-level violation metrics (avg across sites)", os.path.join(args.out_dir, "event_level_violations.png"))
    agg_site_event = {}
    for site, vtmap in site_level_scores.items():
        for vt, m in vtmap.items():
            agg_site_event.setdefault(vt, {"precision":[],"recall":[],"f1":[]})
            agg_site_event[vt]["precision"].append(m["precision"])
            agg_site_event[vt]["recall"].append(m["recall"])
            agg_site_event[vt]["f1"].append(m["f1"])
    agg_site_event_avg = {vt: {"precision": float(np.mean(vals["precision"])) if vals["precision"] else 0.0, "recall": float(np.mean(vals["recall"])) if vals["recall"] else 0.0, "f1": float(np.mean(vals["f1"])) if vals["f1"] else 0.0} for vt, vals in agg_site_event.items()}
    if agg_site_event_avg:
        bar_plot_metrics(agg_site_event_avg, "Site-level event metrics (ID-agnostic)", os.path.join(args.out_dir, "event_level_violations_sitelevel.png"))
    id_metrics = {"ID_MATCH": frame_metrics.get("ID_MATCH", {"precision":0,"recall":0})}
    plt.figure(figsize=(4,4))
    plt.bar(['precision','recall'], [id_metrics["ID_MATCH"]["precision"], id_metrics["ID_MATCH"]["recall"]])
    plt.ylim(0,1)
    plt.title("ID matching (frame-level)")
    plt.savefig(os.path.join(args.out_dir, "id_matching.png"))
    plt.close()
    overall_cer, cer_by_site = compute_cer_all(all_predictions, ground_truth)
    with open(os.path.join(args.out_dir, "cer_summary.json"), "w") as fh:
        json.dump(make_json_friendly({"overall_CER": overall_cer, "by_site": cer_by_site}), fh, indent=2)
    with open(os.path.join(args.out_dir, "cer.txt"), "w") as fh:
        fh.write(f"mean_CER_overall:{overall_cer}\n")
        for s, v in cer_by_site.items():
            fh.write(f"{s}:{v}\n")
    rows = []
    for vt, m in vt_metrics.items():
        rows.append({"violation":vt, "precision":m["precision"], "recall":m["recall"], "f1":m["f1"], "tp":m["tp"], "fp":m["fp"], "fn":m["fn"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "frame_violation_summary.csv"), index=False)
    print("Evaluation results saved to", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="./videos", help="directory with site videos")
    ap.add_argument("--out_dir", default="./eval_results", help="output directory")
    ap.add_argument("--frame_step", type=int, default=30, help="process every Nth frame (default=30)")
    ap.add_argument("--max_frames", type=int, default=None, help="max frames per video (for debug)")
    ap.add_argument("--event_gap", type=int, default=2, help="max gap (frames) to merge events")
    ap.add_argument("--event_iou", type=float, default=0.5, help="temporal IoU threshold for event match")
    args = ap.parse_args()
    main(args)
